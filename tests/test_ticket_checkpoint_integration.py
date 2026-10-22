"""真实 PostgreSQL 存档与业务写入；仅模拟模型输出和首次人工登记故障。"""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from serviceflow.agent import ticket_collection as collection
from serviceflow.db.database import engine
from serviceflow.db.models import TicketConfirmationTable, TicketReviewTaskTable, TicketTable
from serviceflow.settings import settings
from serviceflow.ticket.review import enqueue_manual_review
from serviceflow.ticket.rules import TicketEligibilityDecision, TicketEligibilityResult


def test_postgres_checkpoint_survives_rebuilding_graph_and_resumes_review(monkeypatch):
    thread_id = "checkpoint-test-" + uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    ticket_id = None
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=collection.TicketExtraction(
        order_id="20260721001", issue_description="集成测试：左耳无声", requested_action="换货",
    )))
    monkeypatch.setattr(collection, "assess_return_request", Mock(return_value=TicketEligibilityResult(
        decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW, reason="测试：需要人工检测",
    )))
    create = Mock(wraps=collection.create_ticket_service)
    monkeypatch.setattr(collection, "create_ticket_service", create)
    failed_enqueue = Mock(side_effect=RuntimeError("测试：人工登记暂时失败"))
    monkeypatch.setattr(collection, "enqueue_manual_review", failed_enqueue)

    try:
        with PostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
            saver.setup()
            first_graph = collection.build_ticket_collection_graph(saver)
            collection.run_ticket_collection_turn("申请换货", first_graph, thread_id)
            with pytest.raises(RuntimeError, match="人工登记暂时失败"):
                collection.run_ticket_collection_turn("确认", first_graph, thread_id)
            snapshot = first_graph.get_state(config)
            ticket_id = snapshot.values["ticket_id"]
            submission_id = snapshot.values["submission_id"]
            assert snapshot.next == ("enqueue_manual_review",)
            assert ticket_id and submission_id

        # 关闭旧连接，用新的 saver 和图读取数据库存档，而非复用内存对象。
        with PostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
            restored_graph = collection.build_ticket_collection_graph(saver)
            restored = restored_graph.get_state(config)
            assert restored.values["ticket_id"] == ticket_id
            assert restored.values["submission_id"] == submission_id
            with pytest.raises(collection.TicketRecoveryRequiredError):
                collection.run_ticket_collection_turn("不登记了", restored_graph, thread_id)
            assert failed_enqueue.call_count == 1

            retry_enqueue = Mock(wraps=enqueue_manual_review)
            monkeypatch.setattr(collection, "enqueue_manual_review", retry_enqueue)
            result = collection.resume_ticket_collection(restored_graph, thread_id)
            create.assert_called_once()
            retry_enqueue.assert_called_once_with(ticket_id)
            assert result["ticket_id"] == ticket_id
            assert result["submission_id"] == submission_id
            assert result["confirmation_message"] == "确认"
            assert restored_graph.get_state(config).next == ()

        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(TicketTable).where(
                TicketTable.submission_id == submission_id,
            )) == 1
            assert session.scalar(select(func.count()).select_from(TicketConfirmationTable).where(
                TicketConfirmationTable.ticket_id == ticket_id,
            )) == 1
            assert session.scalar(select(func.count()).select_from(TicketReviewTaskTable).where(
                TicketReviewTaskTable.ticket_id == ticket_id,
            )) == 1
    finally:
        # 存档只清理本次随机线程；业务记录只清理该线程生成的工单。
        with PostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
            if ticket_id is None:
                ticket_id = saver.get_tuple(config).checkpoint["channel_values"].get("ticket_id") if saver.get_tuple(
                    config) else None
            saver.delete_thread(thread_id)
        if ticket_id:
            with Session(engine) as session, session.begin():
                for table in (TicketReviewTaskTable, TicketConfirmationTable, TicketTable):
                    session.execute(delete(table).where(table.ticket_id == ticket_id))
