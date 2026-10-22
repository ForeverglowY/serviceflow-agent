"""需要已迁移并初始化数据的 PostgreSQL，不调用大模型。"""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from serviceflow.db.database import engine
from serviceflow.db.models import TicketConfirmationTable, TicketTable
from serviceflow.models import TicketConfirmationCreate, TicketCreate
from serviceflow.ticket.service import create_ticket


def test_concurrent_submissions_recover_unique_constraint_conflict(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from sqlalchemy.exc import IntegrityError
    from serviceflow.ticket import service

    submission_id = uuid4().hex
    barrier = Barrier(2, timeout=10)
    conflicts = []

    class RacingSession(Session):
        def scalar(self, statement, *args, **kwargs):
            result = super().scalar(statement, *args, **kwargs)
            # 强制两个事务首次查重都得到空结果，再同时进入创建流程。
            if not self.info.get("first_lookup_done"):
                self.info["first_lookup_done"] = True
                if result is None:
                    barrier.wait()
            return result

        def flush(self, *args, **kwargs):
            try:
                return super().flush(*args, **kwargs)
            except IntegrityError:
                conflicts.append(True)
                raise

    monkeypatch.setattr(service, "Session", RacingSession)
    request = TicketCreate(
        order_id="20260721001", user_id="USER-CONCURRENCY-TEST",
        issue_type="product_fault", description="并发集成测试：左耳无声",
    )
    confirmation = TicketConfirmationCreate(
        confirmed_by=request.user_id, confirmed_at=datetime.now(timezone.utc),
        confirmation_message="确认", application_snapshot={"order_id": request.order_id},
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(service.create_ticket, request, confirmation,
                                   submission_id=submission_id) for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]
        assert results[0].ticket_id == results[1].ticket_id
        assert len(conflicts) == 1
        with Session(engine) as session:
            assert session.scalar(select(func.count()).select_from(TicketTable).where(
                TicketTable.submission_id == submission_id,
            )) == 1
            assert session.scalar(select(func.count()).select_from(TicketConfirmationTable).where(
                TicketConfirmationTable.ticket_id == results[0].ticket_id,
            )) == 1
    finally:
        with Session(engine) as session, session.begin():
            ids = select(TicketTable.ticket_id).where(
                TicketTable.submission_id == submission_id,
                TicketTable.user_id == request.user_id,
            )
            session.execute(delete(TicketConfirmationTable).where(
                TicketConfirmationTable.ticket_id.in_(ids),
            ))
            session.execute(delete(TicketTable).where(
                TicketTable.submission_id == submission_id,
                TicketTable.user_id == request.user_id,
            ))


def test_repeated_submission_persists_one_ticket_and_original_confirmation():
    submission_id = uuid4().hex
    request = TicketCreate(
        order_id="20260721001",
        user_id="USER-IDEMPOTENCY-TEST",
        issue_type="product_fault",
        description="集成测试：左耳无声，希望换货",
    )
    confirmation = TicketConfirmationCreate(
        confirmed_by=request.user_id,
        confirmed_at=datetime.now(timezone.utc),
        confirmation_message="确认",
        application_snapshot={"order_id": request.order_id, "requested_action": "换货"},
    )
    try:
        first = create_ticket(request, confirmation, submission_id=submission_id)
        retry_confirmation = confirmation.model_copy(update={
            "confirmed_at": datetime.now(timezone.utc),
            "confirmation_message": "重试确认",
        })
        second = create_ticket(request, retry_confirmation, submission_id=submission_id)

        assert first.ticket_id == second.ticket_id
        with Session(engine) as session:
            ticket_count = session.scalar(select(func.count()).select_from(TicketTable).where(
                TicketTable.submission_id == submission_id,
            ))
            confirmation_count = session.scalar(
                select(func.count()).select_from(TicketConfirmationTable).where(
                    TicketConfirmationTable.ticket_id == first.ticket_id,
                )
            )
            saved_confirmation = session.get(TicketConfirmationTable, first.ticket_id)
            assert ticket_count == 1
            assert confirmation_count == 1
            assert saved_confirmation.confirmation_message == "确认"
            assert saved_confirmation.confirmed_at == confirmation.confirmed_at
            assert saved_confirmation.application_snapshot == confirmation.application_snapshot
    finally:
        # 只清理本次随机提交 ID 所创建的测试记录，不修改订单或其他工单。
        with Session(engine) as session, session.begin():
            ticket_ids = select(TicketTable.ticket_id).where(
                TicketTable.submission_id == submission_id,
                TicketTable.user_id == request.user_id,
            )
            session.execute(delete(TicketConfirmationTable).where(
                TicketConfirmationTable.ticket_id.in_(ticket_ids),
            ))
            session.execute(delete(TicketTable).where(
                TicketTable.submission_id == submission_id,
                TicketTable.user_id == request.user_id,
            ))
