from datetime import date, datetime, timezone
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from serviceflow.agent import ticket_collection as collection
from serviceflow.agent.ticket_collection import TicketExtraction, merge_ticket_extraction
from serviceflow.llm.service import LLMServiceError
from serviceflow.models import Ticket, TicketStatus
from serviceflow.ticket.rules import TicketEligibilityDecision, TicketEligibilityResult
from serviceflow.ticket.service import TicketEligibilityRejectedError


@pytest.fixture(autouse=True)
def stub_return_assessment(monkeypatch):
    monkeypatch.setattr(collection, "enqueue_manual_review", Mock())
    monkeypatch.setattr(collection, "assess_return_request", Mock(return_value=TicketEligibilityResult(
        decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
        reason="疑似质量问题，需要检测",
    )))


def test_assess_return_request_node_saves_rule_result(monkeypatch):
    assess = Mock(return_value=TicketEligibilityResult(
        decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
        reason="疑似质量问题，需要检测",
    ))
    monkeypatch.setattr(collection, "assess_return_request", assess)
    state = {
        "user_message": "换货",
        "order_id": "20260721001",
        "issue_description": "左耳无声",
        "requested_action": "换货",
        "missing_fields": [],
        "answer": None,
        "status": "collecting",
        "decision": None,
        "reason": None,
    }

    result = collection.assess_return_request_node(state)

    assert result == {
        "decision": "need_manual_review",
        "reason": "疑似质量问题，需要检测",
    }
    assess.assert_called_once_with(
        order_id="20260721001",
        issue_description="左耳无声",
        requested_action="换货",
        requested_at=date.today(),
    )


def test_assess_return_request_node_rejects_blank_order_id(monkeypatch):
    assess = Mock()
    monkeypatch.setattr(collection, "assess_return_request", assess)
    state = {
        "user_message": "换货",
        "order_id": "   ",
        "issue_description": "左耳无声",
        "requested_action": "换货",
        "missing_fields": [],
        "answer": None,
        "status": "collecting",
        "decision": None,
        "reason": None,
    }

    with pytest.raises(ValueError, match="信息不完整"):
        collection.assess_return_request_node(state)

    assess.assert_not_called()


@pytest.mark.parametrize(
    "decision,reason",
    [
        (None, "有说明"),
        ("unexpected", "有说明"),
        ("allow", None),
        ("reject", "   "),
    ],
)
def test_create_ticket_node_rejects_missing_or_invalid_assessment(
        monkeypatch, decision, reason,
):
    create_ticket = Mock()
    monkeypatch.setattr(collection, "create_ticket_service", create_ticket)
    state = {
        "user_message": "确认",
        "order_id": "20260721001",
        "issue_description": "左耳无声",
        "requested_action": "换货",
        "missing_fields": [],
        "answer": None,
        "status": "ready_for_confirmation",
        "decision": decision,
        "reason": reason,
    }

    with pytest.raises(ValueError, match="预评估结果缺失或无效"):
        collection.create_ticket_node(state)

    create_ticket.assert_not_called()


@pytest.mark.parametrize("old_id,new_id,clears", [
    ("A", None, False),
    ("A", "  ", False),
    (None, "B", False),
    ("  ", "B", False),
    ("A", " A ", False),
    ("A", "B", True),
])
def test_order_change_clears_only_old_order_details(old_id, new_id, clears):
    state = {
        "user_message": "测试", "order_id": old_id,
        "issue_description": "左耳无声", "requested_action": "换货",
        "missing_fields": [], "answer": None,
    }
    original = state.copy()
    result = merge_ticket_extraction(state, TicketExtraction(order_id=new_id))

    assert result is not state
    assert state == original
    assert result["order_id"] == (new_id.strip() if new_id and new_id.strip() else old_id)
    assert result["issue_description"] == (None if clears else "左耳无声")
    assert result["requested_action"] == (None if clears else "换货")


def test_order_change_keeps_newly_supplied_details():
    state = {
        "user_message": "换订单", "order_id": "A",
        "issue_description": "左耳无声", "requested_action": "换货",
        "missing_fields": [], "answer": None,
    }
    result = merge_ticket_extraction(state, TicketExtraction(
        order_id=" B ", issue_description=" 空格键失灵 ", requested_action=" 维修 ",
    ))

    assert result["order_id"] == "B"
    assert result["issue_description"] == "空格键失灵"
    assert result["requested_action"] == "维修"
    assert state["order_id"] == "A"


def test_migrated_graph_restores_fields_and_previous_reply(monkeypatch):
    extraction = Mock(side_effect=[
        TicketExtraction(order_id="20260721001"),
        TicketExtraction(issue_description="左耳无声"),
        TicketExtraction(requested_action="换货"),
    ])
    monkeypatch.setattr(collection, "extract_ticket_fields", extraction)
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    first = collection.run_ticket_collection_turn("申请订单售后", graph, "test-session")
    second = collection.run_ticket_collection_turn("左耳无声", graph, "test-session")
    third = collection.run_ticket_collection_turn("第二个", graph, "test-session")

    assert first["missing_fields"] == ["issue_description", "requested_action"]
    assert second["missing_fields"] == ["requested_action"]
    assert second["status"] == "collecting"
    assert third["order_id"] == "20260721001"
    assert third["issue_description"] == "左耳无声"
    assert third["requested_action"] == "换货"
    assert third["missing_fields"] == []
    assert type(third["status"]) is str
    assert third["status"] == "ready_for_confirmation"
    assert "尚未创建工单" in third["answer"]
    assert extraction.call_args_list[0].kwargs["previous_question"] is None
    assert extraction.call_args_list[1].kwargs["previous_question"] == first["answer"]
    assert extraction.call_args_list[2].kwargs["previous_question"] == second["answer"]


def test_migrated_graph_accepts_complete_first_turn(monkeypatch):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001", issue_description="左耳无声", requested_action="换货",
    )))
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    result = collection.run_ticket_collection_turn("一次提供完整信息", graph, "complete-session")
    assert result["status"] == "ready_for_confirmation"
    assert result["missing_fields"] == []
    assert "20260721001" in result["answer"]
    assert "请回复“确认”" in result["answer"]
    assert "请回复“取消售后”" in result["answer"]


def test_ticket_collection_can_be_canceled_without_llm_call(monkeypatch):
    extraction = Mock()
    monkeypatch.setattr(collection, "extract_ticket_fields", extraction)
    graph = collection.build_ticket_collection_graph(InMemorySaver())

    result = collection.run_ticket_collection_turn("取消售后", graph, "cancel-session")

    assert result["status"] == "canceled"
    assert result["order_id"] is None
    assert result["issue_description"] is None
    assert result["requested_action"] is None
    assert result["missing_fields"] == []
    assert "已取消本次售后信息收集" in result["answer"]
    extraction.assert_not_called()


def test_ticket_collection_cancels_existing_collection_and_clears_fields(monkeypatch):
    extraction = Mock(return_value=TicketExtraction(order_id="20260721001"))
    monkeypatch.setattr(collection, "extract_ticket_fields", extraction)
    graph = collection.build_ticket_collection_graph(InMemorySaver())

    first = collection.run_ticket_collection_turn("帮订单20260721001创建售后工单", graph, "cancel-active-session")
    second = collection.run_ticket_collection_turn("不申请了", graph, "cancel-active-session")

    assert first["status"] == "collecting"
    assert second["status"] == "canceled"
    assert second["order_id"] is None
    assert second["issue_description"] is None
    assert second["requested_action"] is None
    assert second["missing_fields"] == []
    assert extraction.call_count == 1


@pytest.mark.parametrize("previous_status", ["created", "canceled"])
def test_new_application_clears_old_submission_in_same_thread(monkeypatch, previous_status):
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "new-application-test"}}
    graph.update_state(config, {
        "user_message": "旧申请",
        "order_id": "OLD-ORDER",
        "issue_description": "旧问题",
        "requested_action": "退货",
        "missing_fields": [],
        "answer": "旧回答",
        "status": previous_status,
        "decision": "allow",
        "reason": "旧原因",
        "confirmed_at": "2026-10-01T00:00:00+00:00",
        "confirmation_message": "旧确认",
        "ticket_id": "TICKET-OLD",
        "submission_id": "old-submission",
    }, as_node="prepare_ticket_summary")
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001",
    )))
    new_application = collection.run_ticket_collection_turn("申请另一笔售后", graph, "new-application-test")
    assert new_application["status"] == "collecting"
    assert new_application["order_id"] == "20260721001"
    for field in ("issue_description", "requested_action", "decision", "reason",
                  "confirmed_at", "confirmation_message", "ticket_id", "submission_id"):
        assert new_application[field] is None

    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        issue_description="新问题", requested_action="换货",
    )))
    collection.run_ticket_collection_turn("新问题，想换货", graph, "new-application-test")
    create = Mock(return_value=Ticket(
        ticket_id="TICKET-NEW", order_id="20260721001", user_id="USER-DEMO",
        issue_type="product_fault", description="新申请",
    ))
    monkeypatch.setattr(collection, "create_ticket_service", create)
    result = collection.run_ticket_collection_turn("确认", graph, "new-application-test")
    assert result["submission_id"] != "old-submission"
    assert result["submission_id"]
    assert create.call_args.kwargs["submission_id"] == result["submission_id"]


@pytest.mark.parametrize("values,next_nodes,error", [
    ({}, (), "没有可恢复"),
    ({"status": "collecting"}, (), "没有待恢复"),
])
def test_resume_rejects_missing_or_finished_execution(values, next_nodes, error):
    from types import SimpleNamespace

    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(values=values, next=next_nodes)
    expected_error = (
        collection.TicketSessionNotFoundError if not values
        else collection.TicketResumeNotRequiredError
    )
    with pytest.raises(expected_error, match=error):
        collection.resume_ticket_collection(graph, "resume-test")
    graph.invoke.assert_not_called()


def test_resume_passes_none_and_original_thread_config():
    from types import SimpleNamespace

    graph = Mock()
    graph.get_state.return_value = SimpleNamespace(
        values={"ticket_id": "TICKET-SAVED"}, next=("enqueue_manual_review",),
    )
    result = collection.resume_ticket_collection(graph, "resume-test")
    config = {"configurable": {"thread_id": "resume-test"}}
    graph.get_state.assert_called_once_with(config=config)
    graph.invoke.assert_called_once_with(None, config=config)
    assert result is graph.invoke.return_value


def test_resume_manual_review_does_not_repeat_ticket_creation(monkeypatch):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001", issue_description="左耳无声", requested_action="换货",
    )))
    create = Mock(return_value=Ticket(
        ticket_id="TICKET-RESUME", order_id="20260721001", user_id="USER-DEMO",
        issue_type="product_fault", description="测试恢复",
    ))
    enqueue = Mock(side_effect=[RuntimeError("模拟登记失败"), None])
    monkeypatch.setattr(collection, "create_ticket_service", create)
    monkeypatch.setattr(collection, "enqueue_manual_review", enqueue)
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    thread_id = "manual-review-resume-test"
    config = {"configurable": {"thread_id": thread_id}}
    collection.run_ticket_collection_turn("申请换货", graph, thread_id)
    with pytest.raises(RuntimeError, match="模拟登记失败"):
        collection.run_ticket_collection_turn("确认", graph, thread_id)
    before = graph.get_state(config)
    assert before.next == ("enqueue_manual_review",)
    assert before.values["ticket_id"] == "TICKET-RESUME"

    result = collection.resume_ticket_collection(graph, thread_id)

    create.assert_called_once()
    assert enqueue.call_count == 2
    assert result["submission_id"] == before.values["submission_id"]
    assert result["confirmation_message"] == "确认"
    assert result["ticket_id"] == "TICKET-RESUME"
    assert graph.get_state(config).next == ()


def test_cancel_clears_existing_submission_id():
    result = collection.cancel_ticket_collection_node({"submission_id": "old-submission"})
    assert result["submission_id"] is None


def test_ticket_collection_creates_ticket_after_confirmation(monkeypatch):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001", issue_description="左耳无声", requested_action="换货",
    )))
    create_ticket = Mock(return_value=Ticket(
        ticket_id="TICKET-TEST-001",
        order_id="20260721001",
        user_id="USER-DEMO",
        issue_type="product_fault",
        description="左耳无声；用户希望处理方式：换货",
        status=TicketStatus.OPEN,
    ))
    monkeypatch.setattr(collection, "create_ticket_service", create_ticket)
    graph = collection.build_ticket_collection_graph(InMemorySaver())

    first = collection.run_ticket_collection_turn("订单20260721001的耳机左耳无声，我想换货", graph, "confirm-session")
    second = collection.run_ticket_collection_turn("确认", graph, "confirm-session")

    assert first["status"] == "ready_for_confirmation"
    assert first["confirmed_at"] is None
    assert first["confirmation_message"] is None
    assert first["submission_id"] is None
    assert len(second["submission_id"]) == 32
    assert second["status"] == "created"
    assert second["confirmation_message"] == "确认"
    assert datetime.fromisoformat(second["confirmed_at"]).utcoffset() == timezone.utc.utcoffset(None)
    saved = graph.get_state({"configurable": {"thread_id": "confirm-session"}}).values
    assert saved["confirmed_at"] == second["confirmed_at"]
    assert saved["confirmation_message"] == "确认"
    assert saved["submission_id"] == second["submission_id"]
    assert "TICKET-TEST-001" in second["answer"]
    assert "已登记人工核验" in second["answer"]
    collection.enqueue_manual_review.assert_called_once_with("TICKET-TEST-001")
    create_ticket.assert_called_once()
    ticket_create = create_ticket.call_args.kwargs["ticket_create"]
    assert ticket_create.order_id == "20260721001"
    assert ticket_create.user_id == "USER-DEMO"
    assert ticket_create.issue_type == "product_fault"
    assert "问题描述：左耳无声" in ticket_create.description
    assert "用户诉求：换货" in ticket_create.description
    assert "预评估结果：" in ticket_create.description
    assert "需要进一步核验，暂不能承诺退换货" in ticket_create.description
    assert "预评估原因：" in ticket_create.description
    assert "疑似质量问题，需要检测" in ticket_create.description
    confirmation = create_ticket.call_args.kwargs["confirmation_create"]
    assert confirmation.application_snapshot["requested_action"] == "换货"
    assert confirmation.application_snapshot["decision"] == "need_manual_review"
    assert confirmation.confirmation_message == "确认"


@pytest.mark.parametrize("submission_id", [None, "", "   "])
def test_prepare_submission_generates_id_when_empty(monkeypatch, submission_id):
    from uuid import UUID

    fixed_uuid = UUID("00000000-0000-4000-8000-000000000001")
    monkeypatch.setattr(collection, "uuid4", lambda: fixed_uuid)
    assert collection.prepare_ticket_submission_node(
        {"submission_id": submission_id}
    ) == {"submission_id": fixed_uuid.hex}


def test_prepare_submission_preserves_existing_id(monkeypatch):
    generate_uuid = Mock()
    monkeypatch.setattr(collection, "uuid4", generate_uuid)
    assert collection.prepare_ticket_submission_node(
        {"submission_id": "existing-submission"}
    ) == {}
    generate_uuid.assert_not_called()


def test_prepare_submission_supports_old_state_without_id():
    from uuid import UUID

    result = collection.prepare_ticket_submission_node({})
    assert UUID(hex=result["submission_id"]).hex == result["submission_id"]


def test_submission_is_checkpointed_before_creation_failure(monkeypatch):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001", issue_description="左耳无声", requested_action="换货",
    )))
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "submission-failure-test"}}
    collection.run_ticket_collection_turn("申请换货", graph, "submission-failure-test")

    def fail_creation(**kwargs):
        raise RuntimeError("模拟创建失败")

    monkeypatch.setattr(collection, "create_ticket_service", fail_creation)
    with pytest.raises(RuntimeError, match="模拟创建失败"):
        collection.run_ticket_collection_turn("确认", graph, "submission-failure-test")

    snapshot = graph.get_state(config)
    assert snapshot.values["submission_id"]
    assert "create_ticket" in snapshot.next


@pytest.mark.parametrize("decision,queued", [("allow", False), ("reject", False), ("need_manual_review", True)])
def test_confirmed_ticket_follows_manual_review_route(monkeypatch, decision, queued):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001", issue_description="左耳无声", requested_action="换货",
    )))
    monkeypatch.setattr(collection, "assess_return_request", Mock(return_value=TicketEligibilityResult(
        decision=TicketEligibilityDecision(decision), reason="测试评估说明",
    )))
    monkeypatch.setattr(collection, "create_ticket_service", Mock(return_value=Ticket(
        ticket_id="TICKET-ROUTE-001", order_id="20260721001", user_id="USER-DEMO",
        issue_type="product_fault", description="测试申请",
    )))
    enqueue = Mock()
    monkeypatch.setattr(collection, "enqueue_manual_review", enqueue)
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    collection.run_ticket_collection_turn("申请换货", graph, "route-test")
    enqueue.assert_not_called()
    result = collection.run_ticket_collection_turn("确认", graph, "route-test")
    if decision == "reject":
        assert result["ticket_id"] is None
        assert result["status"] == "collecting"
        collection.create_ticket_service.assert_not_called()
        enqueue.assert_not_called()
        return
    assert result["ticket_id"] == "TICKET-ROUTE-001"
    if queued:
        enqueue.assert_called_once_with("TICKET-ROUTE-001")
        assert "已登记人工核验" in result["answer"]
    else:
        enqueue.assert_not_called()
        assert "已登记人工核验" not in result["answer"]


def test_manual_review_node_propagates_registration_failure(monkeypatch):
    error = RuntimeError("队列登记失败")
    enqueue = Mock(side_effect=error)
    monkeypatch.setattr(collection, "enqueue_manual_review", enqueue)
    state = {"ticket_id": "TICKET-FAILED-001", "answer": "已创建工单"}
    with pytest.raises(RuntimeError) as captured:
        collection.enqueue_manual_review_node(state)
    assert captured.value is error
    assert state["answer"] == "已创建工单"


def test_ticket_collection_can_update_requested_action_before_confirmation(monkeypatch):
    extraction = Mock(side_effect=[
        TicketExtraction(order_id="20260721001", issue_description="左耳无声", requested_action="换货"),
        TicketExtraction(requested_action="退货"),
    ])
    create_ticket = Mock()
    monkeypatch.setattr(collection, "extract_ticket_fields", extraction)
    monkeypatch.setattr(collection, "create_ticket_service", create_ticket)
    graph = collection.build_ticket_collection_graph(InMemorySaver())

    first = collection.run_ticket_collection_turn("订单20260721001左耳无声，我想换货", graph, "update-before-confirm")
    second = collection.run_ticket_collection_turn("我要换成退货", graph, "update-before-confirm")

    assert first["status"] == "ready_for_confirmation"
    assert second["status"] == "ready_for_confirmation"
    assert second["requested_action"] == "退货"
    assert "希望处理方式：退货" in second["answer"]
    assert "希望处理方式：换货" not in second["answer"]
    assert "请回复“确认”" in second["answer"]
    create_ticket.assert_not_called()


def test_changing_requested_action_reassesses_before_confirmation(monkeypatch):
    extraction = Mock(side_effect=[
        TicketExtraction(order_id="20260721001", issue_description="左耳无声", requested_action="换货"),
        TicketExtraction(requested_action="退货"),
    ])
    assess = Mock(side_effect=[
        TicketEligibilityResult(
            decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
            reason="需要质量检测",
        ),
        TicketEligibilityResult(
            decision=TicketEligibilityDecision.REJECT,
            reason="当前退货条件不满足",
        ),
    ])
    create_ticket = Mock()
    monkeypatch.setattr(collection, "extract_ticket_fields", extraction)
    monkeypatch.setattr(collection, "assess_return_request", assess)
    monkeypatch.setattr(collection, "create_ticket_service", create_ticket)
    graph = collection.build_ticket_collection_graph(InMemorySaver())

    first = collection.run_ticket_collection_turn("订单20260721001左耳无声，我想换货", graph, "reassess-session")
    second = collection.run_ticket_collection_turn("改成退货", graph, "reassess-session")

    assert first["decision"] == "need_manual_review"
    assert "需要质量检测" in first["answer"]
    assert second["requested_action"] == "退货"
    assert second["decision"] == "reject"
    assert second["reason"] == "当前退货条件不满足"
    assert "当前退货条件不满足" in second["answer"]
    assert "需要质量检测" not in second["answer"]
    assert [call.kwargs["requested_action"] for call in assess.call_args_list] == ["换货", "退货"]
    saved = graph.get_state({"configurable": {"thread_id": "reassess-session"}}).values
    assert saved["decision"] == "reject"
    assert saved["reason"] == "当前退货条件不满足"
    create_ticket.assert_not_called()


def test_ticket_collection_reports_eligibility_rejection_on_confirmation(monkeypatch):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721001", issue_description="左耳无声", requested_action="换货",
    )))
    reason = "订单未支付，暂不能创建售后工单。"
    create_ticket = Mock(side_effect=TicketEligibilityRejectedError(reason))
    monkeypatch.setattr(collection, "create_ticket_service", create_ticket)
    graph = collection.build_ticket_collection_graph(InMemorySaver())

    first = collection.run_ticket_collection_turn("订单20260721001左耳无声，我想换货", graph, "rejected-session")
    assert first["status"] == "ready_for_confirmation"

    with pytest.raises(LLMServiceError, match=reason):
        collection.run_ticket_collection_turn("确认", graph, "rejected-session")

    create_ticket.assert_called_once()


def test_preassessment_rejection_never_enters_confirmation_or_creates_ticket(monkeypatch):
    monkeypatch.setattr(collection, "extract_ticket_fields", Mock(return_value=TicketExtraction(
        order_id="20260721002", issue_description="键盘坏了", requested_action="换货",
    )))
    monkeypatch.setattr(collection, "assess_return_request", Mock(return_value=TicketEligibilityResult(
        decision=TicketEligibilityDecision.REJECT, reason="订单未支付，暂不能创建售后工单。",
    )))
    create = Mock()
    monkeypatch.setattr(collection, "create_ticket_service", create)
    graph = collection.build_ticket_collection_graph(InMemorySaver())
    first = collection.run_ticket_collection_turn("键盘坏了，我想换货", graph, "unpaid-test")
    assert first["status"] == "collecting"
    assert "订单未支付" in first["answer"]
    assert "请回复“确认”" not in first["answer"]
    second = collection.run_ticket_collection_turn("确认", graph, "unpaid-test")
    assert second["status"] == "collecting"
    create.assert_not_called()
