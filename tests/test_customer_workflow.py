from unittest.mock import Mock
from uuid import UUID

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from serviceflow.agent import ticket_collection

from serviceflow.agent import workflow
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.llm.models import CustomerIntent, IntentResult
from serviceflow.llm.service import LLMServiceError
from serviceflow.models import Ticket, TicketStatus
from serviceflow.ticket.rules import TicketEligibilityDecision, TicketEligibilityResult


@pytest.fixture(autouse=True)
def stub_return_assessment(monkeypatch):
    monkeypatch.setattr(ticket_collection, "enqueue_manual_review", Mock())
    monkeypatch.setattr(ticket_collection, "assess_return_request", Mock(return_value=TicketEligibilityResult(
        decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
        reason="疑似质量问题，需要检测",
    )))


def make_state(intent_result=None) -> workflow.CustomerWorkflowState:
    return {
        "user_message": "查询订单20260721001",
        "intent_result": intent_result,
        "answer": None,
        "thread_id": None,
        "steps": [],
    }


@pytest.mark.parametrize("thread_id", [None, "existing-thread"])
def test_workflow_entry_prepares_input_and_reads_final_state(thread_id, caplog):
    graph = Mock()
    graph.invoke.return_value = {
        "answer": "最终回答", "thread_id": "result-thread",
        "steps": ["classify_intent", "handle_tools"],
    }

    with caplog.at_level("INFO", logger=workflow.__name__):
        result = workflow.run_customer_workflow("客户问题", graph, thread_id)

    assert result == ("最终回答", "result-thread")
    assert "客服工作流完成 thread_id=result-thread steps=['classify_intent', 'handle_tools']" in caplog.text
    assert "客户问题" not in caplog.text
    assert "最终回答" not in caplog.text
    graph.invoke.assert_called_once()
    state = graph.invoke.call_args.args[0]
    assert state["user_message"] == "客户问题"
    assert state["intent_result"] is None
    assert state["answer"] is None
    assert state["steps"] == []
    if thread_id is None:
        assert UUID(state["thread_id"]).version == 4
    else:
        assert state["thread_id"] == thread_id


@pytest.mark.parametrize("answer", [None, "", " \n\t"])
def test_workflow_entry_rejects_empty_answer(answer):
    graph = Mock()
    graph.invoke.return_value = {"answer": answer, "thread_id": "test-thread"}

    with pytest.raises(AgentServiceError, match="客服工作流没有返回回答"):
        workflow.run_customer_workflow("客户问题", graph)


def test_workflow_entry_rejects_missing_thread_id():
    graph = Mock()
    graph.invoke.return_value = {"answer": "有效回答", "thread_id": None}

    with pytest.raises(AgentServiceError, match="客服工作流没有返回会话编号"):
        workflow.run_customer_workflow("客户问题", graph)


def test_workflow_entry_propagates_graph_error():
    original = AgentServiceError("节点执行失败")
    graph = Mock()
    graph.invoke.side_effect = original

    with pytest.raises(AgentServiceError) as captured:
        workflow.run_customer_workflow("客户问题", graph)

    assert captured.value is original


def test_classification_returns_only_intent_update(monkeypatch):
    expected = IntentResult(
        intent=CustomerIntent.ORDER_QUERY,
        order_id="20260721001",
        confidence=0.95,
        reason="用户查询订单",
    )
    classify = Mock(return_value=expected)
    monkeypatch.setattr(workflow, "classify_intent", classify)
    state = make_state()

    update = workflow.classify_customer_intent_node(state)

    classify.assert_called_once_with("查询订单20260721001")
    assert update == {"intent_result": expected, "steps": ["classify_intent"]}
    assert state == make_state()


def test_classification_converts_service_error(monkeypatch):
    original = LLMServiceError("模拟模型调用失败")
    monkeypatch.setattr(workflow, "classify_intent", Mock(side_effect=original))

    with pytest.raises(AgentServiceError, match="客户意图识别失败") as captured:
        workflow.classify_customer_intent_node(make_state())

    assert captured.value.__cause__ is original


@pytest.mark.parametrize("intent, branch", [
    (CustomerIntent.ORDER_QUERY, "handle_tools"),
    (CustomerIntent.LOGISTICS_QUERY, "handle_tools"),
    (CustomerIntent.PRODUCT_FAULT, "handle_rag"),
    (CustomerIntent.RETURN_REFUND, "handle_rag"),
    (CustomerIntent.CREATE_TICKET, "handle_ticket"),
    (CustomerIntent.PRODUCT_QUERY, "handle_fallback"),
    (CustomerIntent.OTHER, "handle_fallback"),
])
def test_compiled_workflow_runs_only_selected_branch(monkeypatch, intent, branch):
    intent_result = IntentResult(
        intent=intent, order_id="20260721001", confidence=0.95, reason="测试路由",
    )
    classify = Mock(return_value=intent_result)
    agent = Mock(return_value=("工具回答", "returned-thread"))
    rag = Mock(return_value="政策回答")
    monkeypatch.setattr(workflow, "classify_intent", classify)
    monkeypatch.setattr(workflow, "run_customer_agent_graph", agent)
    monkeypatch.setattr(workflow, "answer_policy_question", rag)
    inner_graph = Mock()
    embedding_model = Mock()
    ticket_graph = ticket_collection.build_ticket_collection_graph(InMemorySaver())
    ticket = Mock(return_value={"answer": "工单20260721001的追问"})
    monkeypatch.setattr(workflow, "run_ticket_collection_turn", ticket)
    graph = workflow.build_customer_workflow(inner_graph, embedding_model, ticket_graph)
    state = make_state()
    state["thread_id"] = "existing-thread"

    # 使用真实图和真实节点，仅替换外部模型、工具与检索服务。
    updates = list(graph.stream(state, stream_mode="updates"))

    assert [name for update in updates for name in update] == ["classify_intent", branch]
    assert updates[0]["classify_intent"] == {"intent_result": intent_result, "steps": ["classify_intent"]}
    classify.assert_called_once_with(state["user_message"])
    final_update = updates[-1][branch]
    assert final_update["steps"] == [branch]
    if branch == "handle_tools":
        agent.assert_called_once_with(
            user_text=state["user_message"], graph=inner_graph, thread_id="existing-thread",
        )
        assert final_update == {"answer": "工具回答", "thread_id": "returned-thread", "steps": [branch]}
    else:
        agent.assert_not_called()
    if branch == "handle_rag":
        rag.assert_called_once_with(model=embedding_model, user_question=state["user_message"])
        assert final_update == {"answer": "政策回答", "steps": [branch]}
    else:
        rag.assert_not_called()
    if branch == "handle_ticket":
        assert "20260721001" in final_update["answer"]
        ticket.assert_called_once_with(user_message=state["user_message"], graph=ticket_graph, thread_id="existing-thread")
    else:
        ticket.assert_not_called()
    if branch == "handle_fallback":
        assert "请补充您的具体需求" in final_update["answer"]

    # 单独的节点更新不等于合并后的状态，必须检查 Reducer 的实际结果。
    result = graph.invoke(state)
    assert result["steps"] == ["classify_intent", branch]
    assert state["steps"] == []


def test_ticket_without_order_id_records_complete_path(monkeypatch):
    intent_result = IntentResult(
        intent=CustomerIntent.CREATE_TICKET, confidence=0.9, reason="申请工单但未提供订单号",
    )
    monkeypatch.setattr(workflow, "classify_intent", Mock(return_value=intent_result))
    monkeypatch.setattr(ticket_collection, "extract_ticket_fields", Mock(return_value=ticket_collection.TicketExtraction()))
    ticket_graph = ticket_collection.build_ticket_collection_graph(InMemorySaver())
    graph = workflow.build_customer_workflow(Mock(), Mock(), ticket_graph)
    state = make_state()
    state["thread_id"] = "missing-order"
    result = graph.invoke(state)

    assert result["steps"] == ["classify_intent", "handle_ticket"]
    assert "请提供需要申请售后的订单号" in result["answer"]


def test_compiled_workflow_stops_when_classification_fails(monkeypatch):
    original = LLMServiceError("模拟分类失败")
    monkeypatch.setattr(workflow, "classify_intent", Mock(side_effect=original))
    agent = Mock()
    rag = Mock()
    monkeypatch.setattr(workflow, "run_customer_agent_graph", agent)
    monkeypatch.setattr(workflow, "answer_policy_question", rag)
    ticket_graph = ticket_collection.build_ticket_collection_graph(InMemorySaver())
    graph = workflow.build_customer_workflow(Mock(), Mock(), ticket_graph)

    with pytest.raises(AgentServiceError, match="客户意图识别失败") as captured:
        state = make_state()
        state["thread_id"] = "classification-failure"
        graph.invoke(state)

    assert captured.value.__cause__ is original
    agent.assert_not_called()
    rag.assert_not_called()


@pytest.mark.parametrize("intent, expected_route", [
    (CustomerIntent.ORDER_QUERY, "tools"),
    (CustomerIntent.LOGISTICS_QUERY, "tools"),
    (CustomerIntent.PRODUCT_FAULT, "rag"),
    (CustomerIntent.RETURN_REFUND, "rag"),
    (CustomerIntent.CREATE_TICKET, "ticket"),
    (CustomerIntent.PRODUCT_QUERY, "fallback"),
    (CustomerIntent.OTHER, "fallback"),
])
def test_workflow_routes_all_supported_intents(intent, expected_route):
    result = IntentResult(intent=intent, confidence=0.9, reason="测试分类")
    assert workflow.route_customer_workflow(make_state(result)) == expected_route


def test_active_ticket_skips_classification_on_followup_turns(monkeypatch, caplog):
    classify = Mock(return_value=IntentResult(
        intent=CustomerIntent.CREATE_TICKET, confidence=0.9, reason="申请工单",
    ))
    monkeypatch.setattr(workflow, "classify_intent", classify)
    monkeypatch.setattr(ticket_collection, "extract_ticket_fields", Mock(side_effect=[
        ticket_collection.TicketExtraction(order_id="20260721001"),
        ticket_collection.TicketExtraction(issue_description="左耳无声"),
        ticket_collection.TicketExtraction(requested_action="换货"),
    ]))
    ticket_graph = ticket_collection.build_ticket_collection_graph(InMemorySaver())
    graph = workflow.build_customer_workflow(Mock(), Mock(), ticket_graph)
    thread_id = None
    ids = []
    with caplog.at_level("INFO", logger=workflow.__name__):
        for message in ["帮我创建售后工单", "左耳无声", "换货"]:
            answer, thread_id = workflow.run_customer_workflow(message, graph, thread_id)
            ids.append(thread_id)
    assert len(set(ids)) == 1
    classify.assert_called_once_with("帮我创建售后工单")
    assert "尚未创建工单" in answer
    saved = ticket_graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert saved["order_id"] == "20260721001"
    assert saved["status"] == "ready_for_confirmation"
    records = [r.getMessage() for r in caplog.records if r.name == workflow.__name__]
    assert len(records) == 3
    assert "steps=['classify_intent', 'handle_ticket']" in records[0]
    assert all("steps=['handle_ticket']" in line for line in records[1:])


def test_active_ticket_can_be_canceled_without_reclassification_or_extraction(monkeypatch):
    classify = Mock(return_value=IntentResult(
        intent=CustomerIntent.CREATE_TICKET, confidence=0.9, reason="申请工单",
    ))
    extract = Mock(return_value=ticket_collection.TicketExtraction(order_id="20260721001"))
    monkeypatch.setattr(workflow, "classify_intent", classify)
    monkeypatch.setattr(ticket_collection, "extract_ticket_fields", extract)
    ticket_graph = ticket_collection.build_ticket_collection_graph(InMemorySaver())
    graph = workflow.build_customer_workflow(Mock(), Mock(), ticket_graph)

    first_answer, thread_id = workflow.run_customer_workflow("帮订单20260721001创建售后工单", graph)
    second_answer, second_thread_id = workflow.run_customer_workflow("取消售后", graph, thread_id)

    assert second_thread_id == thread_id
    assert "请描述商品遇到的具体问题" in first_answer
    assert "已取消本次售后信息收集" in second_answer
    classify.assert_called_once_with("帮订单20260721001创建售后工单")
    extract.assert_called_once_with(user_message="帮订单20260721001创建售后工单", previous_question=None)
    saved = ticket_graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert saved["status"] == ticket_collection.TicketCollectionStatus.CANCELED.value
    assert saved["order_id"] is None
    assert saved["missing_fields"] == []


def test_active_ticket_creates_ticket_after_confirmation_without_reclassification(monkeypatch):
    classify = Mock(return_value=IntentResult(
        intent=CustomerIntent.CREATE_TICKET, confidence=0.9, reason="申请工单",
    ))
    extract = Mock(side_effect=[
        ticket_collection.TicketExtraction(order_id="20260721001"),
        ticket_collection.TicketExtraction(issue_description="左耳无声"),
        ticket_collection.TicketExtraction(requested_action="换货"),
    ])
    create_ticket = Mock(return_value=Ticket(
        ticket_id="TICKET-WORKFLOW-001",
        order_id="20260721001",
        user_id="USER-DEMO",
        issue_type="product_fault",
        description="左耳无声；用户希望处理方式：换货",
        status=TicketStatus.OPEN,
    ))
    monkeypatch.setattr(workflow, "classify_intent", classify)
    monkeypatch.setattr(ticket_collection, "extract_ticket_fields", extract)
    monkeypatch.setattr(ticket_collection, "create_ticket_service", create_ticket)
    ticket_graph = ticket_collection.build_ticket_collection_graph(InMemorySaver())
    graph = workflow.build_customer_workflow(Mock(), Mock(), ticket_graph)

    thread_id = None
    for message in ["帮我创建售后工单", "左耳无声", "换货"]:
        answer, thread_id = workflow.run_customer_workflow(message, graph, thread_id)
    final_answer, final_thread_id = workflow.run_customer_workflow("确认", graph, thread_id)

    assert final_thread_id == thread_id
    assert "尚未创建工单" in answer
    assert "已为您创建售后工单" in final_answer
    assert "TICKET-WORKFLOW-001" in final_answer
    classify.assert_called_once_with("帮我创建售后工单")
    assert extract.call_count == 3
    create_ticket.assert_called_once()
    saved = ticket_graph.get_state({"configurable": {"thread_id": thread_id}}).values
    assert saved["status"] == ticket_collection.TicketCollectionStatus.CREATED.value
    assert saved["order_id"] == "20260721001"


def test_workflow_rejects_missing_intent_result():
    with pytest.raises(AgentServiceError, match="缺少意图识别结果"):
        workflow.route_customer_workflow(make_state())


@pytest.mark.parametrize("thread_id", [None, "existing-conversation"])
def test_tools_node_forwards_question_without_current_order_id(monkeypatch, thread_id):
    state = make_state(IntentResult(
        intent=CustomerIntent.LOGISTICS_QUERY,
        order_id=None,
        confidence=0.9,
        reason="用户追问物流",
    ))
    state["user_message"] = "帮我查一下物流"
    state["thread_id"] = thread_id
    graph = Mock()
    returned_id = thread_id or "generated-conversation"
    agent = Mock(return_value=("工具Agent的回答或追问", returned_id))
    monkeypatch.setattr(workflow, "run_customer_agent_graph", agent)

    update = workflow.handle_tools_node(state, agent_graph=graph)

    agent.assert_called_once_with(
        user_text="帮我查一下物流", graph=graph, thread_id=thread_id,
    )
    assert update == {"answer": "工具Agent的回答或追问", "thread_id": returned_id, "steps": ["handle_tools"]}
    assert state["answer"] is None


def test_tools_node_propagates_agent_failure(monkeypatch):
    original = AgentServiceError("工具执行失败")
    monkeypatch.setattr(workflow, "run_customer_agent_graph", Mock(side_effect=original))

    with pytest.raises(AgentServiceError) as captured:
        workflow.handle_tools_node(make_state(), agent_graph=Mock())

    assert captured.value is original


def test_rag_node_forwards_question_and_embedding_model(monkeypatch):
    state = make_state()
    state["user_message"] = "耳机拆封后左耳无声，可以退货吗？"
    model = Mock()
    rag = Mock(return_value="可以申请质量检测，具体处理以检测结果为准。")
    monkeypatch.setattr(workflow, "answer_policy_question", rag)

    update = workflow.handle_rag_node(state, embedding_model=model)

    rag.assert_called_once_with(model=model, user_question=state["user_message"])
    assert update == {"answer": "可以申请质量检测，具体处理以检测结果为准。", "steps": ["handle_rag"]}
    assert state["answer"] is None


def test_rag_node_converts_service_error(monkeypatch):
    original = workflow.RAGServiceError("引用校验失败")
    monkeypatch.setattr(workflow, "answer_policy_question", Mock(side_effect=original))

    with pytest.raises(AgentServiceError, match="售后政策问答失败") as captured:
        workflow.handle_rag_node(make_state(), embedding_model=Mock())

    assert captured.value.__cause__ is original
