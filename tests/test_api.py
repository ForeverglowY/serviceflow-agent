from collections.abc import Iterator
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from serviceflow.agent.runtime import AgentServiceError
from serviceflow.api import app
from serviceflow.db.database import engine
from serviceflow.db.models import TicketTable
from serviceflow.llm.models import CustomerIntent, IntentResult
from serviceflow.llm.service import LLMServiceError
from serviceflow.ticket.service import TicketEligibilityRejectedError

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_workflow_graph(monkeypatch: pytest.MonkeyPatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from serviceflow.agent.graph import build_customer_agent_graph
    from serviceflow.agent.ticket_collection import build_ticket_collection_graph
    from serviceflow.agent.workflow import build_customer_workflow

    # 路由单测使用独立内存图，不依赖 PostgreSQL 生命周期。
    agent_graph = build_customer_agent_graph(InMemorySaver())
    ticket_graph = build_ticket_collection_graph(InMemorySaver())
    workflow_graph = build_customer_workflow(agent_graph, Mock(), ticket_graph)
    monkeypatch.setattr(app.state, "agent_graph", agent_graph, raising=False)
    monkeypatch.setattr(app.state, "ticket_graph", ticket_graph, raising=False)
    monkeypatch.setattr(app.state, "workflow_graph", workflow_graph, raising=False)
    return workflow_graph


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_customer_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_result = IntentResult(
        intent=CustomerIntent.LOGISTICS_QUERY,
        order_id="20260721002",
        confidence=0.95,
        reason="用户询问订单送达时间",
    )
    classify_mock = Mock(return_value=expected_result)
    monkeypatch.setattr("serviceflow.api.classify_intent", classify_mock)

    response = client.post(
        "/agent/intent",
        json={
            "message": "订单20260721002什么时候能送到？",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "intent": "logistics_query",
        "order_id": "20260721002",
        "confidence": 0.95,
        "reason": "用户询问订单送达时间",
    }
    classify_mock.assert_called_once_with(
        "订单20260721002什么时候能送到？"
    )


def test_classify_customer_intent_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "serviceflow.api.classify_intent",
        Mock(side_effect=LLMServiceError("模拟模型服务失败")),
    )

    response = client.post(
        "/agent/intent",
        json={"message": "查询订单"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "智能客服服务暂时不可用",
    }


def test_classify_customer_intent_rejects_empty_message() -> None:
    response = client.post(
        "/agent/intent",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_chat_with_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow_mock = Mock(
        return_value=("订单20260721001当前状态为待处理。", "generated-chat-id")
    )
    monkeypatch.setattr(
        "serviceflow.api.run_customer_workflow",
        workflow_mock,
    )

    response = client.post(
        "/agent/chat",
        json={"message": "查询订单20260721001"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "订单20260721001当前状态为待处理。",
        "thread_id": "generated-chat-id",
    }
    workflow_mock.assert_called_once_with(
        user_text="查询订单20260721001",
        workflow_graph=app.state.workflow_graph,
        thread_id=None,
    )


def test_chat_with_agent_passes_thread_id(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow_mock = Mock(return_value=("订单号是20260721001。", "chat-memory-001"))
    monkeypatch.setattr("serviceflow.api.run_customer_workflow", workflow_mock)

    response = client.post("/agent/chat", json={
        "message": "我刚才查询的订单号是什么？",
        "thread_id": "chat-memory-001",
    })

    assert response.status_code == 200
    assert response.json() == {"answer": "订单号是20260721001。", "thread_id": "chat-memory-001"}
    workflow_mock.assert_called_once_with(
        user_text="我刚才查询的订单号是什么？",
        workflow_graph=app.state.workflow_graph,
        thread_id="chat-memory-001",
    )


def test_chat_with_agent_resumes_server_generated_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow_mock = Mock(side_effect=[
        ("已记住订单20260721001。", "server-generated-thread"),
        ("您刚才提到20260721001。", "server-generated-thread"),
    ])
    monkeypatch.setattr("serviceflow.api.run_customer_workflow", workflow_mock)
    first = client.post("/agent/chat", json={"message": "我正在咨询订单20260721001"})
    assert first.status_code == 200
    thread_id = first.json()["thread_id"]
    assert thread_id == "server-generated-thread"

    second = client.post("/agent/chat", json={
        "message": "我刚才提到哪个订单？", "thread_id": thread_id,
    })
    assert second.status_code == 200
    assert second.json() == {
        "answer": "您刚才提到20260721001。", "thread_id": thread_id,
    }
    assert workflow_mock.call_args_list[0].kwargs["thread_id"] is None
    assert workflow_mock.call_args_list[1].kwargs["thread_id"] == "server-generated-thread"


@pytest.mark.parametrize("thread_id", ["", "x" * 101, 123])
def test_chat_with_agent_rejects_invalid_thread_id(
    monkeypatch: pytest.MonkeyPatch, thread_id: object,
) -> None:
    workflow_mock = Mock()
    monkeypatch.setattr("serviceflow.api.run_customer_workflow", workflow_mock)

    response = client.post("/agent/chat", json={
        "message": "你好",
        "thread_id": thread_id,
    })

    assert response.status_code == 422
    assert any(error["loc"] == ["body", "thread_id"] for error in response.json()["detail"])
    workflow_mock.assert_not_called()


def test_chat_with_agent_returns_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "serviceflow.api.run_customer_workflow",
        Mock(side_effect=AgentServiceError("模拟Agent失败")),
    )

    response = client.post(
        "/agent/chat",
        json={"message": "查询订单"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "智能客服服务暂时不可用",
    }


def test_chat_with_agent_rejects_empty_message() -> None:
    response = client.post(
        "/agent/chat",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_get_existing_order() -> None:
    response = client.get("/orders/20260721001")

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "20260721001"
    assert data["product_name"] == "蓝牙耳机"
    assert data["price"] == 399.0


def test_get_unknown_order_returns_404() -> None:
    response = client.get("/orders/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "订单不存在"}


def test_get_existing_product() -> None:
    response = client.get("/products/PRODUCT-003")

    assert response.status_code == 200

    data = response.json()
    assert data["product_id"] == "PRODUCT-003"
    assert data["stock"] == 0
    assert data["is_returnable"] is False


def test_get_unknown_product_returns_404() -> None:
    response = client.get("/products/PRODUCT-999")

    assert response.status_code == 404
    assert response.json() == {"detail": "商品不存在"}


def test_get_existing_logistics() -> None:
    response = client.get("/logistics/20260721002")

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "20260721002"
    assert data["status"] == "pending"
    assert data["latest_event"] is None


def test_get_unknown_logistics_returns_404() -> None:
    response = client.get("/logistics/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "物流信息不存在"}


def test_preview_valid_ticket() -> None:
    ticket_data = {
        "ticket_id": "TICKET-API-001",
        "order_id": "20260721001",
        "user_id": "USER-001",
        "issue_type": "product_fault",
        "description": "蓝牙耳机左耳没有声音",
        "priority": "high",
    }

    response = client.post("/tickets/preview", json=ticket_data)

    assert response.status_code == 200

    data = response.json()
    assert data["ticket_id"] == "TICKET-API-001"
    assert data["priority"] == "high"
    assert data["status"] == "open"
    assert data["assigned_to"] is None


def test_preview_invalid_ticket_returns_422() -> None:
    ticket_data = {
        "ticket_id": "TICKET-API-002",
        "order_id": "20260721001",
        "user_id": "USER-001",
        "issue_type": "product_fault",
        "description": "",
        "priority": "critical",
    }

    response = client.post("/tickets/preview", json=ticket_data)

    assert response.status_code == 422

    errors = response.json()["detail"]
    error_locations = [error["loc"] for error in errors]

    assert ["body", "description"] in error_locations
    assert ["body", "priority"] in error_locations


TEST_TICKET_ID = "TICKET-00000000000040008000000000000001"

VALID_TICKET_CREATE_DATA = {
    "order_id": "20260721001",
    "user_id": "USER-TEST-001",
    "issue_type": "product_fault",
    "description": "自动化测试：蓝牙耳机左耳没有声音",
    "priority": "high",
}


@pytest.fixture
def clean_test_ticket() -> Iterator[None]:
    def delete_test_ticket() -> None:
        with Session(engine) as session:
            with session.begin():
                ticket_record = session.get(
                    TicketTable,
                    TEST_TICKET_ID,
                )

                if ticket_record is not None:
                    session.delete(ticket_record)

    delete_test_ticket()
    yield
    delete_test_ticket()


def test_create_ticket_and_query_it(
    monkeypatch: pytest.MonkeyPatch,
    clean_test_ticket: None,
) -> None:
    fixed_uuid = UUID(
        "00000000-0000-4000-8000-000000000001"
    )
    monkeypatch.setattr(
        "serviceflow.ticket.service.uuid4",
        lambda: fixed_uuid,
    )

    create_response = client.post(
        "/tickets",
        json=VALID_TICKET_CREATE_DATA,
    )

    assert create_response.status_code == 201

    created_ticket = create_response.json()
    assert created_ticket["ticket_id"] == TEST_TICKET_ID
    assert created_ticket["order_id"] == "20260721001"
    assert created_ticket["priority"] == "high"
    assert created_ticket["status"] == "open"
    assert created_ticket["assigned_to"] is None

    query_response = client.get(
        f"/tickets/{TEST_TICKET_ID}"
    )

    assert query_response.status_code == 200
    assert query_response.json() == created_ticket


def test_get_unknown_ticket_returns_404() -> None:
    response = client.get(
        "/tickets/TICKET-NOT-FOUND"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "工单不存在",
    }


def test_create_ticket_for_unknown_order_returns_404(
    monkeypatch: pytest.MonkeyPatch,
    clean_test_ticket: None,
) -> None:
    fixed_uuid = UUID(
        "00000000-0000-4000-8000-000000000001"
    )
    monkeypatch.setattr(
        "serviceflow.ticket.service.uuid4",
        lambda: fixed_uuid,
    )

    invalid_data = {
        **VALID_TICKET_CREATE_DATA,
        "order_id": "ORDER-NOT-FOUND",
    }

    response = client.post(
        "/tickets",
        json=invalid_data,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "订单不存在",
    }

    with Session(engine) as session:
        ticket_record = session.get(
            TicketTable,
            TEST_TICKET_ID,
        )
        assert ticket_record is None


def test_create_ticket_with_invalid_body_returns_422() -> None:
    invalid_data = {
        "order_id": "20260721001",
        "user_id": "USER-TEST-001",
        "issue_type": "product_fault",
        "description": "",
        "priority": "critical",
    }

    response = client.post(
        "/tickets",
        json=invalid_data,
    )

    assert response.status_code == 422


def test_create_ticket_eligibility_rejection_returns_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reason = "订单未支付，暂不能创建售后工单。"
    create_mock = Mock(side_effect=TicketEligibilityRejectedError(reason))
    monkeypatch.setattr("serviceflow.api.create_ticket_service", create_mock)

    response = client.post("/tickets", json=VALID_TICKET_CREATE_DATA)

    assert response.status_code == 409
    assert response.json() == {"detail": reason}
    create_mock.assert_called_once()


def test_create_ticket_id_conflict_returns_409(
    monkeypatch: pytest.MonkeyPatch,
    clean_test_ticket: None,
) -> None:
    with Session(engine) as session:
        with session.begin():
            session.add(
                TicketTable(
                    ticket_id=TEST_TICKET_ID,
                    order_id="20260721001",
                    user_id="USER-ORIGINAL",
                    issue_type="existing_ticket",
                    description="已经存在的测试工单",
                    priority="medium",
                    status="open",
                    assigned_to=None,
                )
            )

    fixed_uuid = UUID(
        "00000000-0000-4000-8000-000000000001"
    )
    monkeypatch.setattr(
        "serviceflow.ticket.service.uuid4",
        lambda: fixed_uuid,
    )

    response = client.post(
        "/tickets",
        json=VALID_TICKET_CREATE_DATA,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "工单创建冲突",
    }

    with Session(engine) as session:
        existing_ticket = session.get(
            TicketTable,
            TEST_TICKET_ID,
        )

        assert existing_ticket is not None
        assert existing_ticket.user_id == "USER-ORIGINAL"
