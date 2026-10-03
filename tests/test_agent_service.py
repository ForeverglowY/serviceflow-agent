from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx import Request
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from serviceflow.agent import service as agent_service
from serviceflow.agent import runtime as agent_runtime
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.agent.service import run_customer_agent
from serviceflow.llm.models import GetOrderResult


def make_response(message: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
    )


def make_direct_message(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        tool_calls=None,
    )


def make_tool_message(
    *,
    name: str = "get_order",
    arguments: str = '{"order_id":"20260721001"}',
    call_id: str = "call-test-001",
) -> Mock:
    tool_call = ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function={
            "name": name,
            "arguments": arguments,
        },
    )
    message = Mock()
    message.content = None
    message.tool_calls = [tool_call]
    message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }
    return message


def make_multiple_tool_message() -> Mock:
    order_call = ChatCompletionMessageFunctionToolCall(
        id="call-order-001",
        type="function",
        function={
            "name": "get_order",
            "arguments": '{"order_id":"20260721001"}',
        },
    )
    logistics_call = ChatCompletionMessageFunctionToolCall(
        id="call-logistics-001",
        type="function",
        function={
            "name": "get_logistics",
            "arguments": '{"order_id":"20260721001"}',
        },
    )
    message = Mock()
    message.content = None
    message.tool_calls = [order_call, logistics_call]
    message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            order_call.model_dump(exclude_none=True),
            logistics_call.model_dump(exclude_none=True),
        ],
    }
    return message


def test_agent_returns_direct_answer_without_calling_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_mock = Mock(
        return_value=make_response(
            make_direct_message("您好，请问有什么可以帮助您？")
        )
    )
    execute_tool_mock = Mock()
    monkeypatch.setattr(agent_service, "call_deepseek", call_mock)
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        execute_tool_mock,
    )

    result = run_customer_agent("你好")

    assert result == "您好，请问有什么可以帮助您？"
    assert call_mock.call_count == 1
    execute_tool_mock.assert_not_called()


def test_agent_executes_get_order_and_returns_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_message = make_tool_message()
    final_message = make_direct_message(
        "订单20260721001当前状态为待处理。"
    )
    call_mock = Mock(
        side_effect=[
            make_response(first_message),
            make_response(final_message),
        ]
    )
    tool_result = GetOrderResult(
            order_id="20260721001",
            product_name="蓝牙耳机",
            unit_price=399.0,
            quantity=2,
            total_price=798.0,
            payment_status="已支付",
            order_status="待处理",
            tracking_number=None,
    )
    execute_tool_mock = Mock(
        return_value=tool_result.model_dump_json()
    )
    monkeypatch.setattr(agent_service, "call_deepseek", call_mock)
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        execute_tool_mock,
    )

    result = run_customer_agent("查询订单20260721001")

    assert result == "订单20260721001当前状态为待处理。"
    assert call_mock.call_count == 2
    execute_tool_mock.assert_called_once_with(
        "get_order",
        '{"order_id":"20260721001"}',
    )

    second_messages = call_mock.call_args_list[1].args[0]
    assert len(second_messages) == 4
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[3]["role"] == "tool"
    assert second_messages[3]["tool_call_id"] == "call-test-001"
    assert '"order_status":"待处理"' in second_messages[3]["content"]


def test_agent_returns_not_found_result_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_mock = Mock(
        side_effect=[
            make_response(make_tool_message()),
            make_response(make_direct_message("没有找到该订单。")),
        ]
    )
    monkeypatch.setattr(agent_service, "call_deepseek", call_mock)
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        Mock(return_value='{"error": "未查询到相关数据"}'),
    )

    result = run_customer_agent("查询订单20260721001")

    assert result == "没有找到该订单。"
    second_messages = call_mock.call_args_list[1].args[0]
    assert second_messages[3]["content"] == (
        '{"error": "未查询到相关数据"}'
    )


def test_agent_executes_multiple_tools_in_one_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_mock = Mock(
        side_effect=[
            make_response(make_multiple_tool_message()),
            make_response(
                make_direct_message("订单和物流信息已查询完成。")
            ),
        ]
    )
    execute_tool_mock = Mock(
        side_effect=[
            '{"order_id":"20260721001","order_status":"待处理"}',
            '{"order_id":"20260721001","logistics_status":"运输中"}',
        ]
    )
    monkeypatch.setattr(agent_service, "call_deepseek", call_mock)
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        execute_tool_mock,
    )

    result = run_customer_agent("查询订单和物流")

    assert result == "订单和物流信息已查询完成。"
    assert execute_tool_mock.call_args_list[0].args[0] == "get_order"
    assert execute_tool_mock.call_args_list[1].args[0] == "get_logistics"

    second_messages = call_mock.call_args_list[1].args[0]
    assert len(second_messages) == 5
    assert second_messages[2]["role"] == "assistant"
    assert second_messages[3]["tool_call_id"] == "call-order-001"
    assert second_messages[4]["tool_call_id"] == "call-logistics-001"


def test_agent_stops_after_maximum_tool_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_mock = Mock(
        side_effect=[
            make_response(
                make_tool_message(call_id=f"call-test-{index}")
            )
            for index in range(agent_service.MAX_TOOL_ROUNDS + 1)
        ]
    )
    execute_tool_mock = Mock(return_value='{"result":"ok"}')
    monkeypatch.setattr(agent_service, "call_deepseek", call_mock)
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        execute_tool_mock,
    )

    with pytest.raises(
        AgentServiceError,
        match="超过最大工具调用次数",
    ):
        run_customer_agent("重复调用工具")

    assert call_mock.call_count == agent_service.MAX_TOOL_ROUNDS + 1
    assert execute_tool_mock.call_count == agent_service.MAX_TOOL_ROUNDS


def test_call_deepseek_converts_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout_error = APITimeoutError(
        request=Request("POST", "https://api.deepseek.com/chat/completions")
    )
    monkeypatch.setattr(
        agent_runtime.CLIENT.chat.completions,
        "create",
        Mock(side_effect=timeout_error),
    )

    with pytest.raises(
        AgentServiceError,
        match="DeepSeek请求超时",
    ) as exc_info:
        agent_runtime.call_deepseek([])

    assert exc_info.value.__cause__ is timeout_error


def test_agent_converts_database_pool_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout_error = SQLAlchemyTimeoutError("模拟连接池超时")
    monkeypatch.setattr(
        agent_service,
        "call_deepseek",
        Mock(return_value=make_response(make_tool_message())),
    )
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        Mock(side_effect=timeout_error),
    )

    with pytest.raises(
        AgentServiceError,
        match="工具执行超时或数据库不可用",
    ) as exc_info:
        run_customer_agent("查询订单")

    assert exc_info.value.__cause__ is timeout_error


def test_agent_converts_database_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_error = SQLAlchemyError("模拟数据库错误")
    monkeypatch.setattr(
        agent_service,
        "call_deepseek",
        Mock(return_value=make_response(make_tool_message())),
    )
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        Mock(side_effect=database_error),
    )

    with pytest.raises(
        AgentServiceError,
        match="工具执行失败",
    ) as exc_info:
        run_customer_agent("查询订单")

    assert exc_info.value.__cause__ is database_error


def test_agent_rejects_unknown_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_service,
        "call_deepseek",
        Mock(return_value=make_response(make_tool_message(name="delete_order"))),
    )

    with pytest.raises(
        AgentServiceError,
        match="未注册的工具：delete_order",
    ):
        run_customer_agent("删除订单")


def test_agent_rejects_invalid_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_service,
        "call_deepseek",
        Mock(
            return_value=make_response(
                make_tool_message(arguments='{"unknown":"value"}')
            )
        ),
    )

    with pytest.raises(
        AgentServiceError,
        match="工具调用参数格式错误",
    ) as exc_info:
        run_customer_agent("查询订单")

    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize(
    ("responses", "error_message"),
    [
        ([make_response(make_direct_message(None))], "DeepSeek没有返回内容"),
        (
            [
                make_response(make_tool_message()),
                make_response(make_direct_message(None)),
            ],
            "DeepSeek没有返回内容",
        ),
    ],
)
def test_agent_rejects_empty_model_content(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[SimpleNamespace],
    error_message: str,
) -> None:
    monkeypatch.setattr(
        agent_service,
        "call_deepseek",
        Mock(side_effect=responses),
    )
    monkeypatch.setattr(
        agent_service,
        "execute_tool",
        Mock(return_value='{"error": "未查询到相关数据"}'),
    )

    with pytest.raises(AgentServiceError, match=error_message):
        run_customer_agent("测试消息")
