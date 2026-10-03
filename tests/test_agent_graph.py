from types import SimpleNamespace
from unittest.mock import Mock, call
from uuid import UUID, uuid4

import pytest
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from pydantic import ValidationError
from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLAlchemyTimeoutError,
)

from serviceflow.agent import graph as agent_graph
from serviceflow.agent.graph import (
    AgentGraphState,
    CUSTOMER_AGENT_GRAPH,
    execute_tools_node,
    route_after_model,
    run_customer_agent_graph,
)
from serviceflow.agent.runtime import AGENT_SYSTEM_MESSAGE, AgentServiceError
from serviceflow.agent.tools import ToolNotFoundError
from serviceflow.llm.models import GetOrderArguments


def make_response(message: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
    )


def make_direct_message(content: str | None) -> Mock:
    message = Mock()
    message.content = content
    message.tool_calls = None
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
    }
    return message


def make_tool_call(
    *,
    name: str,
    arguments: str,
    call_id: str,
) -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function={
            "name": name,
            "arguments": arguments,
        },
    )


def make_tool_message(
    tool_calls: list[ChatCompletionMessageFunctionToolCall],
) -> Mock:
    message = Mock()
    message.content = None
    message.tool_calls = tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            tool_call.model_dump(exclude_none=True)
            for tool_call in tool_calls
        ],
    }
    return message


def make_initial_state() -> AgentGraphState:
    return {
        "messages": [
            AGENT_SYSTEM_MESSAGE,
            {
                "role": "user",
                "content": "帮我查询订单20260721001",
            },
        ],
        "pending_tool_calls": [],
        "answer": None,
        "tool_rounds": 0,
        "steps": [],
    }


def make_validation_error() -> ValidationError:
    try:
        GetOrderArguments.model_validate({})
    except ValidationError as exception:
        return exception

    raise AssertionError("测试数据没有触发ValidationError")


def test_run_customer_agent_graph_returns_direct_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_mock = Mock(
        return_value=make_response(
            make_direct_message("您好，请问有什么可以帮助您？")
        )
    )
    execute_tool_mock = Mock()
    monkeypatch.setattr(agent_graph, "call_deepseek", call_mock)
    monkeypatch.setattr(agent_graph, "execute_tool", execute_tool_mock)

    answer, thread_id = run_customer_agent_graph("你好", graph=CUSTOMER_AGENT_GRAPH)

    assert answer == "您好，请问有什么可以帮助您？"
    assert UUID(thread_id).version == 4
    call_mock.assert_called_once()
    execute_tool_mock.assert_not_called()


def test_graph_executes_one_tool_and_returns_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_call = make_tool_call(
        name="get_order",
        arguments='{"order_id":"20260721001"}',
        call_id="call-order-001",
    )
    call_mock = Mock(
        side_effect=[
            make_response(make_tool_message([tool_call])),
            make_response(
                make_direct_message("订单20260721001当前状态为待处理。")
            ),
        ]
    )
    execute_tool_mock = Mock(
        return_value='{"order_id":"20260721001","order_status":"待处理"}'
    )
    monkeypatch.setattr(agent_graph, "call_deepseek", call_mock)
    monkeypatch.setattr(agent_graph, "execute_tool", execute_tool_mock)

    result = CUSTOMER_AGENT_GRAPH.invoke(make_initial_state(), config={"configurable": {"thread_id": uuid4().hex}})

    assert result["answer"] == "订单20260721001当前状态为待处理。"
    assert result["tool_rounds"] == 1
    assert result["pending_tool_calls"] == []
    assert result["steps"] == [
        "call_model",
        "execute_tools",
        "call_model",
    ]
    assert [message["role"] for message in result["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    execute_tool_mock.assert_called_once_with(
        "get_order",
        '{"order_id":"20260721001"}',
    )

    second_messages = call_mock.call_args_list[1].args[0]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-order-001",
        "content": '{"order_id":"20260721001","order_status":"待处理"}',
    }


def test_graph_executes_multiple_tools_in_one_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_call = make_tool_call(
        name="get_order",
        arguments='{"order_id":"20260721001"}',
        call_id="call-order-001",
    )
    logistics_call = make_tool_call(
        name="get_logistics",
        arguments='{"order_id":"20260721001"}',
        call_id="call-logistics-001",
    )
    call_mock = Mock(
        side_effect=[
            make_response(
                make_tool_message([order_call, logistics_call])
            ),
            make_response(make_direct_message("订单和物流信息已查询。")),
        ]
    )
    execute_tool_mock = Mock(
        side_effect=[
            '{"order_status":"待处理"}',
            '{"logistics_status":"运输中"}',
        ]
    )
    monkeypatch.setattr(agent_graph, "call_deepseek", call_mock)
    monkeypatch.setattr(agent_graph, "execute_tool", execute_tool_mock)

    result = CUSTOMER_AGENT_GRAPH.invoke(make_initial_state(), config={"configurable": {"thread_id": uuid4().hex}})

    assert result["answer"] == "订单和物流信息已查询。"
    assert result["tool_rounds"] == 1
    assert result["steps"] == [
        "call_model",
        "execute_tools",
        "call_model",
    ]
    assert [message["role"] for message in result["messages"]].count(
        "tool"
    ) == 2
    assert execute_tool_mock.call_args_list == [
        call("get_order", '{"order_id":"20260721001"}'),
        call("get_logistics", '{"order_id":"20260721001"}'),
    ]


def test_graph_stops_when_tool_round_limit_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_call = make_tool_call(
        name="get_order",
        arguments='{"order_id":"20260721001"}',
        call_id="call-order-001",
    )
    second_call = make_tool_call(
        name="get_logistics",
        arguments='{"order_id":"20260721001"}',
        call_id="call-logistics-001",
    )
    call_mock = Mock(
        side_effect=[
            make_response(make_tool_message([first_call])),
            make_response(make_tool_message([second_call])),
        ]
    )
    execute_tool_mock = Mock(return_value='{"order_status":"待处理"}')
    monkeypatch.setattr(agent_graph, "MAX_TOOL_ROUNDS", 1)
    monkeypatch.setattr(agent_graph, "call_deepseek", call_mock)
    monkeypatch.setattr(agent_graph, "execute_tool", execute_tool_mock)

    with pytest.raises(
        AgentServiceError,
        match="超过最大工具调用次数",
    ):
        CUSTOMER_AGENT_GRAPH.invoke(make_initial_state(), config={"configurable": {"thread_id": uuid4().hex}})

    assert call_mock.call_count == 2
    assert execute_tool_mock.call_count == 1


def test_graph_rejects_empty_model_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_graph,
        "call_deepseek",
        Mock(return_value=make_response(make_direct_message(None))),
    )

    with pytest.raises(
        AgentServiceError,
        match="DeepSeek没有返回内容",
    ):
        CUSTOMER_AGENT_GRAPH.invoke(make_initial_state(), config={"configurable": {"thread_id": uuid4().hex}})


def test_graph_rejects_non_function_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_tool_call = SimpleNamespace(type="custom")
    message = Mock()
    message.content = None
    message.tool_calls = [custom_tool_call]
    message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [],
    }
    monkeypatch.setattr(
        agent_graph,
        "call_deepseek",
        Mock(return_value=make_response(message)),
    )

    with pytest.raises(
        AgentServiceError,
        match="不支持的工具调用类型：custom",
    ):
        CUSTOMER_AGENT_GRAPH.invoke(make_initial_state(), config={"configurable": {"thread_id": uuid4().hex}})


def test_execute_tools_node_rejects_empty_pending_calls() -> None:
    with pytest.raises(
        AgentServiceError,
        match="没有待执行的工具调用",
    ):
        execute_tools_node(make_initial_state())


@pytest.mark.parametrize(
    ("original_exception", "expected_message"),
    [
        (
            ToolNotFoundError("未注册的工具：unknown_tool"),
            "未注册的工具：unknown_tool",
        ),
        (
            make_validation_error(),
            "工具调用参数格式错误",
        ),
        (
            OperationalError(
                "SELECT 1",
                {},
                RuntimeError("数据库不可用"),
            ),
            "工具执行超时或数据库不可用",
        ),
        (
            SQLAlchemyTimeoutError("连接池超时"),
            "工具执行超时或数据库不可用",
        ),
        (
            SQLAlchemyError("数据库执行失败"),
            "工具执行失败",
        ),
    ],
)
def test_execute_tools_node_converts_tool_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    original_exception: Exception,
    expected_message: str,
) -> None:
    tool_call = make_tool_call(
        name="get_order",
        arguments='{"order_id":"20260721001"}',
        call_id="call-order-001",
    )
    state = make_initial_state()
    state["pending_tool_calls"] = [tool_call]
    monkeypatch.setattr(
        agent_graph,
        "execute_tool",
        Mock(side_effect=original_exception),
    )

    with pytest.raises(
        AgentServiceError,
        match=expected_message,
    ) as exception_info:
        execute_tools_node(state)

    assert exception_info.value.__cause__ is original_exception


def test_route_after_model_rejects_inconsistent_state() -> None:
    with pytest.raises(
        AgentServiceError,
        match="模型既未返回最终回答，也未请求调用工具",
    ):
        route_after_model(make_initial_state())


def test_conversation_preserves_history_without_duplicate_system(monkeypatch: pytest.MonkeyPatch) -> None:
    model_mock = Mock(side_effect=[
        make_response(make_direct_message("第一轮回答")),
        make_response(make_direct_message("第二轮回答")),
    ])
    monkeypatch.setattr(agent_graph, "call_deepseek", model_mock)
    thread_id = uuid4().hex

    run_customer_agent_graph("查询订单20260721001", graph=CUSTOMER_AGENT_GRAPH, thread_id=thread_id)
    answer, returned_thread_id = run_customer_agent_graph("那它到哪里了？", graph=CUSTOMER_AGENT_GRAPH, thread_id=thread_id)

    assert answer == "第二轮回答"
    assert returned_thread_id == thread_id
    second_messages = model_mock.call_args_list[1].args[0]
    assert [message["role"] for message in second_messages] == [
        "system", "user", "assistant", "user",
    ]
    assert [message["content"] for message in second_messages[1:]] == [
        "查询订单20260721001", "第一轮回答", "那它到哪里了？",
    ]


@pytest.mark.parametrize("explicit_ids", [True, False])
def test_separate_conversations_do_not_share_messages(monkeypatch: pytest.MonkeyPatch, explicit_ids: bool) -> None:
    model_mock = Mock(return_value=make_response(make_direct_message("收到")))
    monkeypatch.setattr(agent_graph, "call_deepseek", model_mock)

    run_customer_agent_graph("第一段会话", graph=CUSTOMER_AGENT_GRAPH, thread_id=uuid4().hex if explicit_ids else None)
    run_customer_agent_graph("第二段会话", graph=CUSTOMER_AGENT_GRAPH, thread_id=uuid4().hex if explicit_ids else None)

    second_messages = model_mock.call_args_list[1].args[0]
    assert [message["role"] for message in second_messages] == ["system", "user"]
    assert second_messages[-1]["content"] == "第二段会话"


def test_each_turn_gets_a_fresh_tool_round_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = [make_tool_call(
        name="get_order",
        arguments='{"order_id":"20260721001"}',
        call_id=f"call-round-{index}",
    ) for index in range(2)]
    model_mock = Mock(side_effect=[
        make_response(make_tool_message([calls[0]])),
        make_response(make_direct_message("第一轮完成")),
        make_response(make_tool_message([calls[1]])),
        make_response(make_direct_message("第二轮完成")),
    ])
    tool_mock = Mock(return_value='{"order_status":"待处理"}')
    monkeypatch.setattr(agent_graph, "MAX_TOOL_ROUNDS", 1)
    monkeypatch.setattr(agent_graph, "call_deepseek", model_mock)
    monkeypatch.setattr(agent_graph, "execute_tool", tool_mock)
    thread_id = uuid4().hex

    assert run_customer_agent_graph("查询订单", graph=CUSTOMER_AGENT_GRAPH, thread_id=thread_id) == ("第一轮完成", thread_id)
    assert run_customer_agent_graph("再查一次", graph=CUSTOMER_AGENT_GRAPH, thread_id=thread_id) == ("第二轮完成", thread_id)

    state = CUSTOMER_AGENT_GRAPH.get_state({"configurable": {"thread_id": thread_id}}).values
    assert state["tool_rounds"] == 1
    assert state["pending_tool_calls"] == []
    assert state["steps"] == ["call_model", "execute_tools", "call_model"] * 2
    assert tool_mock.call_count == 2
