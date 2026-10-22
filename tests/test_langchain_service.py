"""固定模型与工具返回，验证编排逻辑；不调用网络或数据库。"""
from unittest.mock import Mock

import pytest
from httpx import Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import APITimeoutError, OpenAIError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from serviceflow.agent import langchain_service as service
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.llm.models import GetOrderArguments


def tool_request(name="get_order_tool", call_id="call-1"):
    return {"name": name, "args": {"order_id": "20260721001"}, "id": call_id, "type": "tool_call"}


@pytest.fixture
def setup_agent(monkeypatch):
    bound_model = Mock()
    model = Mock()
    model.bind_tools.return_value = bound_model
    factory = Mock(return_value=model)
    tool = Mock()
    tool.name = "get_order_tool"
    tool.invoke.side_effect = lambda call: ToolMessage(content='{"payment_status":"已支付"}', tool_call_id=call["id"])
    monkeypatch.setattr(service, "create_chat_model", factory)
    monkeypatch.setattr(service, "CUSTOMER_TOOLS", [tool])
    return factory, model, bound_model, tool


def test_direct_answer_does_not_execute_tool(setup_agent):
    factory, model, bound, tool = setup_agent
    bound.invoke.return_value = AIMessage(content="请提供订单号")
    assert service.run_langchain_agent("查一下订单") == "请提供订单号"
    factory.assert_called_once_with()
    model.bind_tools.assert_called_once_with([tool])
    messages = bound.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "查一下订单"
    tool.invoke.assert_not_called()


def test_tool_result_and_assistant_request_reach_next_model_call(setup_agent):
    _, _, bound, tool = setup_agent
    first = AIMessage(content="", tool_calls=[tool_request()])
    histories = []

    def invoke(messages):
        histories.append(list(messages))
        return first if len(histories) == 1 else AIMessage(content="订单已支付")

    bound.invoke.side_effect = invoke
    assert service.run_langchain_agent("查订单") == "订单已支付"
    tool.invoke.assert_called_once_with(first.tool_calls[0])
    assert len(histories) == 2
    assert histories[1][2] is first
    assert isinstance(histories[1][3], ToolMessage)
    assert histories[1][3].tool_call_id == first.tool_calls[0]["id"]


def test_multiple_tools_use_the_registered_objects(monkeypatch, setup_agent):
    _, _, bound, order_tool = setup_agent
    logistics_tool = Mock()
    logistics_tool.name = "get_logistics_tool"
    logistics_tool.invoke.return_value = ToolMessage(content="运输中", tool_call_id="call-2")
    monkeypatch.setattr(service, "CUSTOMER_TOOLS", [order_tool, logistics_tool])
    calls = [tool_request(), tool_request("get_logistics_tool", "call-2")]
    bound.invoke.side_effect = [AIMessage(content="", tool_calls=calls), AIMessage(content="已支付，运输中")]
    assert service.run_langchain_agent("查询订单和物流") == "已支付，运输中"
    order_tool.invoke.assert_called_once_with(calls[0])
    logistics_tool.invoke.assert_called_once_with(calls[1])


@pytest.mark.parametrize("error,message", [
    (APITimeoutError(request=Request("POST", "https://example.invalid")), "模型请求超时"),
    (OpenAIError("private provider detail"), "模型服务调用失败"),
])
def test_model_errors_are_converted_and_keep_cause(setup_agent, error, message):
    _, _, bound, tool = setup_agent
    bound.invoke.side_effect = error
    with pytest.raises(AgentServiceError, match=message) as caught:
        service.run_langchain_agent("查订单")
    assert caught.value.__cause__ is error
    assert "private provider detail" not in str(caught.value)
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("content,message", [("", "回答为空"), ("  ", "回答为空"), ([], "目前只支持字符串")])
def test_empty_or_unsupported_answer_is_rejected(setup_agent, content, message):
    _, _, bound, _ = setup_agent
    bound.invoke.return_value = AIMessage(content=content)
    with pytest.raises(AgentServiceError, match=message):
        service.run_langchain_agent("问题")


def test_unknown_tool_is_never_executed(setup_agent):
    _, _, bound, tool = setup_agent
    bound.invoke.return_value = AIMessage(content="", tool_calls=[tool_request("delete_database")])
    with pytest.raises(AgentServiceError, match="未定义的工具"):
        service.run_langchain_agent("查订单")
    tool.invoke.assert_not_called()


@pytest.mark.parametrize("error,message", [
    (OperationalError("select", {}, Exception("private database detail")), "工具执行超时或数据库不可用"),
    (SQLAlchemyTimeoutError("private pool detail"), "工具执行超时或数据库不可用"),
    (SQLAlchemyError("private database detail"), "工具执行失败"),
])
def test_database_errors_are_converted(setup_agent, error, message):
    _, _, bound, tool = setup_agent
    bound.invoke.return_value = AIMessage(content="", tool_calls=[tool_request()])
    tool.invoke.side_effect = error
    with pytest.raises(AgentServiceError, match=message) as caught:
        service.run_langchain_agent("查订单")
    assert caught.value.__cause__ is error
    assert bound.invoke.call_count == 1  # 不将失败伪装成工具结果继续调用模型。


def test_validation_error_is_converted(setup_agent):
    _, _, bound, tool = setup_agent
    with pytest.raises(ValidationError) as invalid:
        GetOrderArguments.model_validate({"order_id": ""})
    tool.invoke.side_effect = invalid.value
    bound.invoke.return_value = AIMessage(content="", tool_calls=[tool_request()])
    with pytest.raises(AgentServiceError, match="工具调用参数格式错误") as caught:
        service.run_langchain_agent("查订单")
    assert caught.value.__cause__ is invalid.value


def test_fourth_model_call_can_answer_after_three_tool_rounds(setup_agent):
    _, _, bound, tool = setup_agent
    bound.invoke.side_effect = [
        AIMessage(content="", tool_calls=[tool_request(call_id=f"call-{i}")]) for i in range(3)
    ] + [AIMessage(content="最终回答")]
    assert service.run_langchain_agent("查订单") == "最终回答"
    assert bound.invoke.call_count == 4
    assert tool.invoke.call_count == 3


def test_fourth_tool_round_is_blocked_before_execution(setup_agent):
    _, _, bound, tool = setup_agent
    bound.invoke.side_effect = [
        AIMessage(content="", tool_calls=[tool_request(call_id=f"call-{i}")]) for i in range(4)
    ]
    with pytest.raises(AgentServiceError, match="超过最大工具调用轮数"):
        service.run_langchain_agent("查订单")
    assert bound.invoke.call_count == 4
    assert tool.invoke.call_count == 3
