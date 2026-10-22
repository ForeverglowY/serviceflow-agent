"""检查点恢复测试：模型与工具均为替身，不访问网络或数据库。"""
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.exc import OperationalError

from serviceflow.agent import langchain_graph as module
from serviceflow.agent.runtime import AgentServiceError


@pytest.fixture
def environment(monkeypatch):
    bound = Mock()
    model = Mock()
    model.bind_tools.return_value = bound
    monkeypatch.setattr(module, "create_chat_model", Mock(return_value=model))
    tool = Mock()
    tool.name = "get_order_tool"
    monkeypatch.setattr(module, "CUSTOMER_TOOLS", [tool])
    graph = module.build_langchain_agent_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "recovery-test"}}
    state = {
        "messages": [HumanMessage(content="查订单20260721001")],
        "pending_tool_calls": [], "answer": None, "tool_rounds": 0, "steps": [],
    }
    call = {"name": tool.name, "args": {"order_id": "20260721001"},
            "id": "call-1", "type": "tool_call"}
    return graph, config, state, bound, tool, call


def test_failed_tool_resumes_without_repeating_completed_model_node(environment):
    graph, config, state, bound, tool, call = environment
    bound.invoke.side_effect = [AIMessage(content="", tool_calls=[call]), AIMessage(content="订单已支付")]
    tool.invoke.side_effect = [
        OperationalError("SELECT", {}, RuntimeError("offline")),
        ToolMessage(content="已支付", tool_call_id="call-1"),
    ]
    with pytest.raises(AgentServiceError, match="数据库不可用"):
        graph.invoke(state, config=config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("execute_tools",)
    assert snapshot.values["tool_rounds"] == 0
    assert snapshot.values["steps"] == ["call_model"]
    assert bound.invoke.call_count == 1

    answer, thread_id = module.resume_langchain_agent_graph(graph=graph, thread_id="recovery-test")
    assert answer == "订单已支付"
    assert thread_id == "recovery-test"
    result = graph.get_state(config).values
    assert result["tool_rounds"] == 1
    assert result["steps"] == ["call_model", "execute_tools", "call_model"]
    assert bound.invoke.call_count == 2
    assert tool.invoke.call_count == 2
    assert len([m for m in result["messages"] if isinstance(m, HumanMessage)]) == 1
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 1
    assert not graph.get_state(config).next


def test_failed_final_model_resumes_without_reexecuting_tool(environment):
    graph, config, state, bound, tool, call = environment
    bound.invoke.side_effect = [AIMessage(content="", tool_calls=[call]),
                                AIMessage(content=""), AIMessage(content="查询完成")]
    tool.invoke.return_value = ToolMessage(content="已支付", tool_call_id="call-1")
    with pytest.raises(AgentServiceError, match="有效回答"):
        graph.invoke(state, config=config)
    assert graph.get_state(config).next == ("call_model",)
    result = graph.invoke(None, config=config)
    assert result["answer"] == "查询完成"
    assert result["tool_rounds"] == 1
    assert tool.invoke.call_count == 1
    assert bound.invoke.call_count == 3


def test_tool_limit_is_not_reset_by_resume(environment, monkeypatch):
    graph, config, state, bound, tool, call = environment
    monkeypatch.setattr(module, "MAX_TOOL_ROUNDS", 0)
    bound.invoke.return_value = AIMessage(content="", tool_calls=[call])
    with pytest.raises(AgentServiceError, match="最大工具调用轮数"):
        graph.invoke(state, config=config)
    assert graph.get_state(config).next == ("tool_limit",)
    with pytest.raises(AgentServiceError, match="最大工具调用轮数"):
        graph.invoke(None, config=config)
    bound.invoke.assert_called_once()
    tool.invoke.assert_not_called()


def test_resume_rejects_missing_conversation(environment):
    graph, _, _, bound, tool, _ = environment
    with pytest.raises(AgentServiceError, match="会话不存在"):
        module.resume_langchain_agent_graph(graph=graph, thread_id="missing")
    bound.invoke.assert_not_called()
    tool.invoke.assert_not_called()


def test_resume_rejects_completed_conversation(environment):
    graph, config, state, bound, tool, _ = environment
    bound.invoke.return_value = AIMessage(content="你好")
    graph.invoke(state, config=config)
    with pytest.raises(AgentServiceError, match="没有待恢复的请求"):
        module.resume_langchain_agent_graph(graph=graph, thread_id="recovery-test")
    bound.invoke.assert_called_once()
    tool.invoke.assert_not_called()


def test_model_is_created_and_bound_once_per_graph(monkeypatch):
    bound = Mock()
    bound.invoke.return_value = AIMessage(content="你好")
    model = Mock()
    model.bind_tools.return_value = bound
    factory = Mock(return_value=model)
    monkeypatch.setattr(module, "create_chat_model", factory)
    monkeypatch.setattr(module, "CUSTOMER_TOOLS", [])
    graph = module.build_langchain_agent_graph(checkpointer=InMemorySaver())
    for message in ("你好", "还能帮我什么？"):
        answer, thread_id = module.run_langchain_agent_graph(
            message, graph=graph, thread_id="reuse-model",
        )
        assert answer == "你好"
        assert thread_id == "reuse-model"
    factory.assert_called_once_with()
    model.bind_tools.assert_called_once_with([])
    assert bound.invoke.call_count == 2
