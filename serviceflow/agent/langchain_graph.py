from functools import partial
from operator import add
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from openai import APITimeoutError, OpenAIError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError, SQLAlchemyError, TimeoutError as SQLAlchemyTimeoutError

from serviceflow.agent.langchain_tools import CUSTOMER_TOOLS
from serviceflow.agent.runtime import AgentServiceError, MAX_TOOL_ROUNDS
from serviceflow.llm.chat_model import create_chat_model
from serviceflow.observability import traced_call


class LangChainAgentState(TypedDict):
    # 消息历史
    messages: Annotated[list[BaseMessage], add_messages]
    # 当前待执行的工具请求
    pending_tool_calls: list[ToolCall]
    # 最终回答
    answer: str | None
    # 已执行工具轮数
    tool_rounds: int
    # 节点执行轨迹
    steps: Annotated[list[str], add]


class LangChainAgentStateUpdate(TypedDict, total=False):
    # 消息历史
    messages: list[BaseMessage]
    # 当前待执行的工具请求
    pending_tool_calls: list[ToolCall]
    # 最终回答
    answer: str | None
    # 已执行工具轮数
    tool_rounds: int
    # 节点执行轨迹
    steps: list[str]


def call_model_node(
        state: LangChainAgentState,
        *,
        model_with_tools: Runnable[LanguageModelInput, AIMessage],
) -> LangChainAgentStateUpdate:
    try:
        response = traced_call(
            "agent.call_model",
            model_with_tools.invoke,
            state["messages"],
        )
    except APITimeoutError as exception:
        raise AgentServiceError("模型请求超时") from exception
    except OpenAIError as exception:
        raise AgentServiceError("模型服务调用失败") from exception

    if response.tool_calls:
        return {
            "messages": [response],
            "pending_tool_calls": response.tool_calls,
            "answer": None,
            "steps": ["call_model"],
        }

    content = response.content
    if not isinstance(content, str):
        raise AgentServiceError("模型返回了不支持的内容类型")

    if not content.strip():
        raise AgentServiceError("模型没有返回有效回答")

    return {
        "messages": [response],
        "pending_tool_calls": [],
        "answer": content,
        "steps": ["call_model"],
    }


def execute_tools_node(
        state: LangChainAgentState,
) -> LangChainAgentStateUpdate:
    # 模型节点生成工具请求
    # ↓
    # 执行工具节点：调用Python工具，获取数据库数据
    # ↓
    # 把ToolMessage加入历史，再交给模型
    tool_registry = {
        current_tool.name: current_tool
        for current_tool in CUSTOMER_TOOLS
    }

    tool_messages: list[ToolMessage] = []

    for tool_call in state["pending_tool_calls"]:
        # tool_call 是调用单，包括了 name、args、id
        current_tool = tool_registry.get(tool_call["name"])
        if not current_tool:
            raise AgentServiceError("该工具不存在")

        try:
            tool_message = traced_call(
                f"tool.{tool_call['name']}",
                current_tool.invoke,
                tool_call,
            )

            tool_messages.append(tool_message)
        except ValidationError as exception:
            raise AgentServiceError("工具调用参数格式错误") from exception
        except (OperationalError, SQLAlchemyTimeoutError) as exception:
            raise AgentServiceError("工具执行超时或数据库不可用") from exception
        except SQLAlchemyError as exception:
            raise AgentServiceError("工具执行失败") from exception

    return {
        "messages": tool_messages,
        "pending_tool_calls": [],
        "tool_rounds": state["tool_rounds"] + 1,
        "steps": ["execute_tools"],
    }


def route_after_model(
        state: LangChainAgentState,
) -> str:
    if not state["pending_tool_calls"]:
        return "finish"
    if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
        return "tool_limit"
    return "execute_tools"


def tool_limit_node(
        state: LangChainAgentState,
) -> LangChainAgentStateUpdate:
    raise AgentServiceError("超过最大工具调用轮数")


def build_langchain_agent_graph(
        checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    model = create_chat_model()
    model_with_tools = model.bind_tools(CUSTOMER_TOOLS)

    builder = StateGraph(LangChainAgentState)

    builder.add_node(
        "call_model",
        partial(call_model_node, model_with_tools=model_with_tools),
    )
    builder.add_node(
        "tool_limit",
        tool_limit_node,
    )
    builder.add_node(
        "execute_tools",
        execute_tools_node,
    )

    builder.add_edge(
        START,
        "call_model",
    )
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "finish": END,
            "tool_limit": "tool_limit",
            "execute_tools": "execute_tools",
        }
    )
    builder.add_edge(
        "tool_limit",
        END
    )
    builder.add_edge(
        "execute_tools",
        "call_model",
    )

    return builder.compile(checkpointer=checkpointer)


def run_langchain_agent_graph(
        user_text: str,
        *,
        graph: CompiledStateGraph,
        thread_id: str | None = None,
) -> tuple[str, str]:
    if thread_id is None:
        thread_id = uuid4().hex

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        },
    }

    system_message = SystemMessage(
        content="你是一名电商客服。查询真实订单信息时必须使用提供的工具，不得编造数据。物流进度必须依据物流工具返回结果回答；订单工具不提供物流轨迹。"
                "tracking_number 为 null 仅表示当前订单记录未提供物流单号，不能据此推断未发货或暂无物流信息。用户只询问商品和支付状态时，不主动推断物流情况。"
                "当前仅支持按订单号查询订单和物流。不支持通过手机号、姓名或其他信息查单，不得要求用户提供这些信息，也不得承诺具备未提供的查询能力。",
    )

    snapshot = graph.get_state(config=config)

    if snapshot.next:
        raise AgentServiceError("上次请求尚未完成，请先恢复处理")

    human_message = HumanMessage(content=user_text)

    if not snapshot.values:
        # 第一次输入
        turn_input: LangChainAgentStateUpdate = {
            "messages": [system_message, human_message],
            "pending_tool_calls": [],
            "answer": None,
            "tool_rounds": 0,
            "steps": []
        }
    else:
        turn_input: LangChainAgentStateUpdate = {
            "messages": [human_message],
            "pending_tool_calls": [],
            "answer": None,
            "tool_rounds": 0,
        }

    result = graph.invoke(turn_input, config=config)

    answer = result["answer"]

    if not isinstance(answer, str) or not answer.strip():
        raise AgentServiceError("本次客服回答为空")

    return answer, thread_id


def resume_langchain_agent_graph(
        *,
        graph: CompiledStateGraph,
        thread_id: str,
) -> tuple[str, str]:
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    snapshot = graph.get_state(config=config)

    if not snapshot.values:
        raise AgentServiceError("会话不存在")

    if not snapshot.next:
        raise AgentServiceError("没有待恢复的请求")

    result = graph.invoke(None, config=config)

    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise AgentServiceError("本次客服回答为空")

    return answer, thread_id
