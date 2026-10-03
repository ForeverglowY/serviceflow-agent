from operator import add
from typing import Annotated, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam, ChatCompletionToolMessageParam, ChatCompletionUserMessageParam, )
from pydantic import ValidationError
from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLAlchemyTimeoutError,
)

from serviceflow.agent.runtime import (
    AGENT_SYSTEM_MESSAGE, AgentServiceError,
    MAX_TOOL_ROUNDS, call_deepseek,
)
from serviceflow.agent.tools import ToolNotFoundError, execute_tool


def build_customer_agent_graph(
        checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    _builder = StateGraph(AgentGraphState)
    _builder.add_node(
        "call_model",
        call_model_node,
    )

    _builder.add_node(
        "execute_tools",
        execute_tools_node,
    )

    _builder.add_node(
        "tool_limit",
        handle_tool_limit_node,
    )

    _builder.add_edge(
        START,
        "call_model",
    )

    _builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "finish": END,
            "tool_limit": "tool_limit",
            "execute_tools": "execute_tools",
        },
    )

    _builder.add_edge(
        "execute_tools",
        "call_model",
    )

    _builder.add_edge(
        "tool_limit",
        END,
    )

    return _builder.compile(checkpointer=checkpointer)


class AgentGraphState(TypedDict):
    # 发送给 DeepSeek 的完整消息历史
    messages: Annotated[
        list[ChatCompletionMessageParam],
        add,
    ]
    # 模型当前要求执行的工具
    pending_tool_calls: list[
        ChatCompletionMessageFunctionToolCall
    ]
    # 客服最终回答
    answer: str | None
    # 已执行的工具轮数
    tool_rounds: int
    # 节点执行轨迹
    steps: Annotated[list[str], add]


class AgentGraphStateUpdate(TypedDict, total=False):
    messages: list[ChatCompletionMessageParam]
    pending_tool_calls: list[
        ChatCompletionMessageFunctionToolCall
    ]
    answer: str | None
    tool_rounds: int
    steps: list[str]


def call_model_node(
        state: AgentGraphState,
) -> AgentGraphStateUpdate:
    response = call_deepseek(state["messages"])
    message = response.choices[0].message
    assistant_message = cast(
        ChatCompletionAssistantMessageParam,
        message.model_dump(exclude_none=True),
    )

    if not message.tool_calls:
        if message.content is None:
            raise AgentServiceError("DeepSeek没有返回内容")

        return {
            "messages": [assistant_message],
            "pending_tool_calls": [],
            "answer": message.content,
            "steps": ["call_model"],
        }

    function_tool_calls: list[
        ChatCompletionMessageFunctionToolCall
    ] = []

    for tool_call in message.tool_calls:
        if not isinstance(
                tool_call,
                ChatCompletionMessageFunctionToolCall,
        ):
            raise AgentServiceError(
                f"不支持的工具调用类型：{tool_call.type}"
            )

        function_tool_calls.append(tool_call)

    return {
        "messages": [assistant_message],
        "pending_tool_calls": function_tool_calls,
        "answer": None,
        "steps": ["call_model"],
    }


def execute_tools_node(
        state: AgentGraphState,
) -> AgentGraphStateUpdate:
    if not state["pending_tool_calls"]:
        raise AgentServiceError("没有待执行的工具调用")

    tool_messages: list[ChatCompletionMessageParam] = []
    for tool_call in state["pending_tool_calls"]:
        try:
            tool_content = execute_tool(
                tool_call.function.name,
                tool_call.function.arguments
            )
        except ToolNotFoundError as exception:
            raise AgentServiceError(
                str(exception)
            ) from exception
        except ValidationError as exception:
            raise AgentServiceError("工具调用参数格式错误") from exception
        except (OperationalError, SQLAlchemyTimeoutError,) as exception:
            raise AgentServiceError("工具执行超时或数据库不可用") from exception
        except SQLAlchemyError as exception:
            raise AgentServiceError("工具执行失败") from exception

        tool_message: ChatCompletionToolMessageParam = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_content,
        }
        tool_messages.append(tool_message)

    return {
        "messages": tool_messages,
        "pending_tool_calls": [],
        "tool_rounds": state["tool_rounds"] + 1,
        "steps": ["execute_tools"],
    }


def route_after_model(
        state: AgentGraphState,
) -> str:
    # 已经生成答案，正常结束
    if state["answer"] is not None:
        return "finish"

    # 没有答案且达到工具上限，终止
    if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
        return "tool_limit"

    # 没有答案、有待执行工具且未达到上限，执行工具
    if state["pending_tool_calls"]:
        return "execute_tools"

    # 什么都没有，说明模型状态异常
    raise AgentServiceError(
        "模型既未返回最终回答，也未请求调用工具"
    )


def handle_tool_limit_node(
        _state: AgentGraphState,
) -> AgentGraphStateUpdate:
    raise AgentServiceError(
        "超过最大工具调用次数"
    )


def run_customer_agent_graph(
        user_text: str,
        graph: CompiledStateGraph,
        thread_id: str | None = None,
) -> tuple[str, str]:
    # 对外对话入口
    if thread_id is None:
        thread_id = uuid4().hex

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    snapshot = graph.get_state(config)
    # 没有 messages，或者消息列表为空：新会话。
    is_new_conversation = not snapshot.values.get("messages")

    # 构建这一轮的输入
    turn_input = build_agent_turn_input(user_text=user_text, is_new_conversation=is_new_conversation)

    result = graph.invoke(turn_input, config=config)
    answer = result["answer"]

    if answer is None:
        raise AgentServiceError(
            "Agent没有生成最终回答"
        )

    return answer, thread_id


def build_agent_turn_input(
        user_text: str,
        is_new_conversation: bool,
) -> AgentGraphState:
    # 它负责构造本轮输入，要求如下：
    # 1.创建本轮user_message。
    # 2.创建一个消息列表。如果is_new_conversation为True，先加入AGENT_SYSTEM_MESSAGE。
    # 3.加入本轮user_message。
    # 4.返回完整状态字典，其他字段设置为：
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_text,
    }

    messages: list[ChatCompletionMessageParam] = []
    if is_new_conversation:
        messages.append(AGENT_SYSTEM_MESSAGE)

    messages.append(user_message)

    return {
        "messages": messages,
        "pending_tool_calls": [],
        "answer": None,
        "tool_rounds": 0,
        "steps": [],
    }


checkpointer = InMemorySaver()

CUSTOMER_AGENT_GRAPH = build_customer_agent_graph(
    checkpointer=checkpointer,
)
