from operator import add
from typing import Annotated

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

MAX_TOOL_ROUNDS = 3


class ToolLoopState(TypedDict):
    # 客户问题
    user_message: str
    # 模型是否要求调用工具
    needs_tool: bool
    # 工具返回的数据
    tool_result: str | None
    # 最终回答
    answer: str | None
    # 已经执行了多少轮工具
    tool_rounds: int
    # 实际执行路径
    steps: Annotated[list[str], add]


class ToolLoopStateUpdate(TypedDict, total=False):
    needs_tool: bool
    tool_result: str | None
    answer: str | None
    tool_rounds: int
    steps: list[str]


def call_model_node(
        state: ToolLoopState,
) -> ToolLoopStateUpdate:
    if state["tool_result"] is None:
        # 说明模型还没获取真实数据 需要调用工具
        return {
            "needs_tool": True,
            "answer": None,
            "steps": ["call_model"],
        }

    # 已经获取真实数据
    return {
        "needs_tool": False,
        "answer": f"查询结果：{state['tool_result']}",
        "steps": ["call_model"],
    }


def execute_tool_node(
        state: ToolLoopState,
) -> ToolLoopStateUpdate:
    return {
        "tool_result": "订单状态：待处理",
        "tool_rounds": state["tool_rounds"] + 1,
        "steps": ["execute_tool"],
    }


def route_after_model(
        state: ToolLoopState,
) -> str:
    if not state["needs_tool"]:
        return "finish"

    if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
        return "tool_limit"

    return "execute_tool"


def handle_tool_limit_node(
        _state: ToolLoopState,
) -> ToolLoopStateUpdate:
    return {
        "needs_tool": False,
        "answer": "超过最大工具调用次数",
        "steps": ["tool_limit"],
    }


def main() -> None:
    initial_state: ToolLoopState = {
        "user_message": "查询订单20260721001",
        "needs_tool": False,
        "tool_result": None,
        "answer": None,
        "tool_rounds": 0,
        "steps": [],
    }

    builder = StateGraph(ToolLoopState)

    builder.add_node(
        "call_model",
        call_model_node,
    )

    builder.add_node(
        "execute_tool",
        execute_tool_node,
    )

    builder.add_node(
        "tool_limit",
        handle_tool_limit_node,
    )

    builder.add_edge(
        START,
        "call_model",
    )

    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_tool": "execute_tool",
            "finish": END,
            "tool_limit": "tool_limit",
        },
    )

    builder.add_edge(
        "execute_tool",
        "call_model",
    )

    builder.add_edge(
        "tool_limit",
        END,
    )

    graph = builder.compile()
    result = graph.invoke(initial_state)

    print(result)


if __name__ == "__main__":
    main()
