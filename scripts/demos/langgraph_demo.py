from operator import add
from typing import Annotated

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class CustomerServiceState(TypedDict):
    user_message: str
    intent: str
    answer: str
    steps: Annotated[list[str], add]


def classify_intent_node(
        state: CustomerServiceState,
) -> dict[str, str | list[str]]:
    # 读取state["user_message"]
    # 包含“订单”时，返回{"intent": "order_query"}
    # 否则返回{"intent": "other"}
    if "订单" in state["user_message"]:
        return {
            "intent": "order_query",
            "steps": ["classify_intent"],
        }

    return {
        "intent": "other",
        "steps": ["classify_intent"],
    }


def route_by_intent(
        state: CustomerServiceState,
) -> str:
    if state["intent"] == "order_query":
        return "order_query"
    return "other"


def handle_order_node(
        _state: CustomerServiceState,
) -> dict[str, str | list[str]]:
    return {
        "answer": "正在为您查询订单",
        "steps": ["handle_order"],
    }


def handle_other_node(
        _state: CustomerServiceState,
) -> dict[str, str | list[str]]:
    return {
        "answer": "暂时无法识别您的问题",
        "steps": ["handle_other"],
    }


def main() -> None:
    # 创建一张状态图，这张图中所有节点都使用 CustomerServiceState 传递数据。
    builder = StateGraph(CustomerServiceState)

    # 注册节点
    builder.add_node(
        "classify_intent",
        classify_intent_node,
    )

    builder.add_node(
        "handle_order",
        handle_order_node,
    )
    builder.add_node(
        "handle_other",
        handle_other_node,
    )

    # 连接节点
    builder.add_edge(
        START,
        "classify_intent",
    )

    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "order_query": "handle_order",
            "other": "handle_other",
        },
    )

    builder.add_edge(
        "handle_order",
        END,
    )

    builder.add_edge(
        "handle_other",
        END,
    )

    # 编译状态图
    graph = builder.compile()

    order_result = graph.invoke(
        {
            "user_message": "查询订单20260721001",
            "intent": "",
            "answer": "",
            "steps": [],
        },
    )

    print(order_result)

    other_result = graph.invoke(
        {
            "user_message": "耳机坏了",
            "intent": "",
            "answer": "",
            "steps": [],
        },
    )
    print(other_result)


if __name__ == "__main__":
    main()
