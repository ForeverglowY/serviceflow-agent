from operator import add
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from serviceflow.agent.graph import build_agent_turn_input


class ConversationState(TypedDict):
    messages: Annotated[list[str], add]


class ConversationStateUpdate(TypedDict, total=False):
    messages: list[str]


def reply_node(
        state: ConversationState,
) -> ConversationStateUpdate:
    last_message = state["messages"][-1]
    last_message = "客服收到: " + last_message

    return {
        "messages": [last_message]
    }


def main() -> None:
    builder = StateGraph(ConversationState)

    builder.add_node(
        "reply",
        reply_node,
    )

    builder.add_edge(
        START,
        "reply",
    )

    builder.add_edge(
        "reply",
        END,
    )

    # 创建内存 checkpointer
    checkpointer = InMemorySaver()

    # 编译图，并传入 checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "customer-001",
        }
    }

    first_result = graph.invoke(
        {
            "messages": [
                "查询订单20260721001",
            ]
        },
        config=config,
    )

    print(first_result)

    second_result = graph.invoke(
        {
            "messages": [
                "再帮我看看物流",
            ]
        },
        config=config,
    )

    print(second_result)

    customer_001_state = graph.get_state(config)

    print("customer-001 最新状态：")
    print(customer_001_state.values)
    print("下一批节点：", customer_001_state.next)
    print("checkpoint配置：", customer_001_state.config)

    other_config: RunnableConfig = {
        "configurable": {
            "thread_id": "customer-002",
        }
    }

    third_result = graph.invoke(
        {
            "messages": [
                "我的耳机坏了",
            ]
        },
        config=other_config,
    )

    print(third_result)

    customer_002_state = graph.get_state(other_config)

    print("customer-002 最新状态：")
    print(customer_002_state.values)

    history = list(
        graph.get_state_history(config)
    )

    print("customer-001 checkpoint数量：", len(history))

    for snapshot in history:
        print("-" * 50)
        print("创建时间：", snapshot.created_at)
        print("状态：", snapshot.values)
        print("下一批节点：", snapshot.next)
        print("元数据：", snapshot.metadata)
        print("配置：", snapshot.config)
        if snapshot.metadata.get("step") == 1:
            historical_state = graph.get_state(snapshot.config)

            print("第一轮结束时的状态：")
            print(historical_state.values)
            break

    print("当前最新状态：")
    print(graph.get_state(config).values)

    branch_result = graph.invoke(
        {
            "messages": [
                "我想申请退货",
            ]
        },
        config=historical_state.config,
    )

    print("新分支的状态：")
    print(branch_result)

    print(build_agent_turn_input("查询订单20260721001", True))
    print(build_agent_turn_input("那它到哪里了？", False))

if __name__ == "__main__":
    main()
