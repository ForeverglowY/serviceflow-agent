from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from scripts.demos.langgraph_memory_demo import ConversationState, reply_node

DB_URI = (
    "postgresql://"
    "agent:agent_password@localhost:5432/agent_study"
)


def main() -> None:
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
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

        graph = builder.compile(checkpointer=checkpointer)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "postgres-demo-001",
            }
        }

        print("本次运行之前的状态：")
        print(graph.get_state(config).values)

        user_text = input("请输入消息：")

        result = graph.invoke(
            {"messages": [user_text]},
            config=config,
        )

        print("本次运行之后的状态：")
        print(result)


if __name__ == "__main__":
    main()
