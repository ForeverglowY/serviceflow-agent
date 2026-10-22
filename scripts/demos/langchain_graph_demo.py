from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver

from serviceflow.agent.langchain_graph import build_langchain_agent_graph, resume_langchain_agent_graph, \
    run_langchain_agent_graph
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.settings import settings


def main() -> None:
    with PostgresSaver.from_conn_string(
            settings.checkpoint_database_url
    ) as checkpointer:
        checkpointer.setup()

        graph = build_langchain_agent_graph(checkpointer=checkpointer)

        thread_id = uuid4().hex

        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        existing_thread_id = input("会话ID（留空则新建）：").strip()
        thread_id = existing_thread_id or uuid4().hex

        print("当前会话ID：", thread_id)

        while True:
            user_text = input("请输入问题：")
            if user_text == "exit":
                break

            if user_text == "" or not user_text.strip():
                continue

            snapshot = graph.get_state(config=config)

            if user_text == "/retry":
                try:
                    answer, thread_id = resume_langchain_agent_graph(
                        graph=graph,
                        thread_id=thread_id,
                    )
                except AgentServiceError as exception:
                    print(f"客服：恢复失败：{exception}")
                    continue
            elif user_text == "/new":
                # 新建对话
                thread_id = uuid4().hex
                config: RunnableConfig = {
                    "configurable": {
                        "thread_id": thread_id,
                    },
                }
                print("已开启新会话")
                continue
            else:
                if snapshot.next:
                    print("客服：请先输入 /retry 恢复上次请求。")
                    continue

                try:
                    answer, thread_id = run_langchain_agent_graph(user_text=user_text, graph=graph,
                                                                  thread_id=thread_id)
                except AgentServiceError as exception:
                    print(f"客服：本次处理失败：{exception}")
                    continue

            print("answer=", answer)


if __name__ == "__main__":
    main()
