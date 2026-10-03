from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from serviceflow.agent.ticket_collection import TicketCollectionStatus, build_ticket_collection_graph, \
    run_ticket_collection_turn
from serviceflow.llm.service import LLMServiceError


def main() -> None:
    thread_id = uuid4().hex
    checkpointer = InMemorySaver()
    graph = build_ticket_collection_graph(checkpointer)

    while True:
        user_message = input("客户：").strip()
        if user_message == "":
            continue
        if user_message == "退出":
            break

        # 提取本轮信息
        try:
            result = run_ticket_collection_turn(user_message=user_message, graph=graph, thread_id=thread_id)
        except LLMServiceError as exception:
            print(f"客服：本次信息提取失败，请重试。原因：{exception}")
            continue

        print(f"客服：{result['answer']}")

        if result["status"] == TicketCollectionStatus.READY_FOR_CONFIRMATION.value:
            print("收集结束：", result)
            break

        if result["status"] == TicketCollectionStatus.CANCELED.value:
            print("用户结束请求：", result)
            break


if __name__ == "__main__":
    main()
