import logging

from langgraph.checkpoint.memory import InMemorySaver
from sentence_transformers import SentenceTransformer

from serviceflow.agent.graph import build_customer_agent_graph
from serviceflow.agent.ticket_collection import build_ticket_collection_graph
from serviceflow.agent.workflow import (
    build_customer_workflow, run_customer_workflow,
)
from serviceflow.rag.embeddings import MODEL_NAME


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    agent_graph = build_customer_agent_graph(checkpointer=InMemorySaver())
    embedding_model = SentenceTransformer(MODEL_NAME)
    ticket_graph = build_ticket_collection_graph(checkpointer=InMemorySaver())
    # 编译好的图
    graph = build_customer_workflow(agent_graph=agent_graph, embedding_model=embedding_model, ticket_graph=ticket_graph)

    print(graph.get_graph().draw_mermaid())

    questions = [
        "帮订单20260721001创建售后工单",
        "左耳没有声音",
        "换货",
    ]

    thread_id: str | None = None

    for question in questions:
        answer, thread_id = run_customer_workflow(user_text=question, workflow_graph=graph, thread_id=thread_id)

        print(f"问题：{question}")
        print(f"会话 ID：{thread_id}")
        print(f"回答：{answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
