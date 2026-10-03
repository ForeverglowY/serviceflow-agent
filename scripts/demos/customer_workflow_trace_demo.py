import json
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from sentence_transformers import SentenceTransformer

from serviceflow.agent.graph import build_customer_agent_graph
from serviceflow.agent.workflow import CustomerWorkflowState, build_customer_workflow
from serviceflow.rag.embeddings import MODEL_NAME


def main() -> None:
    checkpointer = InMemorySaver()
    agent_graph = build_customer_agent_graph(checkpointer)
    embedding_model = SentenceTransformer(MODEL_NAME)
    # 编译好的图
    graph = build_customer_workflow(agent_graph=agent_graph, embedding_model=embedding_model)

    print(graph.get_graph().draw_mermaid())

    thread_id = uuid4().hex

    workflow_state = CustomerWorkflowState(
        user_message="帮我查询订单20260721001，并看看它的物流到哪里了。",
        intent_result=None,
        answer=None,
        thread_id=thread_id,
        steps=[],
    )
    trace_states: list[dict] = []
    for state in graph.stream(workflow_state, stream_mode="values"):
        # 把当前状态转换成可保存的数据
        state_dict = serialize_workflow_state(state)

        trace_states.append(state_dict)

    print("记录数量：", len(trace_states))
    # ensure_ascii = False：中文直接显示为中文。
    # indent = 4: 两个空格缩进
    json_text = json.dumps(trace_states, ensure_ascii=False, indent=4)

    # 指定保存位置
    output_path = Path("outputs/workflow_traces") / f"{thread_id}.json"
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(json_text, encoding="utf-8")

    print(f"执行记录已保存：{output_path.resolve()}")


def serialize_workflow_state(
        state: CustomerWorkflowState,
) -> dict:
    state_dict = state.copy()
    intent_result = state["intent_result"]
    if intent_result is not None:
        state_dict["intent_result"] = intent_result.model_dump(mode='json')

    return state_dict


if __name__ == "__main__":
    main()
