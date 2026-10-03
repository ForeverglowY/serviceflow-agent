import logging
from functools import partial
from operator import add
from uuid import uuid4

from langgraph.constants import END, START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from sentence_transformers import SentenceTransformer
from typing_extensions import Annotated, TypedDict

from serviceflow.agent.graph import run_customer_agent_graph
from serviceflow.agent.routing import select_customer_route
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.agent.ticket_collection import is_ticket_collection_active, run_ticket_collection_turn
from serviceflow.llm.models import IntentResult
from serviceflow.llm.service import LLMServiceError, classify_intent
from serviceflow.rag.service import RAGServiceError, answer_policy_question

logger = logging.getLogger(__name__)


class CustomerWorkflowState(TypedDict):
    user_message: str
    intent_result: IntentResult | None
    answer: str | None
    thread_id: str | None
    steps: Annotated[list[str], add]


class CustomerWorkflowStateUpdate(TypedDict, total=False):
    intent_result: IntentResult | None
    answer: str | None
    thread_id: str | None
    steps: list[str]


# 理解客户说了什么，生成意图
def classify_customer_intent_node(
        state: CustomerWorkflowState,
) -> CustomerWorkflowStateUpdate:
    # 读取state["user_message"]。
    user_message = state["user_message"]
    # 调用已有的classify_intent()，它会返回经过Pydantic校验的IntentResult。
    try:
        intent_result = classify_intent(user_message)
    except LLMServiceError as exception:
        raise AgentServiceError(
            "客户意图识别失败"
        ) from exception
    # 返回状态更新：
    return {
        "intent_result": intent_result,
        "steps": ["classify_intent"],
    }


# 根据已经生成的意图，决定下一步去哪
def route_customer_workflow(
        state: CustomerWorkflowState,
) -> str:
    intent_result = state["intent_result"]
    if intent_result is None:
        raise AgentServiceError("缺少意图识别结果")

    return select_customer_route(intent_result.intent)


def handle_fallback_node(
        state: CustomerWorkflowState,
) -> CustomerWorkflowStateUpdate:
    return {
        "answer": "我可以帮助您查询订单和物流、了解售后政策，或申请售后工单。请补充您的具体需求。",
        "steps": ["handle_fallback"],
    }


def handle_ticket_node(
        state: CustomerWorkflowState,
        *,
        ticket_graph: CompiledStateGraph,
) -> CustomerWorkflowStateUpdate:
    thread_id = state["thread_id"]
    if thread_id is None or not thread_id.strip():
        raise AgentServiceError("工单收集缺少会话编号")

    try:
        result = run_ticket_collection_turn(user_message=state["user_message"], graph=ticket_graph, thread_id=thread_id)
    except LLMServiceError as exception:
        raise AgentServiceError("售后信息收集失败") from exception

    answer = result["answer"]

    if answer is None or not answer.strip():
        raise AgentServiceError("没有生成回答")

    return {
        "answer": answer,
        "thread_id": thread_id,
        "steps": ["handle_ticket"],
    }


def handle_tools_node(
        state: CustomerWorkflowState,
        *,
        agent_graph: CompiledStateGraph,
) -> CustomerWorkflowStateUpdate:
    answer, thread_id = run_customer_agent_graph(user_text=state["user_message"], graph=agent_graph,
                                                 thread_id=state["thread_id"])

    return {
        "answer": answer,
        "thread_id": thread_id,
        "steps": ["handle_tools"]
    }


def handle_rag_node(
        state: CustomerWorkflowState,
        *,
        embedding_model: SentenceTransformer,
) -> CustomerWorkflowStateUpdate:
    user_message = state["user_message"]

    try:
        answer = answer_policy_question(
            model=embedding_model,
            user_question=user_message,
        )
    except RAGServiceError as exception:
        raise AgentServiceError(
            "售后政策问答失败"
        ) from exception

    return {
        "answer": answer,
        "steps": ["handle_rag"]
    }


def route_customer_entry(
        state: CustomerWorkflowState,
        *,
        ticket_graph: CompiledStateGraph,
) -> str:
    # 工作流刚开始, 判断是否继续已有工单收集
    thread_id = state["thread_id"]
    if thread_id is None or not thread_id.strip():
        raise AgentServiceError("客服工作流缺少会话编号")

    is_active = is_ticket_collection_active(graph=ticket_graph, thread_id=thread_id)
    if is_active:
        return "continue_ticket"

    return "classify"


# 组装节点和边，创建可运行的图
def build_customer_workflow(
        agent_graph: CompiledStateGraph,
        embedding_model: SentenceTransformer,
        ticket_graph: CompiledStateGraph,
) -> CompiledStateGraph:
    # agent_graph 是工具 agent 图
    # embedding_model 是向量模型
    builder = StateGraph(CustomerWorkflowState)

    builder.add_node(
        "classify_intent",
        classify_customer_intent_node
    )
    builder.add_node(
        "handle_fallback",
        handle_fallback_node
    )
    builder.add_node(
        "handle_ticket",
        partial(handle_ticket_node, ticket_graph=ticket_graph),
    )
    builder.add_node(
        "handle_tools",
        partial(handle_tools_node, agent_graph=agent_graph)
    )
    builder.add_node(
        "handle_rag",
        partial(handle_rag_node, embedding_model=embedding_model)
    )

    builder.add_conditional_edges(
        START,
        partial(route_customer_entry, ticket_graph=ticket_graph),
        {
            "continue_ticket": "handle_ticket",
            "classify": "classify_intent",
        }
    )

    builder.add_conditional_edges(
        "classify_intent",
        route_customer_workflow,
        {
            "tools": "handle_tools",
            "rag": "handle_rag",
            "ticket": "handle_ticket",
            "fallback": "handle_fallback",
        }
    )

    builder.add_edge(
        "handle_tools",
        END,
    )
    builder.add_edge(
        "handle_rag",
        END,
    )
    builder.add_edge(
        "handle_ticket",
        END,
    )
    builder.add_edge(
        "handle_fallback",
        END,
    )

    return builder.compile()


# 用这张图处理一次客户请求
def run_customer_workflow(
        user_text: str,
        workflow_graph: CompiledStateGraph,
        thread_id: str | None = None,
) -> tuple[str, str]:
    if thread_id is None:
        thread_id = uuid4().hex

    workflow_state = CustomerWorkflowState(
        user_message=user_text,
        intent_result=None,
        answer=None,
        thread_id=thread_id,
        steps=[],
    )

    result = workflow_graph.invoke(workflow_state)

    answer = result["answer"]

    if answer is None or not answer.strip():
        raise AgentServiceError("客服工作流没有返回回答")

    result_thread_id = result["thread_id"]

    if result_thread_id is None:
        raise AgentServiceError("客服工作流没有返回会话编号")

    logger.info(
        "客服工作流完成 thread_id=%s steps=%s",
        result_thread_id,
        result["steps"],
    )

    return answer, result_thread_id
