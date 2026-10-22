import logging
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from langgraph.checkpoint.postgres import PostgresSaver
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from serviceflow.agent.langchain_graph import build_langchain_agent_graph
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.agent.ticket_collection import TicketRecoveryRequiredError, TicketResumeNotRequiredError, \
    TicketSessionNotFoundError, build_ticket_collection_graph, \
    resume_ticket_collection
from serviceflow.agent.workflow import build_customer_workflow, run_customer_workflow
from serviceflow.db.database import engine
from serviceflow.db.models import LogisticsTable, OrderTable, ProductTable, TicketTable
from serviceflow.llm.langchain_service import classify_intent, create_intent_classifier
from serviceflow.llm.models import AgentRequest, AgentResponse, IntentRequest, IntentResult
from serviceflow.llm.service import LLMServiceError
from serviceflow.llm.streaming import stream_customer_events
from serviceflow.models import Logistics, Order, Product, Ticket, TicketCreate
from serviceflow.observability import current_trace_id
from serviceflow.rag.embeddings import MODEL_NAME
from serviceflow.settings import settings
from serviceflow.ticket.service import TicketCreateConflictError, TicketEligibilityRejectedError, \
    TicketOrderNotFoundError, \
    create_ticket as create_ticket_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    with PostgresSaver.from_conn_string(
            settings.checkpoint_database_url,
    ) as checkpointer:
        checkpointer.setup()

        # 查询执行器
        app.state.agent_graph = build_langchain_agent_graph(checkpointer=checkpointer)

        # 售后流程执行器
        app.state.ticket_graph = build_ticket_collection_graph(checkpointer=checkpointer)

        # 向量模型
        app.state.embedding_model = SentenceTransformer(MODEL_NAME)

        app.state.classifier = create_intent_classifier()

        # 总调度，判断应该进入哪个分支
        app.state.workflow_graph = build_customer_workflow(
            agent_graph=app.state.agent_graph,
            embedding_model=app.state.embedding_model,
            ticket_graph=app.state.ticket_graph,
            classifier=app.state.classifier,
        )

        # 用户请求
        #     ↓
        # workflow_graph：总调度
        #     ├─ 订单／物流 → agent_graph
        #     ├─ 政策问答   → RAG函数 → embedding_model检索 + 大模型回答
        #     ├─ 售后申请   → ticket_graph
        #     └─ 其他问题   → 兜底回答
        print("服务启动：PostgreSQL 客服工作流已创建")

        yield

    print("服务关闭：Checkpointer 连接已关闭")


app = FastAPI(title="智能客服与工单处理 Agent", lifespan=lifespan)


@app.middleware("http")
async def request_trace(request: Request, call_next):
    trace_id = uuid4().hex
    token = current_trace_id.set(trace_id)
    started = perf_counter()
    try:
        response = await call_next(request)
        response.headers.setdefault("X-Trace-ID", trace_id)
        logging.getLogger(__name__).info(
            "http_response_ready trace_id=%s method=%s status=%d elapsed_ms=%.2f",
            trace_id, request.method, response.status_code, (perf_counter() - started) * 1000,
        )
        return response
    finally:
        current_trace_id.reset(token)


@app.get("/health", tags=["系统"], summary="服务健康检查", )
def health() -> dict:
    return {"status": "ok"}


@app.get("/orders/{order_id}", response_model=Order, tags=["订单"], summary="根据订单号查询订单", )
def get_order(order_id: str) -> Order:
    with Session(engine) as session:
        order_record = session.get(OrderTable, order_id)

        if order_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="订单不存在", )

        return Order.model_validate(order_record, from_attributes=True, )


@app.get("/products/{product_id}", response_model=Product, tags=["商品"], summary="根据商品编号查询商品", )
def get_product(product_id: str) -> Product:
    with Session(engine) as session:
        product_record = session.get(ProductTable, product_id)
        if product_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="商品不存在", )

        return Product.model_validate(product_record, from_attributes=True)


@app.get("/logistics/{order_id}", response_model=Logistics, tags=["物流"], summary="根据订单号查询物流", )
def get_logistic(order_id: str) -> Logistics:
    with Session(engine) as session:
        logistic_record = session.get(LogisticsTable, order_id)

        if logistic_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="物流信息不存在", )
        return Logistics.model_validate(logistic_record, from_attributes=True)


@app.post("/tickets/preview", response_model=Ticket, tags=["工单"], summary="校验并预览工单", )
def preview_ticket(ticket: Ticket) -> Ticket:
    return ticket


@app.post("/tickets", response_model=Ticket, tags=["工单"], summary="创建工单", status_code=status.HTTP_201_CREATED, )
def create_ticket(ticket_create: TicketCreate) -> Ticket:
    try:
        return create_ticket_service(ticket_create=ticket_create)
    except TicketOrderNotFoundError as exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="订单不存在",
        ) from exception
    except TicketCreateConflictError as exception:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="工单创建冲突",
        ) from exception
    except TicketEligibilityRejectedError as exception:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exception),
        ) from exception


@app.get("/tickets/{ticket_id}", response_model=Ticket, tags=["工单"], summary="根据工单编号查询工单")
def get_ticket(ticket_id: str) -> Ticket:
    with Session(engine) as session:
        ticket_record = session.get(TicketTable, ticket_id)
        if ticket_record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="工单不存在", )

        return Ticket.model_validate(ticket_record, from_attributes=True, )


@app.post("/agent/intent", response_model=IntentResult, tags=["Agent"], summary="识别客户消息意图")
def classify_customer_intent(request: IntentRequest, http_request: Request) -> IntentResult:
    try:
        return classify_intent(request.message, classifier=http_request.app.state.classifier)
    except LLMServiceError as exception:
        # 转换成HTTPException
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="智能客服服务暂时不可用", ) from exception


@app.post("/agent/chat", response_model=AgentResponse, tags=["Agent"], summary="智能客服对话")
def chat_with_agent(request: AgentRequest, http_request: Request) -> AgentResponse:
    try:
        answer, thread_id = run_customer_workflow(
            user_text=request.message,
            workflow_graph=http_request.app.state.workflow_graph,
            thread_id=request.thread_id,
        )
        return AgentResponse(answer=answer, thread_id=thread_id)
    except AgentServiceError as exception:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="智能客服服务暂时不可用", ) from exception
    except TicketRecoveryRequiredError as exception:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exception),
        ) from exception


# 恢复入口
@app.post("/agent/tickets/{thread_id}/resume", response_model=AgentResponse, tags=["Agent"],
          summary="恢复未完成的售后流程")
def resume_agent(thread_id: str, http_request: Request) -> AgentResponse:
    ticket_graph = http_request.app.state.ticket_graph
    try:
        result = resume_ticket_collection(graph=ticket_graph, thread_id=thread_id)
    except TicketSessionNotFoundError as exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exception)) from exception
    except TicketResumeNotRequiredError as exception:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exception)) from exception

    answer = result.get("answer")
    if answer is None or not answer.strip():
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="售后流程恢复后未生成有效回答")
    return AgentResponse(answer=answer, thread_id=thread_id)


@app.post("/agent/demo/stream", tags=["Agent"], summary="客服文本流演示")
def demo_stream(request: AgentRequest) -> StreamingResponse:
    trace_id = current_trace_id.get()
    return StreamingResponse(
        stream_customer_events(request.message, trace_id=trace_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Trace-ID": trace_id},
    )
