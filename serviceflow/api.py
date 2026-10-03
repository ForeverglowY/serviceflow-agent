from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from langgraph.checkpoint.postgres import PostgresSaver
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from serviceflow.agent.graph import build_customer_agent_graph
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.agent.ticket_collection import build_ticket_collection_graph
from serviceflow.agent.workflow import build_customer_workflow, run_customer_workflow
from serviceflow.db.database import engine
from serviceflow.db.models import LogisticsTable, OrderTable, ProductTable, TicketTable
from serviceflow.llm.models import AgentRequest, AgentResponse, IntentRequest, IntentResult
from serviceflow.llm.service import LLMServiceError, classify_intent
from serviceflow.models import Logistics, Order, Product, Ticket, TicketCreate
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

        app.state.agent_graph = build_customer_agent_graph(checkpointer=checkpointer)
        app.state.ticket_graph = build_ticket_collection_graph(checkpointer=checkpointer)
        app.state.embedding_model = SentenceTransformer(MODEL_NAME)
        app.state.workflow_graph = build_customer_workflow(
            agent_graph=app.state.agent_graph,
            embedding_model=app.state.embedding_model,
            ticket_graph=app.state.ticket_graph,
        )

        print("服务启动：PostgreSQL 客服工作流已创建")

        yield

    print("服务关闭：Checkpointer 连接已关闭")


app = FastAPI(title="智能客服与工单处理 Agent", lifespan=lifespan)


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
def classify_customer_intent(request: IntentRequest) -> IntentResult:
    try:
        return classify_intent(request.message)
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
