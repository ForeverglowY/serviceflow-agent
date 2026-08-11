from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent_service import AgentServiceError, run_customer_agent
from database import engine
from db_models import LogisticsTable, OrderTable, ProductTable, TicketTable
from llm_models import AgentRequest, AgentResponse, IntentRequest, IntentResult
from llm_service import LLMServiceError, classify_intent
from models import Logistics, Order, Product, Ticket, TicketCreate

app = FastAPI(title="智能客服与工单处理 Agent")


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
        with Session(engine) as session:
            with session.begin():
                # 检查 order_id 对应的订单是否存在
                if session.get(OrderTable, ticket_create.order_id) is None:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="订单不存在")

                ticket_id = f"TICKET-{uuid4().hex}"
                ticket = Ticket(ticket_id=ticket_id, **ticket_create.model_dump(), )
                ticket_record = TicketTable(**ticket.model_dump(mode="json"), )
                session.add(ticket_record)
                session.flush()

                return Ticket.model_validate(ticket_record, from_attributes=True, )

    except IntegrityError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="工单创建冲突", ) from error


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
def chat_with_agent(request: AgentRequest, ) -> AgentResponse:
    try:
        answer = run_customer_agent(request.message)
        return AgentResponse(answer=answer)
    except AgentServiceError as exception:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="智能客服服务暂时不可用", ) from exception
