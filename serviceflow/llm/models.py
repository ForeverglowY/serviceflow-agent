from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from serviceflow.ticket.rules import ReturnReason


class CustomerIntent(str, Enum):
    ORDER_QUERY = "order_query"
    LOGISTICS_QUERY = "logistics_query"
    PRODUCT_QUERY = "product_query"
    PRODUCT_FAULT = "product_fault"
    RETURN_REFUND = "return_refund"
    CREATE_TICKET = "create_ticket"
    OTHER = "other"


class IntentResult(BaseModel):
    intent: CustomerIntent
    order_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=200)


class IntentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class GetOrderArguments(BaseModel):
    order_id: str = Field(min_length=1, max_length=50, description="需要查询的订单号")

    model_config = ConfigDict(extra="forbid")


class GetOrderResult(BaseModel):
    order_id: str
    product_name: str
    unit_price: float
    quantity: int
    total_price: float
    payment_status: str
    order_status: str
    tracking_number: str | None = None


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="会话编号；连续对话时使用同一个编号",
    )


class AgentResponse(BaseModel):
    answer: str
    thread_id: str


class GetLogisticsArguments(BaseModel):
    order_id: str = Field(min_length=1, max_length=50, description="需要查询物流信息的订单号", )
    model_config = ConfigDict(extra="forbid")


class GetLogisticsResult(BaseModel):
    order_id: str
    tracking_number: str
    carrier: str
    logistics_status: str
    latest_event: str | None = None


class ReturnReasonResult(BaseModel):
    return_reason: ReturnReason

    model_config = ConfigDict(extra="forbid")
