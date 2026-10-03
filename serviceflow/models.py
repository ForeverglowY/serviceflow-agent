from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELED = "canceled"


# order_id：长度至少1，最多50
# product_name：长度至少1，最多100
# price：必须大于0
# quantity：必须大于等于1
# is_paid：暂时不增加约束
class Order(BaseModel):
    order_id: str = Field(min_length=1, max_length=50)
    product_name: str = Field(min_length=1, max_length=100)
    product_id: str = Field(min_length=1, max_length=50)
    price: float = Field(gt=0)
    quantity: int = Field(ge=1)
    is_paid: bool
    status: OrderStatus = OrderStatus.PENDING
    tracking_number: str | None = Field(default=None, min_length=1, max_length=100)
    customer_note: str | None = Field(default=None, max_length=500)

    def total_price(self) -> float:
        return self.price * self.quantity

    def payment_status(self) -> str:
        return "是" if self.is_paid else "否"


# User 字段要求
# 字段	类型	约束/默认值
# user_id	str	长度1～50
# name	str	长度1～100
# phone	str | None	默认 None，长度6～30
# is_vip	bool	默认 False
class User(BaseModel):
    user_id: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=100)
    phone: str | None = Field(default=None, min_length=6, max_length=30)
    is_vip: bool = Field(default=False)


# Product 字段要求
# 字段	类型	约束/默认值
# product_id	str	长度1～50
# name	str	长度1～100
# price	float	大于0
# stock	int	大于等于0
# is_returnable	bool	默认 True
class Product(BaseModel):
    product_id: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=100)
    category: str | None = Field(default=None, min_length=1, max_length=50)
    price: float = Field(gt=0)
    stock: int = Field(ge=0)
    is_returnable: bool = Field(default=True)
    requires_special_review: bool = False


# 物流状态
class LogisticsStatus(str, Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    EXCEPTION = "exception"


# 物流
class Logistics(BaseModel):
    # | 字段 | 类型 | 约束 / 默认值 |
    # | --- | --- | --- |
    # | `order_id` | `str` | 长度1～50 |
    # | `tracking_number` | `str` | 长度1～100 |
    # | `carrier` | `str` | 长度1～100 |
    # | `status` | `LogisticsStatus` | 默认
    # `PENDING` |
    # | `latest_event` | `str \ | None
    # ` | 默认
    # `None`，最长500 |
    order_id: str = Field(min_length=1, max_length=50)
    tracking_number: str = Field(min_length=1, max_length=100)
    carrier: str = Field(min_length=1, max_length=100)
    status: LogisticsStatus = Field(default=LogisticsStatus.PENDING)
    latest_event: str | None = Field(default=None, max_length=500)
    signed_at: date | None = None


# 工单模型, 优先级枚举
class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# 工单状态枚举
class TicketStatus(str, Enum):
    OPEN = "open"
    PROCESSING = "processing"
    RESOLVED = "resolved"
    CLOSED = "closed"


# 工单
class Ticket(BaseModel):
    # | 字段 | 类型 | 约束 / 默认值 |
    # | --- | --- | --- |
    # | `ticket_id` | `str` | 长度1～50 |
    # | `order_id` | `str` | 长度1～50 |
    # | `user_id` | `str` | 长度1～50 |
    # | `issue_type` | `str` | 长度1～100 |
    # | `description` | `str` | 长度1～1000 |
    # | `priority` | `TicketPriority` | 默认
    # `MEDIUM` |
    # | `status` | `TicketStatus` | 默认
    # `OPEN` |
    # | `assigned_to` | `str \ | None
    # ` | 默认
    # `None`，长度1～100 |
    ticket_id: str = Field(min_length=1, max_length=50)
    order_id: str = Field(min_length=1, max_length=50)
    user_id: str = Field(min_length=1, max_length=50)
    issue_type: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM)
    status: TicketStatus = Field(default=TicketStatus.OPEN)
    assigned_to: str | None = Field(default=None, min_length=1, max_length=100)


class TicketCreate(BaseModel):
    order_id: str = Field(min_length=1, max_length=50)
    user_id: str = Field(min_length=1, max_length=50)
    issue_type: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM)


class TicketConfirmationCreate(BaseModel):
    confirmed_by: str  # 确认人
    confirmed_at: datetime  # 确认时间
    confirmation_message: str  # 用户原话
    application_snapshot: dict[str, str]  # 确认的是哪份申请
