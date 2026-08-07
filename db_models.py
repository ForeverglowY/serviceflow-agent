from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from models import LogisticsStatus, OrderStatus, TicketPriority, TicketStatus


class Base(DeclarativeBase):
    pass


class OrderTable(Base):
    __tablename__ = "orders"

    # | Python
    # 属性 | 类型 | 数据库规则 |
    # | --- | --- | --- |
    # | `order_id` | `str` | `String(50)`，主键 |
    # | `product_name` | `str` | `String(100)`，不能为空 |
    # | `price` | `float` | `Float`，不能为空 |
    # | `quantity` | `int` | `Integer`，不能为空 |
    # | `is_paid` | `bool` | `Boolean`，不能为空 |
    # | `status` | `str` | `String(30)`，不能为空，默认
    # `"pending"` |
    # | `tracking_number` | `str \ | None
    # ` | `String(100)`，允许为空 |
    # | `customer_note` | `str \ | None
    # ` | `String(500)`，允许为空 |
    order_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    product_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    price: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    is_paid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=OrderStatus.PENDING.value,
        server_default=OrderStatus.PENDING.value
    )

    tracking_number: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    customer_note: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True
    )


class ProductTable(Base):
    __tablename__ = "products"
    # | 属性 | Python 类型 | 数据库配置 |
    # |---|---|---|
    # | `product_id` | `str` | `String(50)`，主键 |
    # | `name` | `str` | `String(100)`，不能为空 |
    # | `price` | `float` | `Float`，不能为空 |
    # | `stock` | `int` | `Integer`，不能为空 |
    # | `is_returnable` | `bool` | `Boolean`，不能为空，默认 `True` |
    product_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    price: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    stock: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    is_returnable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true"
    )


class LogisticsTable(Base):
    __tablename__ = "logistics"

    # | 属性 | Python
    # 类型 | 数据库配置 |
    # | --- | --- | --- |
    # | `order_id` | `str` | `String(50)`，主键，同时外键关联
    # `orders.order_id` |
    # | `tracking_number` | `str` | `String(100)`，不能为空，唯一 |
    # | `carrier` | `str` | `String(100)`，不能为空 |
    # | `status` | `str` | `String(30)`，默认
    # `"pending"` |
    # | `latest_event` | `str \ | None
    # ` | `String(500)`，允许为空 |
    order_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("orders.order_id"),
        primary_key=True,
    )

    tracking_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    carrier: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=LogisticsStatus.PENDING.value,
        server_default=LogisticsStatus.PENDING.value
    )

    latest_event: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )


class TicketTable(Base):
    __tablename__ = "tickets"

    # | 属性 | Python
    # 类型 | 数据库配置 |
    # | --- | --- | --- |
    # | `ticket_id` | `str` | `String(50)`，主键 |
    # | `order_id` | `str` | `String(50)`，外键关联
    # `orders.order_id`，不能为空并建立索引 |
    # | `user_id` | `str` | `String(50)`，不能为空并建立索引 |
    # | `issue_type` | `str` | `String(100)`，不能为空 |
    # | `description` | `str` | `String(1000)`，不能为空 |
    # | `priority` | `str` | `String(30)`，默认
    # `medium` |
    # | `status` | `str` | `String(30)`，默认
    # `open` |
    # | `assigned_to` | `str \ | None
    # ` | `String(100)`，允许为空 |
    ticket_id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
    )

    order_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("orders.order_id"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    issue_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    priority: Mapped[str] = mapped_column(
        String(30),
        default=TicketPriority.MEDIUM.value,
        server_default=TicketPriority.MEDIUM.value,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default=TicketStatus.OPEN.value,
        server_default=TicketStatus.OPEN.value,
    )

    assigned_to: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
