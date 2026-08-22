from datetime import date

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, Float, ForeignKey, Integer, String, Text
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


class PolicyChunkTable(Base):
    __tablename__ = "policy_chunks"

    # | 字段 | Python类型 | 数据库类型 | 作用 |
    # | --- | --- | --- | --- |
    # | `chunk_id` | `str` | `String(80)` | 主键，片段唯一编号 |
    # | `policy_id` | `str` | `String(50)` | 所属政策，建立索引 |
    # | `title` | `str` | `String(200)` | 政策标题 |
    # | `section` | `str` | `String(200)` | 章节标题 |
    # | `content` | `str` | `Text` | 政策正文 |
    # | `category` | `str` | `String(50)` | 政策分类，建立索引 |
    # | `version` | `str` | `String(20)` | 政策版本 |
    # | `effective_date` | `date` | `Date` | 生效日期 |
    # | `status` | `str` | `String(20)` | 是否有效，建立索引 |
    # | `applicable_products` | `str` | `String(100)` | 适用商品，建立索引 |
    # | `source` | `str` | `String(500)` | 来源文件 |
    # | `content_hash` | `str` | `String(64)` | 检测正文是否变化 |
    # | `embedding_model` | `str` | `String(100)` | 生成向量的模型 |
    # | `embedding` | `list[float]` | `Vector(512)` | 512 维向量 |
    chunk_id: Mapped[str] = mapped_column(
        String(80),
        primary_key=True,
    )

    policy_id: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    section: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    effective_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    applicable_products: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    source: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(512),
        nullable=False,
    )
