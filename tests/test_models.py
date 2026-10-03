from datetime import date

import pytest
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from serviceflow.models import Logistics, LogisticsStatus, Order, OrderStatus, Product, Ticket, TicketPriority, TicketStatus, User

VALID_ORDER_DATA = {"order_id": "TEST-001", "product_name": "测试键盘", "product_id": "PRODUCT-TEST-001", "price": 599.0, "quantity": 2, "is_paid": True,
                    "status": "paid", "tracking_number": "SF100000001", "customer_note": "测试订单", }

ORDER_WITHOUT_OPTIONAL_DATA = {"order_id": "TEST-002", "product_name": "测试鼠标", "product_id": "PRODUCT-TEST-002", "price": 199.0, "quantity": 1,
                               "is_paid": False, }

INVALID_PRICE_DATA = {"order_id": "TEST-003", "product_name": "错误价格商品", "product_id": "PRODUCT-TEST-003", "price": 0, "quantity": 1,
                      "is_paid": True, "status": "paid", }

INVALID_QUANTITY_DATA = {"order_id": "TEST-004", "product_name": "错误数量商品", "product_id": "PRODUCT-TEST-004", "price": 99.0, "quantity": 0,
                         "is_paid": True, "status": "paid", }

INVALID_STATUS_DATA = {"order_id": "TEST-005", "product_name": "错误状态商品", "product_id": "PRODUCT-TEST-005", "price": 299.0, "quantity": 1,
                       "is_paid": True, "status": "unknown", }


def test_valid_order() -> None:
    # 要求断言：
    # order_id正确；
    # price正确；
    # status is OrderStatus.PAID；
    # 总价是
    # 1198.0；
    # 支付状态是“是”。
    order: Order = Order.model_validate(VALID_ORDER_DATA)

    assert order.order_id == "TEST-001"
    assert order.product_name == "测试键盘"
    assert order.price == 599.0
    assert order.quantity == 2
    assert order.status is OrderStatus.PAID
    assert order.tracking_number == "SF100000001"
    assert order.total_price() == 1198.0
    assert order.payment_status() == "是"


# 2. 可选字段默认值
def test_optional_fields_default_to_none() -> None:
    # 使用 ORDER_WITHOUT_OPTIONAL_DATA，断言：
    # tracking_number is None；
    # customer_note is None；
    # 默认状态是 OrderStatus.PENDING。
    order: Order = Order.model_validate(ORDER_WITHOUT_OPTIONAL_DATA)

    assert order.tracking_number is None
    assert order.customer_note is None
    assert order.status is OrderStatus.PENDING


def test_price_must_be_positive() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Order.model_validate(INVALID_PRICE_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()

    assert errors[0]["loc"] == ("price",)
    assert errors[0]["type"] == "greater_than"


def test_quantity_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Order.model_validate(INVALID_QUANTITY_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()

    assert errors[0]["loc"] == ("quantity",)
    assert errors[0]["type"] == "greater_than_equal"


def test_status_must_be_known() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Order.model_validate(INVALID_STATUS_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()

    assert errors[0]["loc"] == ("status",)
    assert errors[0]["type"] == "enum"


VALID_USER_DATA = {"user_id": "USER-001", "name": "张三", "phone": "13800138000", "is_vip": True, }

USER_WITH_DEFAULTS_DATA = {"user_id": "USER-002", "name": "李四", }

INVALID_USER_DATA = {"user_id": "", "name": "错误用户", "phone": "123", }

VALID_PRODUCT_DATA = {"product_id": "PRODUCT-001", "name": "蓝牙耳机", "category": "in_ear_earphone", "price": 399.0, "stock": 100,
                      "is_returnable": True, }

PRODUCT_WITH_DEFAULTS_DATA = {"product_id": "PRODUCT-002", "name": "机械键盘", "price": 599.0, "stock": 20, }

INVALID_PRODUCT_PRICE_DATA = {"product_id": "PRODUCT-003", "name": "错误价格商品", "price": -1, "stock": 10, }

INVALID_PRODUCT_STOCK_DATA = {"product_id": "PRODUCT-004", "name": "错误库存商品", "price": 99.0, "stock": -1, }


def test_valid_user() -> None:
    user: User = User.model_validate(VALID_USER_DATA)
    assert user.user_id == "USER-001"
    assert user.name == "张三"
    assert user.phone == "13800138000"
    assert user.is_vip == True


def test_user_defaults() -> None:
    user: User = User.model_validate(USER_WITH_DEFAULTS_DATA)
    assert user.user_id == "USER-002"
    assert user.name == "李四"
    assert user.phone is None
    assert user.is_vip is False  # "user_id": "USER-002",  # "name": "李四",


def test_invalid_user_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        User.model_validate(INVALID_USER_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()
    assert len(errors) == 2

    assert errors[0]["loc"] == ("user_id",)
    assert errors[0]["type"] == "string_too_short"

    assert errors[1]["loc"] == ("phone",)
    assert errors[1]["type"] == "string_too_short"  # "user_id": "",  # "name": "错误用户",  # "phone": "123",


def test_valid_product() -> None:
    product: Product = Product.model_validate(VALID_PRODUCT_DATA)

    assert product.product_id == "PRODUCT-001"
    assert product.name == "蓝牙耳机"
    assert product.category == "in_ear_earphone"
    assert product.price == 399.0
    assert product.stock == 100
    assert product.is_returnable is True

    # "product_id": "PRODUCT-001",  # "name": "蓝牙耳机",  # "price": 399.0,  # "stock": 100,  # "is_returnable": True,


def test_product_category_defaults_to_unknown() -> None:
    product = Product.model_validate(PRODUCT_WITH_DEFAULTS_DATA)

    assert product.category is None


def test_product_price_must_be_positive() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Product.model_validate(INVALID_PRODUCT_PRICE_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()
    assert errors[0]["loc"] == ("price",)
    assert errors[0]["type"] == "greater_than"


def test_product_stock_cannot_be_negative() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Product.model_validate(INVALID_PRODUCT_STOCK_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()
    assert errors[0]["loc"] == ("stock",)
    assert errors[0]["type"] == "greater_than_equal"


VALID_LOGISTICS_DATA = {"order_id": "TEST-001", "tracking_number": "SF100000001", "carrier": "顺丰速运",
                        "status": "in_transit", "latest_event": "快件已到达上海转运中心", }

LOGISTICS_WITH_DEFAULTS_DATA = {"order_id": "TEST-002", "tracking_number": "YT100000002", "carrier": "圆通速递", }

INVALID_LOGISTICS_STATUS_DATA = {"order_id": "TEST-003", "tracking_number": "UNKNOWN-003", "carrier": "未知快递",
                                 "status": "lost", }


def test_valid_logistics() -> None:
    logistics: Logistics = Logistics.model_validate(VALID_LOGISTICS_DATA)
    assert logistics.order_id == "TEST-001"
    assert logistics.tracking_number == "SF100000001"
    assert logistics.carrier == "顺丰速运"
    assert logistics.status == LogisticsStatus.IN_TRANSIT
    assert logistics.latest_event == "快件已到达上海转运中心"


def test_logistics_defaults() -> None:
    logistics: Logistics = Logistics.model_validate(LOGISTICS_WITH_DEFAULTS_DATA)
    assert logistics.order_id == "TEST-002"
    assert logistics.tracking_number == "YT100000002"
    assert logistics.carrier == "圆通速递"
    assert logistics.status == LogisticsStatus.PENDING
    assert logistics.latest_event is None
    assert logistics.signed_at is None

    # "order_id": "TEST-002",  # "tracking_number": "YT100000002",  # "carrier": "圆通速递",


def test_logistics_parses_verified_signing_date() -> None:
    logistics = Logistics.model_validate({
        **VALID_LOGISTICS_DATA,
        "status": "delivered",
        "signed_at": "2026-07-25",
    })

    assert logistics.signed_at == date(2026, 7, 25)


def test_logistics_status_must_be_known() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Logistics.model_validate(INVALID_LOGISTICS_STATUS_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("status",)
    assert errors[0]["type"] == "enum"


VALID_TICKET_DATA = {"ticket_id": "TICKET-001", "order_id": "TEST-001", "user_id": "USER-001",
    "issue_type": "product_fault", "description": "蓝牙耳机左耳没有声音", "priority": "high", "status": "processing",
    "assigned_to": "客服小王", }

TICKET_WITH_DEFAULTS_DATA = {"ticket_id": "TICKET-002", "order_id": "TEST-002", "user_id": "USER-002",
    "issue_type": "logistics_delay", "description": "物流三天没有更新", }

INVALID_TICKET_PRIORITY_DATA = {"ticket_id": "TICKET-003", "order_id": "TEST-003", "user_id": "USER-003",
    "issue_type": "refund", "description": "申请退款", "priority": "critical", }

INVALID_TICKET_DESCRIPTION_DATA = {"ticket_id": "TICKET-004", "order_id": "TEST-004", "user_id": "USER-004",
    "issue_type": "other", "description": "", }


def test_valid_ticket() -> None:
    ticket: Ticket = Ticket.model_validate(VALID_TICKET_DATA)

    assert ticket.ticket_id == "TICKET-001"
    assert ticket.order_id == "TEST-001"
    assert ticket.user_id == "USER-001"
    assert ticket.issue_type == "product_fault"
    assert ticket.description == "蓝牙耳机左耳没有声音"
    assert ticket.priority is TicketPriority.HIGH
    assert ticket.status is TicketStatus.PROCESSING
    assert ticket.assigned_to == "客服小王"


def test_ticket_defaults() -> None:
    ticket: Ticket = Ticket.model_validate(TICKET_WITH_DEFAULTS_DATA)

    assert ticket.priority is TicketPriority.MEDIUM
    assert ticket.status is TicketStatus.OPEN
    assert ticket.assigned_to is None


def test_ticket_priority_must_be_known() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Ticket.model_validate(INVALID_TICKET_PRIORITY_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["loc"] == ("priority",)
    assert errors[0]["type"] == "enum"


def test_ticket_description_cannot_be_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Ticket.model_validate(INVALID_TICKET_DESCRIPTION_DATA)

    errors: list[ErrorDetails] = exc_info.value.errors()

    assert len(errors) == 1
    assert errors[0]["loc"] == ("description",)
    assert errors[0]["type"] == "string_too_short"
