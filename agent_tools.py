# 这个模块专门保存Agent可以调用的真实工具。
import json
from collections.abc import Callable

from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine
from db_models import LogisticsTable, OrderTable
from llm_models import GetLogisticsArguments, GetLogisticsResult, GetOrderArguments, GetOrderResult
from models import Logistics, LogisticsStatus, Order, OrderStatus

# 定义状态映射
ORDER_STATUS_LABELS: dict[OrderStatus, str] = {
    OrderStatus.PENDING: "待处理",
    OrderStatus.PAID: "已支付",
    OrderStatus.SHIPPED: "已发货",
    OrderStatus.DELIVERED: "已送达",
    OrderStatus.CANCELED: "已取消",
}

LOGISTICS_STATUS_LABELS: dict[LogisticsStatus, str] = {
    LogisticsStatus.PENDING: "暂无物流进展",
    LogisticsStatus.IN_TRANSIT: "运输中",
    LogisticsStatus.DELIVERED: "已送达",
    LogisticsStatus.EXCEPTION: "物流异常",
}


def get_order(arguments: GetOrderArguments) -> GetOrderResult | None:
    with Session(engine) as session:
        order_record = session.get(OrderTable, arguments.order_id)
        if order_record is None:
            return None
        order = Order.model_validate(
            order_record,
            from_attributes=True,
        )

    return GetOrderResult(
        order_id=order.order_id,
        product_name=order.product_name,
        unit_price=order.price,
        quantity=order.quantity,
        total_price=order.total_price(),
        payment_status="已支付" if order.is_paid else "未支付",
        order_status=ORDER_STATUS_LABELS[order.status],
        tracking_number=order.tracking_number,
    )


def get_logistics(arguments: GetLogisticsArguments) -> GetLogisticsResult | None:
    with Session(engine) as session:
        logistics_record = session.get(LogisticsTable, arguments.order_id)
        if logistics_record is None:
            return None

        logistics = Logistics.model_validate(
            logistics_record,
            from_attributes=True,
        )

        return GetLogisticsResult(
            order_id=logistics.order_id,
            tracking_number=logistics.tracking_number,
            carrier=logistics.carrier,
            logistics_status=LOGISTICS_STATUS_LABELS[logistics.status],
            latest_event=logistics.latest_event,
        )


class ToolNotFoundError(Exception):
    """模型请求了未注册的工具。"""


def _serialize_tool_result(result: BaseModel | None, ) -> str:
    if result is None:
        return json.dumps(
            {"error": "未查询到相关数据"},
            ensure_ascii=False,
        )

    return result.model_dump_json()


def _execute_get_order(arguments_json: str) -> str:
    arguments = GetOrderArguments.model_validate_json(
        arguments_json
    )
    result = get_order(arguments)
    return _serialize_tool_result(result)


def _execute_get_logistics(arguments_json: str) -> str:
    arguments = GetLogisticsArguments.model_validate_json(
        arguments_json
    )
    result = get_logistics(arguments)
    return _serialize_tool_result(result)


# ToolHandler代表一种函数：接收一个字符串，返回一个字符串。
ToolHandler = Callable[[str], str]

# 注册表
TOOL_HANDLERS: dict[str, ToolHandler] = {
    "get_order": _execute_get_order,
    "get_logistics": _execute_get_logistics,
}


def execute_tool(tool_name: str, arguments_json: str) -> str:
    handler = TOOL_HANDLERS.get(tool_name)

    if handler is None:
        raise ToolNotFoundError(
            f"未注册的工具：{tool_name}"
        )

    return handler(arguments_json)
