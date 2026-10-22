from langchain_core.tools import tool

from serviceflow.agent.tools import get_logistics, get_order
from serviceflow.llm.models import GetLogisticsArguments, GetOrderArguments

"""包装业务工具"""


@tool(args_schema=GetOrderArguments)
def get_order_tool(order_id: str) -> dict:
    """根据订单号查询商品、金额、支付状态和订单状态，不提供物流轨迹。"""
    order_arguments = GetOrderArguments(order_id=order_id)
    result = get_order(order_arguments)
    if result is None:
        return {"error": "订单不存在"}
    else:
        return result.model_dump(mode="json")


@tool(args_schema=GetLogisticsArguments)
def get_logistics_tool(order_id: str) -> dict:
    """查询物流单号、承运商、运输状态和最新动态"""
    logistics_arguments = GetLogisticsArguments(order_id=order_id)
    result = get_logistics(logistics_arguments)

    if result is None:
        return {"error": "物流不存在"}
    else:
        return result.model_dump(mode="json")


CUSTOMER_TOOLS = [get_order_tool, get_logistics_tool]
