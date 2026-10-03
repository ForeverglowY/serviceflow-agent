from serviceflow.llm.models import CustomerIntent


def select_customer_route(
        intent: CustomerIntent | None,
) -> str:
    if intent in (CustomerIntent.ORDER_QUERY, CustomerIntent.LOGISTICS_QUERY):
        # 查询真实业务数据
        return "tools"
    elif intent in (CustomerIntent.PRODUCT_FAULT, CustomerIntent.RETURN_REFUND):
        # 先解释相关售后政策，不直接执行退款
        return "rag"
    elif intent == CustomerIntent.CREATE_TICKET:
        # 进入工单信息收集，暂不写数据库
        return "ticket"
    else:
        # 引导用户补充或说明当前能力范围
        return "fallback"


def main() -> None:
    print(select_customer_route(CustomerIntent.ORDER_QUERY))
    print(select_customer_route(CustomerIntent.RETURN_REFUND))
    print(select_customer_route(CustomerIntent.CREATE_TICKET))
    print(select_customer_route(CustomerIntent.OTHER))


if __name__ == "__main__":
    main()
