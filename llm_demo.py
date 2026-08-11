import logging

from llm_models import GetOrderArguments
from llm_service import GET_ORDER_TOOL, classify_intent


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    print(GetOrderArguments.model_json_schema())

    print(GET_ORDER_TOOL)

    # user_text = "订单20260721002什么时候能送到？"
    # result = classify_intent(user_text)
    #
    # print(f"用户输入：{user_text}")
    # print(f"意图：{result.intent.value}")
    # print(f"订单号：{result.order_id}")
    # print(f"置信度：{result.confidence}")
    # print(f"判断理由：{result.reason}")




if __name__ == "__main__":
    main()
