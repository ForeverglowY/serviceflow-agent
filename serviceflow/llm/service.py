import logging
from time import perf_counter

from openai import OpenAI, OpenAIError
from openai.types.chat import (ChatCompletionFunctionToolParam, ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, )
from openai.types.shared_params import ResponseFormatJSONObject
from pydantic import ValidationError

from serviceflow.llm.models import GetLogisticsArguments, GetOrderArguments, IntentResult, ReturnReasonResult
from serviceflow.settings import settings

GET_ORDER_TOOL = ChatCompletionFunctionToolParam(
    type="function",
    function={
        "name": "get_order",
        "description": (
            "根据订单号查询订单自身信息，"
            "包括商品、金额、支付状态和订单状态。"
            "用户询问订单内容、金额或支付状态时使用。"
            "不用于查询快递轨迹或物流配送进度。"
        ),
        "parameters": GetOrderArguments.model_json_schema(),
    },
)

GET_LOGISTICS_TOOL = ChatCompletionFunctionToolParam(
    type="function",
    function={
        "name": "get_logistics",
        "description": (
            "根据订单号查询物流配送信息，"
            "包括承运商、物流单号、物流状态和最新物流事件。"
            "用户询问快递、配送、物流进度或订单到哪里时使用。"
        ),
        "parameters": GetLogisticsArguments.model_json_schema(),
    },
)

INTENT_SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        "你是电商客服意图分类器。"
        "请判断用户的意图，只返回JSON，不要返回Markdown或其他文字。"
        "JSON必须包含以下字段："
        "intent：只能是order_query、logistics_query、product_query、"
        "product_fault、return_refund、create_ticket、other之一；"
        "分类规则："
        "用户明确要求办理退货、换货、维修或提交售后申请时，"
        "选择 create_ticket，不要求用户必须说出“工单”这个词。"
        "同时描述商品故障和明确处理诉求时，优先选择 create_ticket"
        "用户只咨询退换货条件、政策或可行性时，选择 return_refund。"
        "用户只描述故障、询问如何排查，未明确要求办理售后时，选择 product_fault。"
        "例子："
        "“耳机左耳没有声音，我想换货” → create_ticket"
        "“请帮我申请退货” → create_ticket"
        "“拆封的耳机可以退货吗？” → return_refund"
        "“耳机左耳没声音，怎么排查？” → product_fault"
        "order_id：从用户消息中提取订单号，没有则为null；"
        "confidence：0到1之间的数字；"
        "reason：简短说明分类理由。"
    ),
}

RETURN_REASON_SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        "你是退换货原因分类器。"
        "请判断用户的意图，只返回JSON，不要返回Markdown或其他文字。"
        "JSON必须包含以下字段："
        "return_reason：quality_issue、no_reason、other之一；"
        "比如'左耳无声，想换货'，这个就是 quality_issue；"
        "商品没问题，只是不想要了，就是no_reason；"
        "帮我处理一下售后，就是 other，不能猜测为质量问题或者无理由退货；"
    ),
}

CLIENT = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    timeout=settings.deepseek_timeout_seconds,
    max_retries=settings.deepseek_max_retries,
)

TOOLS: list[ChatCompletionFunctionToolParam] = [
    GET_ORDER_TOOL,
    GET_LOGISTICS_TOOL,
]

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """大模型服务调用或响应处理失败。"""


def classify_intent(user_text: str) -> IntentResult:
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_text, }

    messages: list[ChatCompletionMessageParam] = [INTENT_SYSTEM_MESSAGE, user_message, ]

    response_format = ResponseFormatJSONObject(type="json_object", )

    start_time = perf_counter()
    try:
        response = CLIENT.chat.completions.create(model=settings.deepseek_model, messages=messages,
                                                  response_format=response_format, temperature=0, )
    except OpenAIError as exception:
        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.exception("DeepSeek调用失败 elapsed_ms=%.2f", elapsed_ms, )
        raise LLMServiceError("DeepSeek服务调用失败") from exception

    elapsed_ms = (perf_counter() - start_time) * 1000

    usage = response.usage
    if usage is not None:
        logger.info("DeepSeek调用完成 model=%s elapsed_ms=%.2f "
                    "prompt_tokens=%d completion_tokens=%d total_tokens=%d", response.model, elapsed_ms,
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, )
    else:
        logger.info("DeepSeek调用完成 model=%s elapsed_ms=%.2f usage=unknown", response.model, elapsed_ms, )

    content = response.choices[0].message.content
    if content is None:
        raise LLMServiceError("DeepSeek没有返回内容")
    try:
        return IntentResult.model_validate_json(content)
    except ValidationError as exception:
        logger.exception("DeepSeek结构化输出校验失败")
        raise LLMServiceError("DeepSeek返回内容格式错误") from exception


def classify_return_reason(issue_description: str, requested_action: str) -> ReturnReasonResult:
    user_text: str = (f"用户描述：{issue_description}\n"
                      f"用户请求:{requested_action}")
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_text}

    messages: list[ChatCompletionMessageParam] = [RETURN_REASON_SYSTEM_MESSAGE, user_message, ]

    response_format = ResponseFormatJSONObject(type="json_object", )

    start_time = perf_counter()
    try:
        response = CLIENT.chat.completions.create(model=settings.deepseek_model, messages=messages,
                                                  response_format=response_format, temperature=0, )
    except OpenAIError as exception:
        elapsed_ms = (perf_counter() - start_time) * 1000
        logger.exception("DeepSeek调用失败 elapsed_ms=%.2f", elapsed_ms, )
        raise LLMServiceError("DeepSeek服务调用失败") from exception

    elapsed_ms = (perf_counter() - start_time) * 1000

    usage = response.usage
    if usage is not None:
        logger.info("DeepSeek调用完成 model=%s elapsed_ms=%.2f "
                    "prompt_tokens=%d completion_tokens=%d total_tokens=%d", response.model, elapsed_ms,
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, )
    else:
        logger.info("DeepSeek调用完成 model=%s elapsed_ms=%.2f usage=unknown", response.model, elapsed_ms, )

    content = response.choices[0].message.content
    if content is None:
        raise LLMServiceError("DeepSeek没有返回内容")
    try:
        return ReturnReasonResult.model_validate_json(content)
    except ValidationError as exception:
        logger.exception("DeepSeek结构化输出校验失败")
        raise LLMServiceError("DeepSeek返回内容格式错误") from exception


def main() -> None:
    print(classify_return_reason("左耳没有声音", "换货"))
    print(classify_return_reason("商品没有问题，只是不想要了", "退货"))
    print(classify_return_reason("帮我处理一下售后", "不确定"))


if __name__ == "__main__":
    main()
