import logging
from time import perf_counter

from openai import OpenAI, OpenAIError
from openai.types.chat import (ChatCompletionFunctionToolParam, ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, )
from openai.types.shared_params import ResponseFormatJSONObject
from pydantic import ValidationError

from llm_models import GetLogisticsArguments, GetOrderArguments, IntentResult
from settings import settings

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

SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        "你是电商客服意图分类器。"
        "请判断用户的意图，只返回JSON，不要返回Markdown或其他文字。"
        "JSON必须包含以下字段："
        "intent：只能是order_query、logistics_query、product_query、"
        "product_fault、return_refund、create_ticket、other之一；"
        "order_id：从用户消息中提取订单号，没有则为null；"
        "confidence：0到1之间的数字；"
        "reason：简短说明分类理由。"
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

    messages: list[ChatCompletionMessageParam] = [SYSTEM_MESSAGE, user_message, ]

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
