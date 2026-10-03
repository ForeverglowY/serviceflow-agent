from openai import APITimeoutError, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionSystemMessageParam

from serviceflow.llm.service import CLIENT, TOOLS
from serviceflow.settings import settings

AGENT_SYSTEM_MESSAGE: ChatCompletionSystemMessageParam = {
    "role": "system",
    "content": (
        "你是一名电商智能客服。"
        "当回答需要真实订单数据时，必须调用提供的工具，"
        "不允许编造订单信息。"
        "必须严格依据工具返回的数据回答。"
        "工具未返回的信息，应明确说明暂时无法确认。"
        "tracking_number为null仅表示暂时没有物流单号，"
        "不能据此判断订单是否已经发货。"
        "不得将“待处理”改写成“待发货”。"
    ),
}

MAX_TOOL_ROUNDS = 3


class AgentServiceError(Exception):
    """Agent执行失败。"""


def call_deepseek(messages: list[ChatCompletionMessageParam], ) -> ChatCompletion:
    try:
        return CLIENT.chat.completions.create(model=settings.deepseek_model, messages=messages, tools=TOOLS,
                                              tool_choice="auto", temperature=0, )
    except APITimeoutError as exception:
        raise AgentServiceError(
            "DeepSeek请求超时"
        ) from exception
    except OpenAIError as exception:
        raise AgentServiceError(
            "DeepSeek服务调用失败"
        ) from exception
