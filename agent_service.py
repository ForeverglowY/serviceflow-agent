import logging
from typing import cast

from openai import APITimeoutError, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionAssistantMessageParam, \
    ChatCompletionMessageFunctionToolCall, ChatCompletionMessageParam, \
    ChatCompletionSystemMessageParam, ChatCompletionToolMessageParam, ChatCompletionUserMessageParam
from pydantic import ValidationError
from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLAlchemyTimeoutError,
)

from agent_tools import ToolNotFoundError, execute_tool
from llm_service import CLIENT, TOOLS
from settings import settings

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

# 表示最多允许执行3轮工具，防止模型无限循环。
MAX_TOOL_ROUNDS = 3

logger = logging.getLogger(__name__)


class AgentServiceError(Exception):
    """Agent执行失败。"""


def _call_deepseek(messages: list[ChatCompletionMessageParam], ) -> ChatCompletion:
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


def run_customer_agent(user_text: str) -> str:
    # 1. user消息
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_text, }

    messages: list[ChatCompletionMessageParam] = [AGENT_SYSTEM_MESSAGE, user_message, ]

    for round_index in range(MAX_TOOL_ROUNDS + 1):
        # +1 如果最大允许执行3轮工具，仍然需要再给模型一次机会生成最终回答：
        response = _call_deepseek(messages)
        message = response.choices[0].message

        # 没有工具调用，说明模型准备返回最终回答
        if not message.tool_calls:
            if message.content is None:
                raise AgentServiceError("DeepSeek没有返回内容")

            return message.content

        # 已经执行了允许的最大工具轮数
        if round_index >= MAX_TOOL_ROUNDS:
            raise AgentServiceError(
                "超过最大工具调用次数"
            )

        # 一条assistant消息可能包含多个工具请求
        assistant_message = cast(
            ChatCompletionAssistantMessageParam,
            message.model_dump(exclude_none=True),
        )
        messages.append(assistant_message)

        for tool_call in message.tool_calls:
            if not isinstance(
                    tool_call,
                    ChatCompletionMessageFunctionToolCall,
            ):
                raise AgentServiceError(
                    f"不支持的工具调用类型：{tool_call.type}"
                )

            logger.info("Agent调用工具 name=%s call_id=%s", tool_call.function.name, tool_call.id, )

            # 5. 执行模型选择的工具
            try:
                tool_content = execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
            except ToolNotFoundError as exception:
                raise AgentServiceError(
                    str(exception)
                ) from exception
            except ValidationError as exception:
                raise AgentServiceError(
                    "工具调用参数格式错误"
                ) from exception
            except (
                    OperationalError,
                    SQLAlchemyTimeoutError,
            ) as exception:
                raise AgentServiceError(
                    "工具执行超时或数据库不可用"
                ) from exception
            except SQLAlchemyError as exception:
                raise AgentServiceError(
                    "工具执行失败"
                ) from exception

            tool_message: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content,
            }
            messages.append(tool_message)
    raise AgentServiceError("Agent未生成最终回答")
