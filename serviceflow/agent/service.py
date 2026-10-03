import logging
from typing import cast

from openai.types.chat import ChatCompletionAssistantMessageParam, \
    ChatCompletionMessageFunctionToolCall, ChatCompletionMessageParam, \
    ChatCompletionToolMessageParam, ChatCompletionUserMessageParam
from pydantic import ValidationError
from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLAlchemyTimeoutError,
)

from serviceflow.agent.runtime import AGENT_SYSTEM_MESSAGE, AgentServiceError, MAX_TOOL_ROUNDS, call_deepseek
from serviceflow.agent.tools import ToolNotFoundError, execute_tool

# 表示最多允许执行3轮工具，防止模型无限循环。

logger = logging.getLogger("uvicorn.error.agent_service")
logger.setLevel(logging.INFO)


def run_customer_agent(user_text: str) -> str:
    logger.info("Agent请求开始")
    # 1. user消息
    user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_text, }

    messages: list[ChatCompletionMessageParam] = [AGENT_SYSTEM_MESSAGE, user_message, ]

    for round_index in range(MAX_TOOL_ROUNDS + 1):
        logger.info(
            "Agent模型调用 round=%d",
            round_index + 1,
        )
        # +1 如果最大允许执行3轮工具，仍然需要再给模型一次机会生成最终回答：
        response = call_deepseek(messages)
        message = response.choices[0].message

        # 没有工具调用，说明模型准备返回最终回答
        if not message.tool_calls:
            if message.content is None:
                raise AgentServiceError("DeepSeek没有返回内容")

            logger.info(
                "Agent请求完成 model_rounds=%d",
                round_index + 1,
            )
            return message.content

        # 已经执行了允许的最大工具轮数
        if round_index >= MAX_TOOL_ROUNDS:
            logger.warning(
                "Agent超过最大工具调用次数 max_rounds=%d",
                MAX_TOOL_ROUNDS,
            )
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
