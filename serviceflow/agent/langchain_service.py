from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import APITimeoutError, OpenAIError
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError, SQLAlchemyError, TimeoutError as SQLAlchemyTimeoutError

from serviceflow.agent.langchain_tools import CUSTOMER_TOOLS
from serviceflow.agent.runtime import AgentServiceError
from serviceflow.llm.chat_model import create_chat_model

"""执行模型-工具循环"""


def run_langchain_agent(user_text: str) -> str:
    model = create_chat_model()

    # 工具注册表
    tool_registry = {
        current_tool.name: current_tool
        for current_tool in CUSTOMER_TOOLS
    }
    # 绑定工具
    model_with_tools = model.bind_tools(list(tool_registry.values()))

    system_message = SystemMessage(
        content="你是一名电商客服。查询真实订单信息时必须使用提供的工具，不得编造数据。物流进度必须依据物流工具返回结果回答；订单工具不提供物流轨迹。"
                "tracking_number 为 null 仅表示当前订单记录未提供物流单号，不能据此推断未发货或暂无物流信息。用户只询问商品和支付状态时，不主动推断物流情况。"
                "当前仅支持按订单号查询订单和物流。不支持通过手机号、姓名或其他信息查单，不得要求用户提供这些信息，也不得承诺具备未提供的查询能力。",
    )

    human_message = HumanMessage(
        content=user_text,
    )

    messages: list[BaseMessage] = [system_message, human_message]

    max_tool_rounds = 3

    for round_index in range(max_tool_rounds + 1):
        try:
            response = model_with_tools.invoke(messages)
        except APITimeoutError as exception:
            raise AgentServiceError("模型请求超时") from exception
        except OpenAIError as exception:
            raise AgentServiceError("模型服务调用失败") from exception

        if not response.tool_calls:
            content = response.content
            if not isinstance(content, str):
                raise AgentServiceError("目前只支持字符串")
            if not content.strip():
                raise AgentServiceError("回答为空")
            return response.content

        if round_index >= max_tool_rounds:
            raise AgentServiceError("超过最大工具调用轮数")

        messages.append(response)

        for tool_call in response.tool_calls:
            tool_call_func = tool_registry.get(tool_call["name"])
            if tool_call_func is None:
                raise AgentServiceError("未定义的工具")

            try:
                tool_message = tool_call_func.invoke(tool_call)
            except ValidationError as exception:
                raise AgentServiceError("工具调用参数格式错误") from exception
            except (OperationalError, SQLAlchemyTimeoutError) as exception:
                raise AgentServiceError("工具执行超时或数据库不可用") from exception
            except SQLAlchemyError as exception:
                raise AgentServiceError("工具执行失败") from exception

            messages.append(tool_message)

    raise AgentServiceError("Agent 未生成最终回答")
