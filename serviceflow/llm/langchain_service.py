from typing import cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from openai import OpenAIError
from pydantic import ValidationError

from serviceflow.llm.chat_model import create_chat_model
from serviceflow.llm.models import IntentResult
from serviceflow.llm.service import INTENT_SYSTEM_MESSAGE, LLMServiceError


def create_intent_classifier() -> Runnable[LanguageModelInput, IntentResult]:
    """创建一次分类器"""
    model = create_chat_model()

    structured_model = model.with_structured_output(
        schema=IntentResult,
        method="json_mode",
    )

    return cast(
        Runnable[LanguageModelInput, IntentResult],
        structured_model,
    )


def classify_intent(
        user_text: str,
        *,
        classifier: Runnable[LanguageModelInput, IntentResult],
) -> IntentResult:
    """接收分类器，处理一次用户输入"""
    system_text: str = INTENT_SYSTEM_MESSAGE["content"]
    system_message = SystemMessage(content=system_text)

    human_message: HumanMessage = HumanMessage(content=user_text)

    try:
        result = classifier.invoke([system_message, human_message])
    except OpenAIError as exception:
        raise LLMServiceError("DeepSeek服务调用失败") from exception
    except (OutputParserException, ValidationError) as exception:
        raise LLMServiceError("DeepSeek返回内容格式错误") from exception

    return result
