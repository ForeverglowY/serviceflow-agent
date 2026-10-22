"""分类器调用约定测试；不调用模型服务。"""
from unittest.mock import Mock

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError
from pydantic import ValidationError

from serviceflow.llm import langchain_service as service
from serviceflow.llm.models import CustomerIntent, IntentResult
from serviceflow.llm.service import LLMServiceError


def test_factory_binds_expected_schema(monkeypatch):
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(service, "create_chat_model", factory)
    assert service.create_intent_classifier() is model.with_structured_output.return_value
    factory.assert_called_once_with()
    model.with_structured_output.assert_called_once_with(schema=IntentResult, method="json_mode")


def test_classifier_reused_and_messages_are_per_request(monkeypatch):
    factory = Mock(side_effect=AssertionError("不应重新创建模型"))
    monkeypatch.setattr(service, "create_chat_model", factory)
    expected = IntentResult(intent=CustomerIntent.OTHER, confidence=0.9, reason="测试")
    classifier = Mock()
    classifier.invoke.return_value = expected
    for text in ("你好", "你能做什么"):
        assert service.classify_intent(text, classifier=classifier) is expected
        messages = classifier.invoke.call_args.args[0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == text
    assert classifier.invoke.call_count == 2
    factory.assert_not_called()


@pytest.mark.parametrize("error,message", [
    (OpenAIError("provider detail"), "DeepSeek服务调用失败"),
    (OutputParserException("invalid json"), "DeepSeek返回内容格式错误"),
    (ValidationError.from_exception_data("IntentResult", []), "DeepSeek返回内容格式错误"),
])
def test_errors_are_converted(error, message):
    classifier = Mock()
    classifier.invoke.side_effect = error
    with pytest.raises(LLMServiceError, match=message) as caught:
        service.classify_intent("问题", classifier=classifier)
    assert caught.value.__cause__ is error
