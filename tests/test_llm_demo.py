from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError

import llm_service
from llm_models import CustomerIntent
from llm_service import LLMServiceError, classify_intent


def make_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        model="deepseek-v4-flash",
        usage=SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
    )


def test_classify_intent_returns_valid_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = Mock(
        return_value=make_response(
            """
            {
              "intent": "logistics_query",
              "order_id": "20260721002",
              "confidence": 0.95,
              "reason": "用户询问订单送达时间"
            }
            """
        )
    )
    monkeypatch.setattr(
        llm_service.CLIENT.chat.completions,
        "create",
        create_mock,
    )

    result = classify_intent(
        "订单20260721002什么时候能送到？"
    )

    assert result.intent is CustomerIntent.LOGISTICS_QUERY
    assert result.order_id == "20260721002"
    assert result.confidence == 0.95
    assert result.reason == "用户询问订单送达时间"

    request_arguments = create_mock.call_args.kwargs
    assert request_arguments["model"] == llm_service.settings.deepseek_model
    assert request_arguments["temperature"] == 0
    assert request_arguments["response_format"] == {
        "type": "json_object",
    }
    assert request_arguments["messages"][1]["content"] == (
        "订单20260721002什么时候能送到？"
    )


def test_classify_intent_rejects_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_service.CLIENT.chat.completions,
        "create",
        Mock(return_value=make_response(None)),
    )

    with pytest.raises(
        LLMServiceError,
        match="DeepSeek没有返回内容",
    ):
        classify_intent("查询订单")


@pytest.mark.parametrize(
    "content",
    [
        "这不是JSON",
        """
        {
          "intent": "unknown",
          "order_id": null,
          "confidence": 2,
          "reason": "错误的结构化结果"
        }
        """,
    ],
)
def test_classify_intent_rejects_invalid_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    monkeypatch.setattr(
        llm_service.CLIENT.chat.completions,
        "create",
        Mock(return_value=make_response(content)),
    )

    with pytest.raises(
        LLMServiceError,
        match="DeepSeek返回内容格式错误",
    ) as exc_info:
        classify_intent("测试消息")

    assert exc_info.value.__cause__ is not None


def test_classify_intent_converts_openai_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_error = OpenAIError("模拟DeepSeek调用失败")
    monkeypatch.setattr(
        llm_service.CLIENT.chat.completions,
        "create",
        Mock(side_effect=api_error),
    )

    with pytest.raises(
        LLMServiceError,
        match="DeepSeek服务调用失败",
    ) as exc_info:
        classify_intent("查询订单")

    assert exc_info.value.__cause__ is api_error
