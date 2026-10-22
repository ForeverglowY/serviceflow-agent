from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from serviceflow.llm import streaming


def chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def test_sse_encoder_preserves_json_and_event_boundary():
    import json

    event = streaming.encode_sse_event("delta", {"text": "你好\n下一行"})
    assert event.startswith("event: delta\ndata: ")
    assert event.endswith("\n\n")
    assert json.loads(event.split("data: ", 1)[1]) == {"text": "你好\n下一行"}
    assert event.count("\n") == 3


def test_events_end_with_done(monkeypatch):
    monkeypatch.setattr(streaming, "stream_customer_reply", lambda text, **kwargs: iter(["你好", "！"]))
    assert list(streaming.stream_customer_events("问题", trace_id="test-trace")) == [
        streaming.encode_sse_event("delta", {"text": "你好"}),
        streaming.encode_sse_event("delta", {"text": "！"}),
        streaming.encode_sse_event("done", {"trace_id": "test-trace"}),
    ]


def test_events_emit_safe_error_without_done(monkeypatch):
    from openai import OpenAIError

    def broken_stream(text, **kwargs):
        yield "半段回答"
        raise OpenAIError("内部故障细节")

    monkeypatch.setattr(streaming, "stream_customer_reply", broken_stream)
    events = list(streaming.stream_customer_events("问题", trace_id="test-trace"))
    assert len(events) == 2
    assert events[-1] == streaming.encode_sse_event(
        "error", {"message": "模型服务暂时不可用，请稍后重试", "trace_id": "test-trace"},
    )
    assert "内部故障细节" not in "".join(events)
    assert "event: done" not in "".join(events)


def test_stream_trace_logs_do_not_include_customer_text(monkeypatch, caplog):
    monkeypatch.setattr(streaming, "stream_customer_reply", lambda text, **kwargs: iter(["私密回答"]))
    with caplog.at_level("INFO", logger=streaming.__name__):
        list(streaming.stream_customer_events("私密问题", trace_id="trace-log-test"))
    assert "trace_id=trace-log-test" in caplog.text
    assert "stream_first_text" in caplog.text
    assert "outcome=success" in caplog.text
    assert "chunk_count=1" in caplog.text
    assert "私密问题" not in caplog.text
    assert "私密回答" not in caplog.text


def test_stream_yields_text_and_closes_context(monkeypatch):
    response = MagicMock()
    response.__iter__.return_value = iter([
        SimpleNamespace(choices=[]), chunk(None), chunk(""), chunk("您好"), chunk("，欢迎咨询"),
    ])
    response.__enter__.return_value = response
    create = MagicMock(return_value=response)
    monkeypatch.setattr(streaming.CLIENT.chat.completions, "create", create)

    generator = streaming.stream_customer_reply("你好")
    create.assert_not_called()  # 生成器在被迭代时才开始调用模型。
    assert list(generator) == ["您好", "，欢迎咨询"]
    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert create.call_args.kwargs["messages"][-1]["content"] == "你好"
    assert "tools" not in create.call_args.kwargs
    response.__exit__.assert_called_once()


def test_closing_generator_early_exits_response_context(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value = response
    response.__iter__.return_value = iter([chunk("第一段"), chunk("第二段")])
    monkeypatch.setattr(streaming.CLIENT.chat.completions, "create", MagicMock(return_value=response))
    generator = streaming.stream_customer_reply("你好")
    assert next(generator) == "第一段"
    generator.close()
    response.__exit__.assert_called_once()


def test_stream_error_exits_response_context(monkeypatch):
    def failing_stream():
        yield chunk("第一段")
        raise RuntimeError("模拟流中断")

    response = MagicMock()
    response.__enter__.return_value = response
    response.__iter__.return_value = failing_stream()
    monkeypatch.setattr(streaming.CLIENT.chat.completions, "create", MagicMock(return_value=response))
    with pytest.raises(RuntimeError, match="模拟流中断"):
        list(streaming.stream_customer_reply("你好"))
    response.__exit__.assert_called_once()


def test_stream_usage_only_chunk_is_logged_and_reported(monkeypatch, caplog):
    from serviceflow.observability import current_trace_id
    from serviceflow.trace_report import build_trace_report

    trace_id = "a" * 32
    response = MagicMock()
    response.__enter__.return_value = response
    response.__iter__.return_value = iter([
        chunk("你好"),
        SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )),
    ])
    monkeypatch.setattr(streaming.CLIENT.chat.completions, "create", MagicMock(return_value=response))
    previous_trace = current_trace_id.get()
    with caplog.at_level("INFO", logger=streaming.__name__):
        events = list(streaming.stream_customer_events("私密问题", trace_id=trace_id))
    assert current_trace_id.get() == previous_trace
    assert len(events) == 2  # usage 不作为回答文本发送给客户。
    report = build_trace_report(caplog.text.splitlines(), trace_id)
    assert report["usage_records"] == 1
    assert report["reported_token_totals"]["total_tokens"] == 15
    assert "私密问题" not in caplog.text


def test_interrupted_stream_does_not_invent_usage(monkeypatch, caplog):
    from openai import OpenAIError
    from serviceflow.trace_report import build_trace_report

    def broken():
        yield chunk("部分回答")
        raise OpenAIError("模拟中断")

    response = MagicMock()
    response.__enter__.return_value = response
    response.__iter__.return_value = broken()
    monkeypatch.setattr(streaming.CLIENT.chat.completions, "create", MagicMock(return_value=response))
    with caplog.at_level("INFO", logger=streaming.__name__):
        list(streaming.stream_customer_events("问题", trace_id="b" * 32))
    report = build_trace_report(caplog.text.splitlines(), "b" * 32)
    assert report["reported_token_totals"] is None
    assert report["events"][-1]["outcome"] == "error"
