from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from serviceflow.observability import current_trace_id, traced_call


def test_traced_call_forwards_arguments_and_preserves_result(caplog):
    expected = object()

    def operation(value, *, option):
        assert value == 3
        assert option == "test"
        return expected

    token = current_trace_id.set("trace-unit-test")
    try:
        with caplog.at_level("INFO", logger="serviceflow.observability"):
            assert traced_call("test.operation", operation, 3, option="test") is expected
    finally:
        current_trace_id.reset(token)
    assert "trace_id=trace-unit-test" in caplog.text
    assert "outcome=success" in caplog.text
    assert "elapsed_ms=" in caplog.text


def test_traced_call_preserves_original_exception(caplog):
    error = RuntimeError("private-error-detail")

    def fail():
        raise error

    with caplog.at_level("INFO", logger="serviceflow.observability"):
        with pytest.raises(RuntimeError) as caught:
            traced_call("test.failure", fail)
    assert caught.value is error
    assert "outcome=error" in caplog.text
    assert "private-error-detail" not in caplog.text


def test_traced_call_logs_provider_usage_not_customer_content(caplog):
    result = SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
    ), content="private-answer")
    with caplog.at_level("INFO", logger="serviceflow.observability"):
        assert traced_call("test.model", lambda: result) is result
    assert "prompt_tokens=10" in caplog.text
    assert "completion_tokens=5" in caplog.text
    assert "total_tokens=15" in caplog.text
    assert "private-answer" not in caplog.text


def test_traced_call_logs_langchain_usage_and_preserves_message(caplog):
    result = AIMessage(content="private-answer", usage_metadata={
        "input_tokens": 20, "output_tokens": 7, "total_tokens": 27,
    })
    with caplog.at_level("INFO", logger="serviceflow.observability"):
        assert traced_call("agent.call_model", lambda: result) is result
    assert "prompt_tokens=20" in caplog.text
    assert "completion_tokens=7" in caplog.text
    assert "total_tokens=27" in caplog.text
    assert caplog.text.count("model_usage ") == 1
    assert "private-answer" not in caplog.text


def test_langchain_message_without_usage_does_not_report_zero_tokens(caplog):
    result = AIMessage(content="private-answer")
    with caplog.at_level("INFO", logger="serviceflow.observability"):
        assert traced_call("agent.call_model", lambda: result) is result
    assert "model_usage " not in caplog.text
    assert "outcome=success" in caplog.text
