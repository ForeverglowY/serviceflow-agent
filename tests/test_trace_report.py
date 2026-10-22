import json
import subprocess
import sys
from pathlib import Path

import pytest

from serviceflow.trace_report import build_trace_report

TRACE_ID = "a" * 32


def test_report_collects_only_matching_trace_and_safe_fields():
    lines = [
        f"INFO operation_end trace_id={TRACE_ID} operation=rag.retrieve outcome=success elapsed_ms=20.5 private=secret\n",
        f"INFO model_usage trace_id={TRACE_ID} operation=rag.generate_answer prompt_tokens=10 completion_tokens=5 total_tokens=15\n",
        f"INFO model_usage trace_id={TRACE_ID} operation=llm.classify_intent prompt_tokens=2 completion_tokens=3 total_tokens=5\n",
        f"INFO http_response_ready trace_id={TRACE_ID} method=POST status=200 elapsed_ms=80\n",
        f"INFO operation_end trace_id={'b' * 32} operation=other outcome=error elapsed_ms=1\n",
        "private customer message secret\n",
    ]
    report = build_trace_report(lines, TRACE_ID)
    assert len(report["events"]) == 4
    assert report["events"][0]["elapsed_ms"] == 20.5
    assert report["reported_token_totals"] == {
        "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
    }
    assert "secret" not in str(report)
    assert "operation': 'other" not in str(report)


def test_stream_report_does_not_invent_token_usage():
    report = build_trace_report([
        f"stream_start trace_id={TRACE_ID}",
        f"stream_end trace_id={TRACE_ID} outcome=success elapsed_ms=100 first_chunk_ms=20.5 chunk_count=3",
    ], TRACE_ID)
    assert report["reported_token_totals"] is None
    assert report["events"][-1]["chunk_count"] == 3


def test_report_skips_malformed_numeric_fields():
    report = build_trace_report([
        f"operation_end trace_id={TRACE_ID} operation=bad elapsed_ms=broken",
    ], TRACE_ID)
    assert report["events"] == []


def test_report_rejects_invalid_trace_id():
    with pytest.raises(ValueError):
        build_trace_report([], "not-a-trace-id")


def test_logging_config_captures_one_trace_event_and_cli_reports_it(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    captured = subprocess.run(
        [sys.executable, "-c", (
            "import json, logging.config; "
            "logging.config.dictConfig(json.load(open('logging_config.json'))); "
            "from serviceflow.observability import current_trace_id, traced_call; "
            f"current_trace_id.set('{TRACE_ID}'); "
            "traced_call('test.operation', lambda: None)"
        )], cwd=project_root, capture_output=True, text=True, check=True,
    )
    assert captured.stderr.count("operation_end") == 1
    log_file = tmp_path / "server.log"
    log_file.write_text(captured.stderr, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.report_trace", str(log_file), TRACE_ID],
        cwd=project_root, capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report["events"][0]["operation"] == "test.operation"
    assert report["events"][0]["outcome"] == "success"
