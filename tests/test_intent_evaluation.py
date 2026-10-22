from unittest.mock import Mock

from serviceflow.evaluation import evaluate_intent_cases, format_intent_report
from serviceflow.llm.models import CustomerIntent, IntentResult


def case(case_id="TEST"):
    return {"case_id": case_id, "category": "normal", "mode": "live", "turns": ["查订单ORDER"],
            "expected": {"routes": ["tools"], "order_id": "ORDER", "ticket_created": False}}


def test_classification_pass_fail_error_and_skip_are_separate():
    cases = [case(str(index)) for index in range(4)]
    classifier = Mock(side_effect=[
        IntentResult(intent=CustomerIntent.ORDER_QUERY, order_id="ORDER", confidence=1, reason="测试"),
        IntentResult(intent=CustomerIntent.ORDER_QUERY, order_id="WRONG", confidence=1, reason="测试"),
        RuntimeError("private-key-details"),
    ])
    report = evaluate_intent_cases(cases, {str(index): ["order_query"] for index in range(3)}, classifier)
    assert report["counts"] == {"PASS": 1, "FAIL": 1, "ERROR": 1, "SKIP": 1}
    assert report["pass_rate"] == 1 / 3
    assert "private-key-details" not in str(report)
    assert report["cases"][1]["checks"]["order_id"] is False
    assert "ERROR" in format_intent_report(report)


def test_multiturn_fixture_and_writes_are_never_executed():
    cases = [case(str(index)) for index in range(3)]
    cases[0]["turns"].append("确认")
    cases[1]["mode"] = "fixture"
    cases[2]["expected"]["ticket_created"] = True
    classifier = Mock()
    report = evaluate_intent_cases(cases, {str(index): ["order_query"] for index in range(3)}, classifier)
    classifier.assert_not_called()
    assert report["counts"]["SKIP"] == 3
    assert report["pass_rate"] is None
