import json
from pathlib import Path


def test_workflow_eval_dataset_is_complete_and_explicit():
    path = Path(__file__).resolve().parents[1] / "evaluation/workflow_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert 30 <= len(cases) <= 50
    assert len({case["case_id"] for case in cases}) == len(cases)
    categories = {"normal", "boundary", "rag", "multiturn", "security", "reliability"}
    assert {case["category"] for case in cases} == categories
    for case in cases:
        assert case["mode"] in {"live", "component", "fixture"}
        assert case["turns"] and all(isinstance(turn, str) and turn.strip() for turn in case["turns"])
        expected = case["expected"]
        assert expected["routes"] and set(expected["routes"]) <= {"tools", "rag", "ticket", "fallback"}
        assert set(expected["tools"]) <= {"get_order", "get_logistics"}
        assert expected["criteria"] and all(isinstance(item, str) and item.strip() for item in expected["criteria"])
        assert isinstance(expected["ticket_created"], bool)
        if case["mode"] != "live" or expected["ticket_created"]:
            assert case.get("preconditions")
