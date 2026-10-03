import json
from pathlib import Path

from scripts.evaluate_retrieval import is_retrieval_hit
from serviceflow.rag.models import RetrievalEvalCase


def test_is_retrieval_hit_accepts_any_expected_chunk_by_default() -> None:
    case = RetrievalEvalCase(
        case_id="CASE-ANY",
        query="测试问题",
        expected_chunk_ids=["CHUNK-A", "CHUNK-B"],
    )

    assert is_retrieval_hit(case, ["CHUNK-X", "CHUNK-B"])
    assert not is_retrieval_hit(case, ["CHUNK-X", "CHUNK-Y"])


def test_is_retrieval_hit_requires_all_expected_chunks() -> None:
    case = RetrievalEvalCase(
        case_id="CASE-ALL",
        query="需要多条政策的问题",
        expected_chunk_ids=["CHUNK-A", "CHUNK-B"],
        require_all=True,
    )

    assert is_retrieval_hit(case, ["CHUNK-B", "CHUNK-A", "CHUNK-X"])
    assert not is_retrieval_hit(case, ["CHUNK-A", "CHUNK-X"])


def test_retrieval_evaluation_contains_twenty_unique_cases() -> None:
    raw_cases = json.loads(
        Path("evaluation/retrieval_cases.json").read_text(
            encoding="utf-8"
        )
    )
    cases = [
        RetrievalEvalCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]

    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert all(case.expected_chunk_ids for case in cases)
