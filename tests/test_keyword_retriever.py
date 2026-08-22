from pathlib import Path

from keyword_retriever import retrieve_by_keywords
from policy_loader import load_policies, split_policy
from rag_models import PolicyChunk


POLICY_DIRECTORY = Path("knowledge/policies")


def load_all_chunks() -> list[PolicyChunk]:
    return [
        chunk
        for policy in load_policies(POLICY_DIRECTORY)
        for chunk in split_policy(policy)
    ]


def test_retrieve_returns_global_top_three_matches() -> None:
    results = retrieve_by_keywords(
        load_all_chunks(),
        ["耳机", "无声", "退货"],
        top_k=3,
    )

    assert [chunk.chunk_id for chunk, _ in results] == [
        "POL-RETURN-003-001",
        "POL-RETURN-003-002",
        "POL-RETURN-003-003",
    ]
    assert [score for _, score in results] == [2, 2, 2]


def test_retrieve_respects_top_k() -> None:
    results = retrieve_by_keywords(
        load_all_chunks(),
        ["耳机", "无声", "退货"],
        top_k=2,
    )

    assert len(results) == 2


def test_retrieve_omits_chunks_with_zero_score() -> None:
    results = retrieve_by_keywords(
        load_all_chunks(),
        ["完全不存在的关键词"],
    )

    assert results == []
