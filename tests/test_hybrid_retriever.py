from datetime import date
from pathlib import Path

import pytest

from hybrid_retriever import fuse_rankings, reciprocal_rank_score
from rag_models import PolicyChunk


def make_chunk(chunk_id: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id,
        policy_id=f"POL-{chunk_id}",
        title=f"政策{chunk_id}",
        section="测试章节",
        content="测试正文",
        category="test",
        version="1.0",
        effective_date=date(2026, 1, 1),
        status="active",
        applicable_products="general",
        source=Path(f"knowledge/policies/{chunk_id}.md"),
    )


def test_reciprocal_rank_score_uses_rank_and_rrf_k() -> None:
    assert reciprocal_rank_score(rank=1) == pytest.approx(1 / 61)
    assert reciprocal_rank_score(rank=2, rrf_k=10) == pytest.approx(
        1 / 12
    )


@pytest.mark.parametrize(
    ("rank", "rrf_k", "message"),
    [
        (0, 60, "rank必须大于等于1"),
        (1, -1, "rrf_k必须大于等于0"),
    ],
)
def test_reciprocal_rank_score_rejects_invalid_arguments(
    rank: int,
    rrf_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_score(rank=rank, rrf_k=rrf_k)


def test_fuse_rankings_combines_both_rankings_without_mutation() -> None:
    chunk_a = make_chunk("A")
    chunk_b = make_chunk("B")
    chunk_c = make_chunk("C")
    chunk_d = make_chunk("D")
    vector_results = [
        (chunk_a, 0.9),
        (chunk_b, 0.8),
        (chunk_c, 0.7),
    ]
    keyword_results = [
        (chunk_c, 3),
        (chunk_a, 2),
        (chunk_d, 1),
    ]
    original_vector_results = list(vector_results)
    original_keyword_results = list(keyword_results)

    results = fuse_rankings(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=4,
    )

    assert [chunk.chunk_id for chunk, _ in results] == [
        "A",
        "C",
        "B",
        "D",
    ]
    assert [score for _, score in results] == pytest.approx(
        [
            1 / 61 + 1 / 62,
            1 / 63 + 1 / 61,
            1 / 62,
            1 / 63,
        ]
    )
    assert vector_results == original_vector_results
    assert keyword_results == original_keyword_results


def test_fuse_rankings_respects_top_k_and_empty_results() -> None:
    chunk_a = make_chunk("A")
    chunk_b = make_chunk("B")

    results = fuse_rankings(
        vector_results=[(chunk_a, 0.9), (chunk_b, 0.8)],
        keyword_results=[],
        top_k=1,
    )

    assert [chunk.chunk_id for chunk, _ in results] == ["A"]
    assert fuse_rankings([], []) == []


@pytest.mark.parametrize(
    ("top_k", "rrf_k", "message"),
    [
        (0, 60, "top_k必须大于等于1"),
        (3, -1, "rrf_k必须大于等于0"),
    ],
)
def test_fuse_rankings_rejects_invalid_arguments(
    top_k: int,
    rrf_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fuse_rankings([], [], top_k=top_k, rrf_k=rrf_k)
