from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from scripts.demos import compare_chunking
from scripts.demos.compare_chunking import evaluate_strategy
from serviceflow.rag.models import PolicyChunk, RetrievalEvalCase


def make_chunk(chunk_id: str, policy_id: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id,
        policy_id=policy_id,
        title="测试政策",
        section="测试章节",
        content="测试正文",
        category="test",
        version="1.0",
        effective_date=date(2026, 1, 1),
        status="active",
        applicable_products="general",
        source=Path("knowledge/policies/test.md"),
    )


def test_evaluate_strategy_returns_hit_percentage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_a_chunk = make_chunk("POL-A-001", "POL-A")
    policy_b_chunk = make_chunk("POL-B-001", "POL-B")
    cases = [
        RetrievalEvalCase(
            case_id="CASE-001",
            query="命中问题",
            expected_chunk_ids=["POL-A-001"],
        ),
        RetrievalEvalCase(
            case_id="CASE-002",
            query="未命中问题",
            expected_chunk_ids=["POL-B-001"],
        ),
    ]

    def fake_retrieve_by_vector(
        model: SentenceTransformer,
        chunks: list[PolicyChunk],
        chunk_vectors: np.ndarray,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[PolicyChunk, float]]:
        del model, chunks, chunk_vectors, top_k
        if query == "命中问题":
            return [(policy_a_chunk, 0.9)]
        return [(policy_a_chunk, 0.8)]

    monkeypatch.setattr(
        compare_chunking,
        "retrieve_by_vector",
        fake_retrieve_by_vector,
    )

    hit_rate = evaluate_strategy(
        model=cast(SentenceTransformer, object()),
        cases=cases,
        chunks=[policy_a_chunk, policy_b_chunk],
        chunk_vectors=np.zeros((2, 2), dtype=np.float32),
        top_k=1,
    )

    assert hit_rate == 50.0
