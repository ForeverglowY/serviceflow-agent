from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from serviceflow.rag.loader import load_policy, split_policy
from serviceflow.rag.vector_retriever import retrieve_by_vector


POLICY_PATH = Path(
    "knowledge/policies/03-earphone-hygiene-return.md"
)


class FakeEmbeddingModel:
    def __init__(self, query_vector: np.ndarray) -> None:
        self.query_vector = query_vector
        self.encoded_text: str | None = None
        self.normalize_embeddings: bool | None = None

    def encode(
        self,
        text: str,
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.encoded_text = text
        self.normalize_embeddings = normalize_embeddings
        return self.query_vector


def test_retrieve_by_vector_returns_ranked_top_k() -> None:
    chunks = split_policy(load_policy(POLICY_PATH))[:3]
    chunk_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    fake_model = FakeEmbeddingModel(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    results = retrieve_by_vector(
        model=cast(SentenceTransformer, fake_model),
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        query="耳机单侧无声怎么办？",
        top_k=2,
    )

    assert [chunk.chunk_id for chunk, _ in results] == [
        "POL-RETURN-003-001",
        "POL-RETURN-003-003",
    ]
    assert [score for _, score in results] == pytest.approx(
        [1.0, 0.8]
    )
    assert fake_model.encoded_text == (
        "为这个句子生成表示以用于检索相关文章："
        "耳机单侧无声怎么办？"
    )
    assert fake_model.normalize_embeddings is True


def test_retrieve_by_vector_rejects_mismatched_inputs() -> None:
    chunks = split_policy(load_policy(POLICY_PATH))[:2]
    chunk_vectors = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )
    fake_model = FakeEmbeddingModel(
        np.array([1.0, 0.0], dtype=np.float32)
    )

    with pytest.raises(ValueError):
        retrieve_by_vector(
            model=cast(SentenceTransformer, fake_model),
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            query="测试问题",
        )
