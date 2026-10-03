from datetime import date
from typing import Any, cast

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from serviceflow.db.models import PolicyChunkTable
from serviceflow.rag import database_retriever as database_vector_retriever
from serviceflow.rag.database_retriever import retrieve_from_database


class FakeEmbeddingModel:
    def __init__(self) -> None:
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
        return np.zeros(512, dtype=np.float32)


class FakeResult:
    def __init__(
        self,
        rows: list[tuple[PolicyChunkTable, float]],
    ) -> None:
        self.rows = rows

    def all(self) -> list[tuple[PolicyChunkTable, float]]:
        return self.rows


class FakeSession:
    rows: list[tuple[PolicyChunkTable, float]] = []
    executed_statement: Any = None

    def __init__(self, _engine: Any) -> None:
        pass

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def execute(self, statement: Any) -> FakeResult:
        type(self).executed_statement = statement
        return FakeResult(type(self).rows)


def make_record(chunk_id: str, section: str) -> PolicyChunkTable:
    return PolicyChunkTable(
        chunk_id=chunk_id,
        policy_id="POL-RETURN-003",
        title="耳机类商品退换货特别规则",
        section=section,
        content="测试政策正文",
        category="return",
        version="1.1",
        effective_date=date(2026, 3, 1),
        status="active",
        applicable_products="earphone",
        source="knowledge/policies/test.md",
        content_hash="a" * 64,
        embedding_model="BAAI/bge-small-zh-v1.5",
        embedding=[0.0] * 512,
    )


def test_retrieve_from_database_returns_all_ranked_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeSession.rows = [
        (make_record("CHUNK-001", "质量问题例外"), 0.2),
        (make_record("CHUNK-002", "已拆封商品"), 0.35),
    ]
    monkeypatch.setattr(
        database_vector_retriever,
        "Session",
        FakeSession,
    )
    fake_model = FakeEmbeddingModel()

    results = retrieve_from_database(
        model=cast(SentenceTransformer, fake_model),
        query="耳机左耳没有声音，可以退货吗？",
        top_k=2,
    )

    assert [chunk.chunk_id for chunk, _ in results] == [
        "CHUNK-001",
        "CHUNK-002",
    ]
    assert [similarity for _, similarity in results] == pytest.approx(
        [0.8, 0.65]
    )
    assert fake_model.encoded_text == (
        "为这个句子生成表示以用于检索相关文章："
        "耳机左耳没有声音，可以退货吗？"
    )
    assert fake_model.normalize_embeddings is True


def test_retrieve_from_database_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeSession.rows = []
    monkeypatch.setattr(
        database_vector_retriever,
        "Session",
        FakeSession,
    )

    results = retrieve_from_database(
        model=cast(SentenceTransformer, FakeEmbeddingModel()),
        query="不存在的政策",
    )

    assert results == []


def test_retrieve_from_database_adds_metadata_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeSession.rows = []
    FakeSession.executed_statement = None
    monkeypatch.setattr(
        database_vector_retriever,
        "Session",
        FakeSession,
    )

    retrieve_from_database(
        model=cast(SentenceTransformer, FakeEmbeddingModel()),
        query="耳机拆封后左耳无声，可以退货吗？",
        top_k=3,
        category="return",
        applicable_product="earphone",
    )

    statement = FakeSession.executed_statement
    assert statement is not None
    compiled = statement.compile()
    sql = str(compiled)
    parameter_values = list(compiled.params.values())

    assert "policy_chunks.status =" in sql
    assert "policy_chunks.category =" in sql
    assert "policy_chunks.applicable_products IN" in sql
    assert "active" in parameter_values
    assert "return" in parameter_values
    assert ["earphone", "general"] in parameter_values
    assert 3 in parameter_values


def test_retrieve_from_database_excludes_future_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeSession.rows = []
    FakeSession.executed_statement = None
    monkeypatch.setattr(
        database_vector_retriever,
        "Session",
        FakeSession,
    )
    query_date = date(2026, 8, 21)

    retrieve_from_database(
        model=cast(SentenceTransformer, FakeEmbeddingModel()),
        query="耳机可以退货吗？",
        as_of_date=query_date,
    )

    statement = FakeSession.executed_statement
    assert statement is not None
    compiled = statement.compile()
    sql = str(compiled)
    parameter_values = list(compiled.params.values())

    assert "policy_chunks.effective_date <=" in sql
    assert query_date in parameter_values
