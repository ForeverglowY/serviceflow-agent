from pathlib import Path

import numpy as np
import pytest

from embedding_demo import MODEL_NAME
from policy_indexer import (
    PolicyIndexState,
    build_policy_chunk_record,
    find_chunks_to_index,
)
from policy_loader import load_policy, split_policy
from rag_models import PolicyChunk
from rag_utils import calculate_content_hash


POLICY_PATH = Path(
    "knowledge/policies/03-earphone-hygiene-return.md"
)


def make_index_state(chunk: PolicyChunk) -> PolicyIndexState:
    return PolicyIndexState(
        content_hash=calculate_content_hash(
            chunk.text_for_embedding()
        ),
        embedding_model=MODEL_NAME,
        version=chunk.version,
        effective_date=chunk.effective_date,
        status=chunk.status,
        category=chunk.category,
        applicable_products=chunk.applicable_products,
        source=str(chunk.source),
    )


def test_calculate_content_hash_is_stable_sha256_hex() -> None:
    text = "政策：耳机退换货\n章节：质量问题\n内容：单侧无声"

    first_hash = calculate_content_hash(text)
    second_hash = calculate_content_hash(text)

    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert all(character in "0123456789abcdef" for character in first_hash)


def test_calculate_content_hash_changes_with_embedding_text() -> None:
    original_hash = calculate_content_hash("政策内容")
    changed_hash = calculate_content_hash("政策内容已更新")

    assert original_hash != changed_hash


def test_build_policy_chunk_record_maps_chunk_and_embedding() -> None:
    policy = load_policy(
        Path("knowledge/policies/03-earphone-hygiene-return.md")
    )
    chunk = split_policy(policy)[2]
    embedding = np.arange(512, dtype=np.float32)

    record = build_policy_chunk_record(chunk, embedding)

    assert record.chunk_id == chunk.chunk_id
    assert record.policy_id == chunk.policy_id
    assert record.title == chunk.title
    assert record.section == chunk.section
    assert record.content == chunk.content
    assert record.category == chunk.category
    assert record.version == chunk.version
    assert record.effective_date == chunk.effective_date
    assert record.status == chunk.status
    assert record.applicable_products == chunk.applicable_products
    assert record.source == str(chunk.source)
    assert record.content_hash == calculate_content_hash(
        chunk.text_for_embedding()
    )
    assert record.embedding_model == MODEL_NAME
    assert isinstance(record.embedding, list)
    assert len(record.embedding) == 512
    assert record.embedding[:3] == [0.0, 1.0, 2.0]


def test_find_chunks_to_index_marks_new_chunk_for_indexing() -> None:
    chunk = split_policy(load_policy(POLICY_PATH))[0]

    chunks_to_index, skipped_count = find_chunks_to_index(
        [chunk],
        {},
    )

    assert chunks_to_index == [chunk]
    assert skipped_count == 0


def test_find_chunks_to_index_skips_unchanged_chunk() -> None:
    chunk = split_policy(load_policy(POLICY_PATH))[0]
    existing_state = {
        chunk.chunk_id: make_index_state(chunk)
    }

    chunks_to_index, skipped_count = find_chunks_to_index(
        [chunk],
        existing_state,
    )

    assert chunks_to_index == []
    assert skipped_count == 1


@pytest.mark.parametrize(
    "stored_hash, stored_model",
    [
        ("changed-hash", MODEL_NAME),
        ("current-hash", "different-embedding-model"),
    ],
)
def test_find_chunks_to_index_reindexes_changed_chunk(
    stored_hash: str,
    stored_model: str,
) -> None:
    chunk = split_policy(load_policy(POLICY_PATH))[0]
    if stored_hash == "current-hash":
        stored_hash = calculate_content_hash(
            chunk.text_for_embedding()
        )
    stored_state = make_index_state(chunk).model_copy(
        update={
            "content_hash": stored_hash,
            "embedding_model": stored_model,
        }
    )
    existing_state = {
        chunk.chunk_id: stored_state
    }

    chunks_to_index, skipped_count = find_chunks_to_index(
        [chunk],
        existing_state,
    )

    assert chunks_to_index == [chunk]
    assert skipped_count == 0


def test_find_chunks_to_index_reindexes_metadata_change() -> None:
    chunk = split_policy(load_policy(POLICY_PATH))[0]
    retired_state = make_index_state(chunk).model_copy(
        update={"status": "retired"}
    )

    chunks_to_index, skipped_count = find_chunks_to_index(
        [chunk],
        {chunk.chunk_id: retired_state},
    )

    assert chunks_to_index == [chunk]
    assert skipped_count == 0
