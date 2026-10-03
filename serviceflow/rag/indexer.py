from datetime import date
from pathlib import Path

from numpy import ndarray
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from serviceflow.db.database import engine
from serviceflow.db.models import PolicyChunkTable
from serviceflow.rag.embeddings import MODEL_NAME
from serviceflow.rag.loader import load_policies, split_policy
from serviceflow.rag.models import PolicyChunk
from serviceflow.rag.utils import calculate_content_hash


class PolicyIndexState(BaseModel):
    content_hash: str
    embedding_model: str
    version: str
    effective_date: date
    status: str
    category: str
    applicable_products: str
    source: str


def build_policy_chunk_record(chunk: PolicyChunk, embedding: ndarray, ) -> PolicyChunkTable:
    # PolicyChunk + NumPy向量 -> PolicyChunkTable ORM记录
    content_hash = calculate_content_hash(chunk.text_for_embedding())

    embedding_values: list[float] = [
        float(value)
        for value in embedding
    ]

    return PolicyChunkTable(
        chunk_id=chunk.chunk_id,
        policy_id=chunk.policy_id,
        title=chunk.title,
        section=chunk.section,
        content=chunk.content,
        category=chunk.category,
        version=chunk.version,
        effective_date=chunk.effective_date,
        status=chunk.status,
        applicable_products=chunk.applicable_products,
        source=str(chunk.source),
        content_hash=content_hash,
        embedding_model=MODEL_NAME,
        embedding=embedding_values,
    )


def load_existing_index_state() -> dict[str, PolicyIndexState]:
    existing_state: dict[
        str,
        PolicyIndexState,
    ] = {}

    with Session(engine) as session:
        rows = session.execute(
            select(
                PolicyChunkTable.chunk_id,
                PolicyChunkTable.content_hash,
                PolicyChunkTable.embedding_model,
                PolicyChunkTable.version,
                PolicyChunkTable.effective_date,
                PolicyChunkTable.status,
                PolicyChunkTable.category,
                PolicyChunkTable.applicable_products,
                PolicyChunkTable.source,
            )
        )

        for (
                chunk_id,
                content_hash,
                embedding_model,
                version,
                effective_date,
                status,
                category,
                applicable_products,
                source
        ) in rows:
            existing_state[chunk_id] = PolicyIndexState(
                content_hash=content_hash,
                embedding_model=embedding_model,
                version=version,
                effective_date=effective_date,
                status=status,
                category=category,
                applicable_products=applicable_products,
                source=source,
            )

        return existing_state


def find_chunks_to_index(
        chunks: list[PolicyChunk],
        existing_state: dict[str, PolicyIndexState],
) -> tuple[list[PolicyChunk], int]:
    skipped_count: int = 0
    chunks_to_index: list[PolicyChunk] = []

    for chunk in chunks:
        content_hash = calculate_content_hash(
            chunk.text_for_embedding()
        )

        # 使用 get，如果有新增的，则 current_state = None，下面判断为 false，不会跳过
        current_state = existing_state.get(
            chunk.chunk_id
        )

        expected_state = PolicyIndexState(
            content_hash=content_hash,
            embedding_model=MODEL_NAME,
            version=chunk.version,
            effective_date=chunk.effective_date,
            status=chunk.status,
            category=chunk.category,
            applicable_products=chunk.applicable_products,
            source=str(chunk.source),
        )

        if current_state == expected_state:
            skipped_count += 1
        else:
            chunks_to_index.append(chunk)

    return chunks_to_index, skipped_count


def main() -> None:
    policies = load_policies(Path("knowledge/policies"))
    all_chunks: list[PolicyChunk] = []

    for policy in policies:
        chunks = split_policy(policy)
        all_chunks.extend(chunks)

    existing_state = load_existing_index_state()

    chunks_to_index, skipped_count = find_chunks_to_index(all_chunks, existing_state)

    inserted_count = sum(
        chunk.chunk_id not in existing_state
        for chunk in chunks_to_index
    )

    updated_count = len(chunks_to_index) - inserted_count

    if not chunks_to_index:
        print("新增：0")
        print("更新：0")
        print(f"跳过：{skipped_count}")
        return

    embedding_texts = [
        chunk.text_for_embedding()
        for chunk in chunks_to_index
    ]

    model = SentenceTransformer(model_name_or_path=MODEL_NAME)

    embeddings = model.encode(
        embedding_texts,
        normalize_embeddings=True,
    )

    records: list[PolicyChunkTable] = [
        build_policy_chunk_record(chunk, embedding)
        for chunk, embedding in zip(chunks_to_index, embeddings, strict=True)
    ]

    with Session(engine) as session:
        with session.begin():
            for record in records:
                # merge()会根据主键chunk_id判断：
                # 数据库不存在：执行INSERT
                # 数据库已经存在：执行UPDATE
                session.merge(record)

    print(f"新增：{inserted_count}")
    print(f"更新：{updated_count}")
    print(f"跳过：{skipped_count}")


if __name__ == "__main__":
    main()
