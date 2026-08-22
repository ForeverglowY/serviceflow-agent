from datetime import date

from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from db_models import PolicyChunkTable
from embedding_demo import MODEL_NAME
from rag_models import PolicyChunk


def retrieve_from_database(
        model: SentenceTransformer,
        query: str,
        top_k: int = 3,
        category: str | None = None,
        applicable_product: str | None = None,
        as_of_date: date | None = None
) -> list[tuple[PolicyChunk, float]]:
    # 接收用户问题，到PostgreSQL的政策向量库中找出语义最相关的几条政策片段。

    # 如果用户没有传 as_of_date，就用今天
    effective_on = as_of_date or date.today()

    # 给用户问题生成向量
    query_text = "为这个句子生成表示以用于检索相关文章：" + query
    query_vector = model.encode(query_text, normalize_embeddings=True)

    query_values: list[float] = [
        float(value)
        for value in query_vector
    ]

    # 定义余弦距离
    distance = PolicyChunkTable.embedding.cosine_distance(query_values).label("distance")

    statement = (
        select(PolicyChunkTable, distance)
        .where(PolicyChunkTable.status == "active")
        .where(PolicyChunkTable.effective_date <= effective_on)
    )

    if category is not None:
        statement = statement.where(PolicyChunkTable.category == category)

    if applicable_product is not None:
        statement = statement.where(
            PolicyChunkTable.applicable_products.in_(
                [applicable_product, "general"]
            )
        )

    statement = (
        statement
        .order_by(distance)
        .limit(top_k)
    )

    results: list[tuple[PolicyChunk, float]] = []

    with Session(engine) as session:
        rows = session.execute(statement).all()
        for record, distance_value in rows:
            chunk = PolicyChunk.model_validate(
                record,
                from_attributes=True,
            )

            similarity = 1 - float(distance_value)
            results.append((chunk, similarity))

    return results


def main() -> None:
    query: str = "耳机拆封后左耳无声，可以退货吗？"
    model = SentenceTransformer(
        model_name_or_path=MODEL_NAME,
    )

    results = retrieve_from_database(
        model=model,
        query=query,
        category="return",
        applicable_product="earphone",
    )
    print(results)


if __name__ == "__main__":
    main()
