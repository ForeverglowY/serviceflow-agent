from pathlib import Path

from sentence_transformers import SentenceTransformer

from database_vector_retriever import retrieve_from_database
from embedding_demo import MODEL_NAME
from keyword_retriever import retrieve_by_keywords
from policy_loader import load_policies, split_policy
from rag_models import PolicyChunk


def reciprocal_rank_score(
        rank: int,
        rrf_k: int = 60,
) -> float:
    # 计算单次排名得分
    if rank < 1:
        raise ValueError('rank必须大于等于1')
    if rrf_k < 0:
        raise ValueError('rrf_k必须大于等于0')

    return 1.0 / (rrf_k + rank)


def fuse_rankings(
        vector_results: list[tuple[PolicyChunk, float]],
        keyword_results: list[tuple[PolicyChunk, int]],
        top_k: int = 3,
        rrf_k: int = 60,
) -> list[tuple[PolicyChunk, float]]:
    # 接收向量检索排行榜和关键词检索排行榜，把它们融合成一个新的排行榜。
    if top_k < 1:
        raise ValueError("top_k必须大于等于1")

    if rrf_k < 0:
        raise ValueError("rrf_k必须大于等于0")

    # chunk_id: RRF累积分数
    chunk_scores: dict[str, float] = {}

    # chunk_id: chunk
    chunk_dict: dict[str, PolicyChunk] = {}

    for rank, (chunk, _similarity) in enumerate(vector_results, start=1):
        rrf_score = reciprocal_rank_score(
            rank=rank,
            rrf_k=rrf_k, )
        chunk_scores[chunk.chunk_id] = (
                chunk_scores.get(chunk.chunk_id, 0.0)
                + rrf_score
        )
        chunk_dict[chunk.chunk_id] = chunk

    for rank, (chunk, _score) in enumerate(keyword_results, start=1):
        rrf_score = reciprocal_rank_score(
            rank=rank,
            rrf_k=rrf_k,
        )

        chunk_scores[chunk.chunk_id] = (
                chunk_scores.get(chunk.chunk_id, 0.0)
                + rrf_score
        )

        chunk_dict[chunk.chunk_id] = chunk

    sorted_scores = sorted(
        chunk_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    result: list[tuple[PolicyChunk, float]] = []
    for chunk_id, score in sorted_scores[:top_k]:
        result.append((chunk_dict[chunk_id], score))

    return result


def main() -> None:
    policies = load_policies(Path("knowledge/policies"))
    all_chunks: list[PolicyChunk] = []

    for policy in policies:
        all_chunks.extend(split_policy(policy))

    # 关键词检索与向量检索使用相同的元数据范围。
    eligible_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.status == "active"
        and chunk.category == "return"
        and chunk.applicable_products in ("earphone", "general")
    ]

    model = SentenceTransformer(model_name_or_path=MODEL_NAME)
    query = "耳机拆封后左耳无声，可以退货吗？"
    keywords = [
        "耳机",
        "拆封",
        "无声",
        "退货",
    ]
    candidate_k = 10

    vector_results = retrieve_from_database(
        model=model,
        query=query,
        top_k=candidate_k,
        category="return",
        applicable_product="earphone",
    )

    keyword_results = retrieve_by_keywords(
        chunks=eligible_chunks,
        keywords=keywords,
        top_k=candidate_k,
    )

    hybrid_results = fuse_rankings(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=3,
    )

    print(f"用户问题：{query}")
    print(f"向量候选数量：{len(vector_results)}")
    print(f"关键词候选数量：{len(keyword_results)}")
    print("-" * 50)

    for chunk, rrf_score in hybrid_results:
        print(f"片段编号：{chunk.chunk_id}")
        print(f"政策名称：{chunk.title}")
        print(f"章节：{chunk.section}")
        print(f"RRF分数：{rrf_score:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
