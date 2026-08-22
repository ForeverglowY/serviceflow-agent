from pathlib import Path

from numpy import ndarray
from sentence_transformers import SentenceTransformer

from embedding_demo import MODEL_NAME
from policy_loader import load_policies, split_policy
from rag_models import PolicyChunk


def retrieve_by_vector(
        model: SentenceTransformer,
        chunks: list[PolicyChunk],
        chunk_vectors: ndarray,
        query: str,
        top_k: int = 3,
) -> list[tuple[PolicyChunk, float]]:
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_text = instruction + query
    query_vector = model.encode(query_text, normalize_embeddings=True, )
    scores = chunk_vectors @ query_vector

    scored_results = [
        (chunk, float(score))
        for chunk, score in zip(chunks, scores, strict=True)
    ]

    scored_results.sort(key=lambda item: item[1], reverse=True)

    return scored_results[:top_k]


def main() -> None:
    # 加载10篇政策。
    policies = load_policies(Path("knowledge/policies"))

    all_chunks: list[PolicyChunk] = []

    for policy in policies:
        chunks: list[PolicyChunk] = split_policy(policy)
        all_chunks.extend(chunks)

    # 切分成40个PolicyChunk。
    # 使用列表推导式生成Embedding文本：
    embedding_texts = [
        chunk.text_for_embedding()
        for chunk in all_chunks
    ]

    model = SentenceTransformer(
        model_name_or_path=MODEL_NAME,
    )

    # 生成所有向量
    chunk_vectors = model.encode(
        embedding_texts,
        normalize_embeddings=True,
    )

    print(f"片段数量：{len(all_chunks)}")
    print(f"Embedding文本数量：{len(embedding_texts)}")
    print(f"知识库向量形状：{chunk_vectors.shape}")

    query = "我的耳机左边没有声音，已经拆封了，可以退货吗？"
    top_results = retrieve_by_vector(
        model=model,
        chunks=all_chunks,
        chunk_vectors=chunk_vectors,
        query=query,
        top_k=3,
    )

    for chunk, score in top_results:
        print(f"相似度：{score:.4f}")
        print(f"政策：{chunk.title}")
        print(f"章节：{chunk.section}")
        print(f"正文：{chunk.content}")
        print("-" * 50)


if __name__ == "__main__":
    main()
