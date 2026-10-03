from pathlib import Path

from serviceflow.rag.loader import load_policies, split_policy
from serviceflow.rag.models import PolicyChunk, PolicyDocument


def retrieve_by_keywords(
        chunks: list[PolicyChunk],
        keywords: list[str],
        top_k: int = 3,
) -> list[tuple[PolicyChunk, int]]:
    # 创建空的结果列表。
    results: list[tuple[PolicyChunk, int]] = []
    # 遍历所有chunk。
    for chunk in chunks:
        score = 0
        embedding_text = chunk.text_for_embedding()

        for keyword in keywords:
            if keyword in embedding_text:
                score += 1
        # 统计有多少个关键词出现在chunk.text_for_embedding()中。
        # 分数为0的不加入结果。
        if score > 0:
            result = (chunk, score)
            results.append(result)
    # 按分数从高到低排序。
    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )
    # 返回前top_k条。
    return results[:top_k]


def main() -> None:
    policies: list[PolicyDocument] = load_policies(Path("knowledge/policies"))
    keywords = ["耳机", "无声", "退货"]
    all_chunks: list[PolicyChunk] = []

    for policy in policies:
        chunks: list[PolicyChunk] = split_policy(policy)
        all_chunks.extend(chunks)
    results: list[tuple[PolicyChunk, int]] = retrieve_by_keywords(all_chunks, keywords, top_k=3)

    for chunk, score in results:
        print(f"政策：{chunk.title}")
        print(f"章节：{chunk.section}")
        print(f"分数：{score}")
        print("-" * 50)


if __name__ == "__main__":
    main()
