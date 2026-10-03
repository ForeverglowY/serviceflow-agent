import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from serviceflow.rag.database_retriever import retrieve_from_database
from serviceflow.rag.embeddings import MODEL_NAME
from serviceflow.rag.models import RetrievalEvalCase


def is_retrieval_hit(
        case: RetrievalEvalCase,
        retrieved_ids: list[str],
) -> bool:
    # case.require_all为True
    # → 使用all()
    # → 所有expected_chunk_ids都必须出现
    #
    # case.require_all为False
    # → 使用any()
    # → 任意一个expected_chunk_id出现即可

    require_all = case.require_all
    if require_all:
        return all(retrieved_id in retrieved_ids for retrieved_id in case.expected_chunk_ids)
    else:
        return any(retrieved_id in retrieved_ids for retrieved_id in case.expected_chunk_ids)


def main() -> None:
    # 用 Path.read_text() 读取评测JSON。
    path = Path("evaluation/retrieval_cases.json")
    json_text = path.read_text(encoding="utf-8")

    # 使用 json.loads() 转成Python数据。
    raw_cases = json.loads(json_text)
    # 转换成 RetrievalEvalCase 对象。
    # 只创建一次 SentenceTransformer。
    # 遍历20条问题，调用retrieve_from_database()。
    # 从返回结果中提取 chunk_id。
    # 判断预期ID是否出现在检索结果中。
    # 打印每条的 PASS 或 FAIL。
    # 最后打印指定Top-K下的总体命中率。

    cases: list[RetrievalEvalCase] = [
        RetrievalEvalCase.model_validate(case)
        for case in raw_cases
    ]

    model = SentenceTransformer(MODEL_NAME)

    pass_count = 0

    top_k = 5

    for case in cases:
        results = retrieve_from_database(model, case.query, top_k)

        retrieved_ids = [
            chunk.chunk_id
            for chunk, _similarity in results
        ]

        hit = is_retrieval_hit(
            case=case,
            retrieved_ids=retrieved_ids,
        )

        if hit:
            print(f"PASS {case.case_id}")
            pass_count += 1
        else:
            print(f"FAIL {case.case_id}")
            print(f"预期片段：{case.expected_chunk_ids}")
            print(f"实际召回：{retrieved_ids}")

    print(
        f"Hit@{top_k}：{pass_count}/{len(cases)}"
        f" = {pass_count / len(cases) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
