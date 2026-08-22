import json
from pathlib import Path

from numpy import ndarray
from sentence_transformers import SentenceTransformer

from embedding_demo import MODEL_NAME
from policy_loader import load_policies, policy_to_document_chunk, split_policy
from rag_models import PolicyChunk, RetrievalEvalCase
from vector_retriever import retrieve_by_vector


def evaluate_strategy(
        model: SentenceTransformer,
        cases: list[RetrievalEvalCase],
        chunks: list[PolicyChunk],
        chunk_vectors: ndarray,
        top_k: int = 3,
) -> float:
    pass_count = 0
    # 使用指定的一组片段和向量运行全部评测问题，返回Hit@K百分比。
    for case in cases:
        results = retrieve_by_vector(model, chunks, chunk_vectors, case.query, top_k)
        retrieved_policy_ids = [
            chunk.policy_id
            for chunk, _similarity in results
        ]

        # 从预期片段ID中得到政策ID。
        # 例如：
        # expected_chunk_id = "POL-RETURN-003-003"
        # 要得到：
        # "POL-RETURN-003"
        expected_policy_ids = [
            expected_id.rsplit("-", maxsplit=1)[0]
            for expected_id in case.expected_chunk_ids
        ]

        hit = any(
            expected_id in retrieved_policy_ids
            for expected_id in expected_policy_ids
        )

        if hit:
            pass_count += 1
            print(f"PASS {case.case_id}")
        else:
            print(f"FAIL {case.case_id}")
            print(f"预期政策：{expected_policy_ids}")
            print(f"实际召回：{retrieved_policy_ids}")

    return pass_count / len(cases) * 100


def main() -> None:
    # 加载10篇政策。
    policies = load_policies(Path("knowledge/policies"))

    # 加载10条评测数据。
    json_path = Path("evaluation/retrieval_cases.json")
    json_text = json_path.read_text(encoding="utf-8")

    raw_cases = json.loads(json_text)

    cases: list[RetrievalEvalCase] = [
        RetrievalEvalCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]

    model = SentenceTransformer(MODEL_NAME)

    document_chunks: list[PolicyChunk] = []
    section_chunks: list[PolicyChunk] = []

    for policy in policies:
        # 生成10个整篇文档片段。
        document_chunks.append(policy_to_document_chunk(policy))

        # 生成40个章节片段。
        section_chunks.extend(split_policy(policy))

    document_texts = [
        chunk.text_for_embedding()
        for chunk in document_chunks
    ]

    section_texts = [
        chunk.text_for_embedding()
        for chunk in section_chunks
    ]

    # 分别生成两组向量：
    document_vectors = model.encode(document_texts, normalize_embeddings=True, )
    section_vectors = model.encode(section_texts, normalize_embeddings=True, )

    document_hit_3 = evaluate_strategy(model, cases, document_chunks, document_vectors, top_k=3)
    section_hit_3 = evaluate_strategy(model, cases, section_chunks, section_vectors, top_k=3)

    print(f"整篇文档切分 Hit@3：{document_hit_3:.2f}%")
    print(f"章节切分 Hit@3：{section_hit_3:.2f}%")

    document_hit_1 = evaluate_strategy(
        model,
        cases,
        document_chunks,
        document_vectors,
        top_k=1,
    )

    section_hit_1 = evaluate_strategy(
        model,
        cases,
        section_chunks,
        section_vectors,
        top_k=1,
    )

    print(f"整篇文档切分 Hit@1：{document_hit_1:.2f}%")
    print(f"章节切分 Hit@1：{section_hit_1:.2f}%")
    print(f"整篇文档切分 Hit@3：{document_hit_3:.2f}%")
    print(f"章节切分 Hit@3：{section_hit_3:.2f}%")


if __name__ == "__main__":
    main()
