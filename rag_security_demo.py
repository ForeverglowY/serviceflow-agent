from pathlib import Path

from sentence_transformers import SentenceTransformer

from embedding_demo import MODEL_NAME
from policy_loader import load_policy, split_policy
from rag_service import build_rag_context, format_rag_answer, generate_rag_answer, validate_rag_citations
from vector_retriever import retrieve_by_vector


def main() -> None:
    query = "我的耳机左耳没有声音，应该怎么办？"

    policy = load_policy(Path("evaluation/security_documents/bad.md"))

    chunks = split_policy(policy)

    model = SentenceTransformer(model_name_or_path=MODEL_NAME)

    embedding_texts = [
        chunk.text_for_embedding()
        for chunk in chunks
    ]

    chunk_vectors = model.encode(embedding_texts, normalize_embeddings=True)

    results = retrieve_by_vector(model=model, chunks=chunks, chunk_vectors=chunk_vectors, query=query, top_k=3)

    rag_context = build_rag_context(results)

    result = generate_rag_answer(user_question=query, context=rag_context)
    print("模型结构化结果：", result)
    validate_rag_citations(result, results)

    formatted_answer = format_rag_answer(result, results)
    print("最终回答：", formatted_answer)


if __name__ == "__main__":
    main()
