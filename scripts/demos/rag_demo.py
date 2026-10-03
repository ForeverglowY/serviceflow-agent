from sentence_transformers import SentenceTransformer

from serviceflow.rag.embeddings import MODEL_NAME
from serviceflow.rag.service import answer_policy_question


def main() -> None:
    model = SentenceTransformer(model_name_or_path=MODEL_NAME)

    questions = [
        "入耳式耳机已经拆封了，但是左耳没有声音，可以退货吗？",
        "你们公司食堂几点开门？",
    ]

    for question in questions:
        answer = answer_policy_question(
            model=model,
            user_question=question,
            top_k=3,
        )
        print(question)
        print(answer)
        print("")


if __name__ == "__main__":
    main()
