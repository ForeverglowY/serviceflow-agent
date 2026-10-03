from sentence_transformers import SentenceTransformer

from serviceflow.rag.embeddings import MODEL_NAME

def main() -> None:
    model = SentenceTransformer(model_name_or_path=MODEL_NAME)
    text = "我的蓝牙耳机左耳没有声音"
    embeddings = model.encode(
        [text],
        normalize_embeddings=True,
    )

    vector = embeddings[0]
    print(type(embeddings))
    print(embeddings.shape)
    print(len(vector))
    print(vector[:10])

    query = "我的耳机左边没有声音，可以退货吗？"
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_text = instruction + query

    passages = [
        "已拆封耳机出现单侧无声时，可以申请质量检测。",
        "订单发货后，消费者可以通过物流单号查询运输进度。",
        "电子发票将在订单完成后发送至用户填写的邮箱。",
    ]

    query_vector = model.encode(query_text, normalize_embeddings=True)

    passage_vector = model.encode(passages, normalize_embeddings=True)

    # @表示矩阵乘法
    scores = passage_vector @ query_vector

    for passage, score in zip(passages, scores):
        print(f"相似度：{score:.4f}")
        print(f"内容：{passage}")
        print("-" * 50)


if __name__ == "__main__":
    main()
