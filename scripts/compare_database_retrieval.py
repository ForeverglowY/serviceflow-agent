"""同一有效政策范围下比较纯向量与统一分词RRF，不修改正式检索路径。"""
import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.orm import Session

from scripts.evaluate_retrieval import is_retrieval_hit
from serviceflow.db.database import engine
from serviceflow.db.models import PolicyChunkTable
from serviceflow.rag.database_retriever import retrieve_from_database
from serviceflow.rag.embeddings import MODEL_NAME
from serviceflow.rag.hybrid_retriever import fuse_rankings
from serviceflow.rag.keyword_retriever import retrieve_by_keywords
from serviceflow.rag.models import PolicyChunk, RetrievalEvalCase
from serviceflow.rag.query_keywords import extract_query_keywords

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="比较检索策略，不调用聊天模型、不写业务数据库")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.top_k <= args.candidate_k:
        parser.error("需要 1 <= top-k <= candidate-k")
    engine.echo = False
    effective_on = date.today()
    raw_cases = json.loads((ROOT / "evaluation/retrieval_cases.json").read_text(encoding="utf-8"))
    cases = [RetrievalEvalCase.model_validate(case) for case in raw_cases]
    if not cases:
        parser.error("评测集为空")
    # 两路检索使用数据库中相同状态和日期范围，不混用磁盘文件。
    with Session(engine) as session:
        records = session.scalars(select(PolicyChunkTable).where(
            PolicyChunkTable.status == "active", PolicyChunkTable.effective_date <= effective_on,
        ).order_by(PolicyChunkTable.chunk_id)).all()
        chunks = [PolicyChunk.model_validate(record, from_attributes=True) for record in records]
    if not chunks:
        parser.error("没有有效政策片段，请先索引")
    # 使用现有本地缓存，避免评测时意外下载不同版本模型。
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    rows = []
    for case in cases:
        keywords = extract_query_keywords(case.query)
        vector = retrieve_from_database(model, case.query, top_k=args.candidate_k, as_of_date=effective_on)
        keyword = retrieve_by_keywords(chunks, keywords, top_k=args.candidate_k)
        hybrid = fuse_rankings(vector, keyword, top_k=args.top_k)
        vector_ids = [chunk.chunk_id for chunk, _ in vector[:args.top_k]]
        hybrid_ids = [chunk.chunk_id for chunk, _ in hybrid]
        vector_hit = is_retrieval_hit(case, vector_ids)
        hybrid_hit = is_retrieval_hit(case, hybrid_ids)
        transition = ("improved" if hybrid_hit else "regressed") if vector_hit != hybrid_hit else "unchanged"
        rows.append({"case_id": case.case_id, "query": case.query, "keywords": keywords,
                     "expected_ids": case.expected_chunk_ids, "require_all": case.require_all,
                     "vector_ids": vector_ids, "hybrid_ids": hybrid_ids,
                     "vector_hit": vector_hit, "hybrid_hit": hybrid_hit, "transition": transition})
        print(f"{case.case_id} vector={'PASS' if vector_hit else 'FAIL'} hybrid={'PASS' if hybrid_hit else 'FAIL'} {transition}")
    vector_pass = sum(row["vector_hit"] for row in rows)
    hybrid_pass = sum(row["hybrid_hit"] for row in rows)
    print(f"纯向量 Hit@{args.top_k}：{vector_pass}/{len(rows)} = {vector_pass / len(rows):.2%}")
    print(f"混合检索 Hit@{args.top_k}：{hybrid_pass}/{len(rows)} = {hybrid_pass / len(rows):.2%}")
    improved = [row["case_id"] for row in rows if row["transition"] == "improved"]
    regressed = [row["case_id"] for row in rows if row["transition"] == "regressed"]
    print("改善案例：", improved)
    print("退步案例：", regressed)
    report = {"model": MODEL_NAME, "as_of_date": effective_on.isoformat(), "top_k": args.top_k,
              "candidate_k": args.candidate_k, "rrf_k": 60, "eligible_chunks": len(chunks),
              "vector_pass": vector_pass, "hybrid_pass": hybrid_pass, "total": len(rows),
              "improved": improved, "regressed": regressed, "cases": rows,
              "note": "实验仅评测检索，未替换正式RAG路径；未检查最终回答正确性。"}
    output = ROOT / "outputs/evaluations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-retrieval-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"对照报告：{output / 'comparison.json'}")


if __name__ == "__main__":
    main()
