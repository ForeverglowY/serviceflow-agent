import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from serviceflow.rag.answer_evaluation import check_rag_answer, format_answer_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG回答结构检查与人工复核，不创建工单")
    parser.add_argument("--run", action="store_true", help="允许真实模型调用和API费用")
    parser.add_argument("--top-k", type=int, default=3, help="默认与正式RAG相同")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k必须大于等于1")
    cases = json.loads((ROOT / "evaluation/rag_answer_cases.json").read_text(encoding="utf-8"))
    print(f"{len(cases)}条RAG组件级案例；Top-K={args.top_k}，绕过外层意图路由，不创建工单。")
    if not args.run:
        print("仅预览；加 --run 才调用模型。")
        return
    from sentence_transformers import SentenceTransformer
    from serviceflow.db.database import engine
    from serviceflow.rag.database_retriever import retrieve_from_database
    from serviceflow.rag.embeddings import MODEL_NAME
    from serviceflow.rag.service import build_rag_context, generate_rag_answer, RAGServiceError, validate_rag_citations
    from serviceflow.settings import settings

    engine.echo = False
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    rows = []
    for case in cases:
        started = perf_counter()
        try:
            results = retrieve_from_database(model, case["query"], top_k=args.top_k)
            result = generate_rag_answer(case["query"], build_rag_context(results))
            row = check_rag_answer(case, result, [chunk.chunk_id for chunk, _ in results])
            try:
                validate_rag_citations(result, results)
            except RAGServiceError:
                row["status"] = "FAIL"
                row["checks"]["production_citation_validator"] = False
            else:
                row["checks"]["production_citation_validator"] = True
            # 保存实际资料，复核不能仅依据片段编号猜测是否支持回答。
            row["context"] = [chunk.model_dump(mode="json") for chunk, _ in results]
        except Exception as error:
            row = {"case_id": case["case_id"], "query": case["query"],
                   "status": "ERROR", "error_type": type(error).__name__}
        row["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        rows.append(row)
        print(f"{case['case_id']} {row['status']}")
    report = {"scope":"rag_component_answer_review", "model":settings.deepseek_model,
              "embedding_model":MODEL_NAME, "top_k":args.top_k,
              "created_at":datetime.now(timezone.utc).isoformat(), "cases":rows,
              "counts":{status:sum(row["status"] == status for row in rows) for status in ("REVIEW","FAIL","ERROR")}}
    output = ROOT / "outputs/evaluations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-answers-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / "answers.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "answers.md").write_text(format_answer_report(report), encoding="utf-8")
    print("统计：",report["counts"])
    print(f"复核报告：{output}")


if __name__ == "__main__":
    main()
