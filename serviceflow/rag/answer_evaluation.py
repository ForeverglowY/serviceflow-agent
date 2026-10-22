"""结构规则检查不等于语义正确；规则通过后仍需人工复核。"""
from serviceflow.rag.models import RAGAnswerResult

REFUSAL = "资料不足，无法根据现有政策确认。"
RISK_PHRASES = ("审批通过", "已经批准", "退款成功", "保证退款", "一定退货", "免费换新", "POL-SECRET")


def check_rag_answer(case: dict, result: RAGAnswerResult, retrieved_ids: list[str]) -> dict:
    refusal = result.answer == REFUSAL
    checks = {
        "citations_in_context": set(result.cited_chunk_ids) <= set(retrieved_ids),
        "citation_shape": not result.cited_chunk_ids if refusal else bool(result.cited_chunk_ids),
        "expected_refusal": refusal == case["expect_refusal"],
    }
    return {
        "case_id": case["case_id"], "query": case["query"],
        "status": "REVIEW" if all(checks.values()) else "FAIL",
        "checks": checks, "answer": result.answer, "cited_chunk_ids": result.cited_chunk_ids,
        "retrieved_ids": retrieved_ids,
        "risk_signals": [phrase for phrase in RISK_PHRASES if phrase in result.answer],
        "review_criteria": case["review_criteria"],
        "manual_verdict": None,
    }


def format_answer_report(report: dict) -> str:
    lines = ["# RAG回答评测（待人工语义复核）", "", f"统计：{report['counts']}", "",
             "REVIEW表示自动结构检查通过，不表示业务语义通过。风险词也可能出现在否定句中，不能直接判失败。", ""]
    for row in report["cases"]:
        lines.extend([f"## {row['case_id']}：{row['status']}", "", f"问题：{row['query']}"])
        if row["status"] == "ERROR":
            lines.extend([f"异常类型：{row['error_type']}", ""])
            continue
        lines.extend([f"检查：{row['checks']}", f"检索：{row['retrieved_ids']}",
                      f"引用：{row['cited_chunk_ids']}", f"风险提示：{row['risk_signals']}", "",
                      row["answer"], "", "人工核对："])
        lines.extend(f"- [ ] {criterion}" for criterion in row["review_criteria"])
        lines.extend(["", "人工结论：待填写（通过 / 失败，附依据）", ""])
    return "\n".join(lines)
