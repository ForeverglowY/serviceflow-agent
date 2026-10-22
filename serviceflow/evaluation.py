"""分类层评测，不执行工具、RAG或数据库写入。"""
from collections.abc import Callable
from time import perf_counter

from serviceflow.agent.routing import select_customer_route
from serviceflow.llm.models import CustomerIntent, IntentResult


def evaluate_intent_cases(cases: list[dict], labels: dict[str, list[str]],
                          classifier: Callable[[str], IntentResult]) -> dict:
    known_ids = {case["case_id"] for case in cases}
    if set(labels) - known_ids:
        raise ValueError("意图标签包含不存在的案例编号")
    for values in labels.values():
        if not values:
            raise ValueError("允许意图列表不能为空")
        for value in values:
            CustomerIntent(value)

    rows: list[dict] = []
    for case in cases:
        row = {"case_id": case["case_id"], "category": case["category"]}
        if (case["mode"] != "live" or len(case["turns"]) != 1
                or case["expected"]["ticket_created"] or case["case_id"] not in labels):
            row.update(status="SKIP", reason="不属于本次单轮意图分类范围")
            rows.append(row)
            continue
        expected = case["expected"]
        row.update(expected_intents=labels[case["case_id"]], expected_routes=expected["routes"])
        started = perf_counter()
        try:
            result = classifier(case["turns"][0])
            route = select_customer_route(result.intent)
            checks = {
                "intent": result.intent.value in row["expected_intents"],
                "route": route in expected["routes"],
            }
            if "order_id" in expected:
                checks["order_id"] = result.order_id == expected["order_id"]
            row.update(
                status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                actual_intent=result.intent.value, actual_route=route, actual_order_id=result.order_id,
            )
        except Exception as error:
            # 不写入异常原文，避免密钥、URL或内部数据进入报告。
            row.update(status="ERROR", error_type=type(error).__name__)
        row["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("PASS", "FAIL", "ERROR", "SKIP")}
    attempted = counts["PASS"] + counts["FAIL"] + counts["ERROR"]
    return {
        "scope": "single_turn_intent_only", "counts": counts, "attempted": attempted,
        "pass_rate": counts["PASS"] / attempted if attempted else None,
        "cases": rows,
        "note": "PASS只表示意图、路由及已指定订单号符合预期，不证明工具、引用、安全行为或业务写入正确。",
    }


def format_intent_report(report: dict) -> str:
    rate = report["pass_rate"]
    lines = ["# 意图分类评测报告", "", f"范围：{report['scope']}",
             f"统计：{report['counts']}",
             f"通过率：{rate:.2%}" if rate is not None else "通过率：未知（没有执行案例）",
             "", report["note"], "", "| 案例 | 状态 | 实际意图 | 实际路由 | 耗时ms |",
             "|---|---|---|---|---|"]
    for row in report["cases"]:
        lines.append(f"| {row['case_id']} | {row['status']} | {row.get('actual_intent', '-')} "
                     f"| {row.get('actual_route', '-')} | {row.get('elapsed_ms', '-')} |")
    return "\n".join(lines) + "\n"
