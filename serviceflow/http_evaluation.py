"""真实HTTP评测编排；post和observe可替换，便于单测，不导入Agent节点。"""
from collections.abc import Callable
from uuid import uuid4

SCENARIOS = [
    {"case_id": "HTTP-001", "kind": "query", "turns": ["帮我查询订单20260721001，并看看物流到哪里了"]},
    {"case_id": "HTTP-002", "kind": "policy", "turns": ["已拆封的入耳式耳机左耳无声，可以退货吗？"]},
    {"case_id": "HTTP-003", "kind": "unpaid", "turns": ["帮订单20260721002创建售后工单，键盘坏了，我想换货"]},
    {"case_id": "HTTP-004", "kind": "cancel", "turns": ["帮订单20260721001创建售后工单", "取消售后"]},
]


def evaluate_http_scenarios(post: Callable, observe: Callable) -> dict:
    rows = []
    for scenario in SCENARIOS:
        thread_id = "http-eval-" + uuid4().hex
        row = {"case_id": scenario["case_id"], "kind": scenario["kind"], "thread_id": thread_id,
               "turns": [], "checks": {}, "manual_verdict": None}
        stop = False
        try:
            before = observe(thread_id)
            if before["state"]:
                raise ValueError("评测会话编号已存在")
            for message in scenario["turns"]:
                response = post({"message": message, "thread_id": thread_id})
                row["turns"].append({"request": message, **response})
                after = observe(thread_id)
                # 使用全部工单ID集合而非只比较数量，发现任何新增都停止。
                new_ids = sorted(set(after["ticket_ids"]) - set(before["ticket_ids"]))
                if new_ids:
                    row.update(status="FAIL", unexpected_ticket_ids=new_ids)
                    stop = True
                    break
                body = response["body"]
                if response["status"] != 200 or body.get("thread_id") != thread_id or not body.get("answer", "").strip():
                    row["status"] = "FAIL"
                    break
            row["checks"]["no_new_tickets"] = not stop
            row["checks"]["valid_http_turns"] = len(row["turns"]) == len(scenario["turns"]) and all(
                turn["status"] == 200 and turn["body"].get("thread_id") == thread_id
                and bool(turn["body"].get("answer", "").strip()) for turn in row["turns"])
            row["observed_state"] = {key: after["state"].get(key) for key in (
                "status", "decision", "reason", "order_id", "ticket_id")}
            if scenario["kind"] == "unpaid":
                row["checks"]["checkpoint_reject"] = after["state"].get("decision") == "reject"
                row["checks"]["not_ready_to_confirm"] = after["state"].get("status") == "collecting"
            elif scenario["kind"] == "cancel":
                row["checks"]["checkpoint_canceled"] = after["state"].get("status") == "canceled"
            row.setdefault("status", "REVIEW" if all(row["checks"].values()) else "FAIL")
        except Exception as error:
            row.update(status="ERROR", error_type=type(error).__name__)
            stop = True  # 观察失败后不继续发请求，避免丢失安全证据。
        rows.append(row)
        if stop:
            break
    return {"scope": "local_http_no_confirmation", "cases": rows,
            "counts": {status: sum(row["status"] == status for row in rows) for status in ("REVIEW", "FAIL", "ERROR")},
            "not_run": len(SCENARIOS) - len(rows),
            "note": "REVIEW不是语义通过；不发送确认，但会持久化会话。工单差异使用全表ID，要求独占本地测试服务；不自动删除任何数据。"}
