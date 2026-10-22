"""仅评测第一轮工具请求，不导入或执行工具处理器。"""
from collections.abc import Callable
from time import perf_counter

from openai.types.chat import ChatCompletion, ChatCompletionMessageFunctionToolCall
from pydantic import ValidationError

from serviceflow.llm.models import GetLogisticsArguments, GetOrderArguments

ARGUMENT_MODELS = {"get_order": GetOrderArguments, "get_logistics": GetLogisticsArguments}


def evaluate_tool_cases(cases: list[dict], labels: dict[str, dict],
                        request_model: Callable[[str], ChatCompletion]) -> dict:
    if set(labels) - {case["case_id"] for case in cases}:
        raise ValueError("工具标签包含未知案例")
    for label in labels.values():
        if not set(label["required"]) <= set(label["allowed"]) <= set(ARGUMENT_MODELS):
            raise ValueError("工具标签范围不合法")
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"]}
        label = labels.get(case["case_id"])
        if label is None or case["mode"] != "live" or len(case["turns"]) != 1 or case["expected"]["ticket_created"]:
            row.update(status="SKIP", reason="不属于第一轮工具请求评测")
            rows.append(row)
            continue
        started = perf_counter()
        row["expected"] = label
        try:
            response = request_model(case["turns"][0])
            calls = response.choices[0].message.tool_calls or []
            names = []
            parameters = []
            invalid_calls = []
            for call in calls:
                if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                    invalid_calls.append("unsupported_type")
                    continue
                name = call.function.name
                names.append(name)
                argument_model = ARGUMENT_MODELS.get(name)
                if argument_model is None:
                    invalid_calls.append("unknown_tool")
                    continue
                try:
                    args = argument_model.model_validate_json(call.function.arguments)
                except ValidationError:
                    invalid_calls.append("invalid_arguments")
                    continue
                parameters.append({"tool": name, "order_id": args.order_id})
            checks = {
                "required_tools": set(label["required"]) <= set(names),
                "allowed_tools": set(names) <= set(label["allowed"]),
                "arguments_valid": not invalid_calls,
                "no_duplicate_tools": len(names) == len(set(names)),
            }
            if "order_id" in case["expected"]:
                checks["order_id"] = all(item["order_id"] == case["expected"]["order_id"] for item in parameters)
            if not label["allowed"]:
                # 不调用工具还不够：应有文本追问，但这里只检查非空，不判语义。
                checks["has_reply"] = bool(response.choices[0].message.content and response.choices[0].message.content.strip())
            row.update(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                       actual_tools=names, actual_parameters=parameters, invalid_calls=invalid_calls)
        except Exception as error:
            row.update(status="ERROR", error_type=type(error).__name__)
        row["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        rows.append(row)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("PASS", "FAIL", "ERROR", "SKIP")}
    attempted = sum(counts[status] for status in ("PASS", "FAIL", "ERROR"))
    return {"scope": "first_model_tool_request_only", "counts": counts, "attempted": attempted,
            "pass_rate": counts["PASS"] / attempted if attempted else None, "cases": rows,
            "note": "未执行工具；未评测后续轮次、最终回答或数据库事实。缺参数案例仅检查无工具调用及非空回复。"}


def format_tool_report(report: dict) -> str:
    rate = report["pass_rate"]
    lines = ["# 第一轮工具选择评测", "", f"统计：{report['counts']}",
             f"通过率：{rate:.2%}" if rate is not None else "通过率：未知", "", report["note"], "",
             "| 案例 | 状态 | 实际工具 | 耗时ms |", "|---|---|---|---|"]
    for row in report["cases"]:
        lines.append(f"| {row['case_id']} | {row['status']} | {', '.join(row.get('actual_tools', [])) or '-'} | {row.get('elapsed_ms', '-')} |")
    return "\n".join(lines) + "\n"
