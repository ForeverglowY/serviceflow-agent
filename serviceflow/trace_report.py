"""从日志中提取一个请求的已埋点步骤，不保留原始日志内容。"""
import re
from collections.abc import Iterable

EVENT_PATTERN = re.compile(
    r"\b(operation_end|model_usage|http_response_ready|stream_start|stream_first_text|stream_end) "
)
FIELD_PATTERN = re.compile(r"\b(\w+)=([^\s]+)")


def build_trace_report(lines: Iterable[str], trace_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        raise ValueError("trace_id 必须是32位小写十六进制字符串")

    events: list[dict] = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_records = 0
    for line in lines:
        match = EVENT_PATTERN.search(line)
        if match is None:
            continue
        fields = dict(FIELD_PATTERN.findall(line[match.end():]))
        if fields.get("trace_id") != trace_id:
            continue

        event = {"event": match.group(1)}
        # 明确的字段白名单，异常详情和客户消息均不进入报告。
        for key in ("operation", "outcome", "method"):
            if key in fields:
                event[key] = fields[key]
        try:
            for key in ("elapsed_ms", "first_chunk_ms"):
                if key in fields and fields[key] != "None":
                    event[key] = float(fields[key])
            for key in ("status", "chunk_count", *token_totals):
                if key in fields and fields[key] != "None":
                    event[key] = int(fields[key])
        except ValueError:
            continue  # 忽略被截断或格式损坏的记录。
        if event["event"] == "model_usage":
            usage_records += 1
            for key in token_totals:
                token_totals[key] += event.get(key, 0)
        events.append(event)

    return {
        "trace_id": trace_id,
        "events": events,
        "usage_records": usage_records,
        "reported_token_totals": token_totals if usage_records else None,
        "note": "仅包含已埋点且被日志文件捕获的步骤；日志顺序为记录完成顺序。"
                "各步骤耗时可能嵌套，不能相加作为请求总耗时。"
                "http_response_ready不代表流式传输结束。Token仅汇总供应商已上报的数据。",
    }
