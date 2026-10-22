"""用法：uv run python -m scripts.report_trace LOG_FILE TRACE_ID"""
import argparse
import json
from pathlib import Path

from serviceflow.trace_report import build_trace_report


def main() -> None:
    parser = argparse.ArgumentParser(description="查看单个请求的调用链报告")
    parser.add_argument("log_file", type=Path)
    parser.add_argument("trace_id")
    args = parser.parse_args()
    try:
        with args.log_file.open(encoding="utf-8") as log_file:
            report = build_trace_report(log_file, args.trace_id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not report["events"]:
        parser.error("未找到该 trace_id 的记录，请检查日志配置和文件")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
