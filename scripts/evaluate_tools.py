import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from serviceflow.tool_evaluation import evaluate_tool_cases, format_tool_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="第一轮工具选择评测，不执行工具")
    parser.add_argument("--run", action="store_true", help="允许真实模型调用及API费用")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation/workflow_cases.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "evaluation/tool_labels.json").read_text(encoding="utf-8"))
    print(f"本次6条案例，仅请求工具选择，不执行工具或数据库写入：{', '.join(labels)}")
    if not args.run:
        print("当前仅预览；加 --run 执行。")
        return
    from serviceflow.agent.runtime import AGENT_SYSTEM_MESSAGE, call_deepseek
    from serviceflow.settings import settings

    def request_model(text):
        return call_deepseek([AGENT_SYSTEM_MESSAGE, {"role": "user", "content": text}])

    report = evaluate_tool_cases(cases, labels, request_model)
    report.update(model=settings.deepseek_model, created_at=datetime.now(timezone.utc).isoformat())
    output = ROOT / "outputs/evaluations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-tools-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / "tools.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "tools.md").write_text(format_tool_report(report), encoding="utf-8")
    print(format_tool_report(report))
    print(f"报告已保存：{output}")


if __name__ == "__main__":
    main()
