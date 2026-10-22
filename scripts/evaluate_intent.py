"""默认预览；--run 才调用真实模型，不执行工单写入。"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from serviceflow.evaluation import evaluate_intent_cases, format_intent_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="单轮意图分类评测（真实调用会产生API费用）")
    parser.add_argument("--run", action="store_true", help="允许真实模型调用")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation/workflow_cases.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "evaluation/intent_labels.json").read_text(encoding="utf-8"))
    selected = [case["case_id"] for case in cases if case["case_id"] in labels
                and case["mode"] == "live" and len(case["turns"]) == 1
                and not case["expected"]["ticket_created"]]
    print(f"本次范围：{len(selected)}条单轮分类案例，不执行工具及工单写入。")
    print("案例：", ", ".join(selected))
    if not args.run:
        print("当前仅预览。确认后加 --run 执行真实模型评测。")
        return
    from serviceflow.llm.service import classify_intent
    from serviceflow.settings import settings

    report = evaluate_intent_cases(cases, labels, classify_intent)
    report["model"] = settings.deepseek_model
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    output = ROOT / "outputs/evaluations" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / "intent.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "intent.md").write_text(format_intent_report(report), encoding="utf-8")
    print(format_intent_report(report))
    print(f"报告已保存：{output}")


if __name__ == "__main__":
    main()
