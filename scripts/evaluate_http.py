import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from serviceflow.http_evaluation import SCENARIOS, evaluate_http_scenarios

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="真实本地HTTP验收；需独占本地演示服务")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口不合法")
    print(f"{len(SCENARIOS)}个场景：查询、政策、未支付拒绝、取消；不发送确认，不主动写工单。")
    if not args.run:
        print("仅预览。确保API与本项目使用同一个本地数据库、运行时没有其他请求；加 --run 执行。")
        return
    from langgraph.checkpoint.postgres import PostgresSaver
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from serviceflow.db.database import engine
    from serviceflow.db.models import LogisticsTable, OrderTable, TicketTable
    from serviceflow.models import Logistics, Order
    from serviceflow.settings import settings

    engine.echo = False
    with Session(engine) as session:
        paid = session.get(OrderTable, "20260721001")
        unpaid = session.get(OrderTable, "20260721002")
        if paid is None or not paid.is_paid or unpaid is None or unpaid.is_paid:
            parser.error("演示订单前置条件不满足，停止评测，不修改数据库")
        logistics = session.get(LogisticsTable, "20260721001")
        if logistics is None:
            parser.error("演示物流数据不存在，停止评测")
        reference_data = {
            "order": Order.model_validate(paid, from_attributes=True).model_dump(mode="json"),
            "logistics": Logistics.model_validate(logistics, from_attributes=True).model_dump(mode="json"),
        }
    # 禁用环境代理，只访问本机。较长超时不代表SDK或服务自动重试无限次。
    opener = build_opener(ProxyHandler({}))
    base_url = f"http://127.0.0.1:{args.port}"

    def post(payload):
        request = Request(base_url + "/agent/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = opener.open(request, timeout=180)
        except HTTPError as error:
            response = error
        with response:
            return {"status": response.code, "trace_id": response.headers.get("X-Trace-ID"),
                    "body": json.loads(response.read().decode())}

    with PostgresSaver.from_conn_string(settings.checkpoint_database_url) as saver:
        def observe(thread_id):
            with Session(engine) as session:
                ids = list(session.scalars(select(TicketTable.ticket_id)).all())
            checkpoint = saver.get_tuple({"configurable": {"thread_id": thread_id}})
            state = checkpoint.checkpoint["channel_values"] if checkpoint else {}
            return {"ticket_ids": ids, "state": state}

        report = evaluate_http_scenarios(post, observe)
    report.update(base_url=base_url, created_at=datetime.now(timezone.utc).isoformat(), reference_data=reference_data)
    output = ROOT / "outputs/evaluations" / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-http-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    (output / "http.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in report["cases"]:
        print(row["case_id"], row["status"], row["checks"])
    print("统计：", report["counts"], "未执行：", report["not_run"])
    print(f"报告：{output / 'http.json'}")


if __name__ == "__main__":
    main()
