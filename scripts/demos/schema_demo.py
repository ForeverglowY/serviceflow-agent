import json

from serviceflow.models import Ticket

def main() -> None:
    schema = Ticket.model_json_schema()
    print(json.dumps(schema, ensure_ascii=False, indent=2))

    LLM_OUTPUT = {
        "ticket_id": "TICKET-AI-001",
        "order_id": "20260721001",
        "user_id": "USER-1001",
        "issue_type": "product_fault",
        "description": "客户反馈蓝牙耳机左耳没有声音",
        "priority": "high",
    }

    # 使用
    # Ticket.model_validate(LLM_OUTPUT)
    # 转换成
    # Ticket。
    ticket = Ticket.model_validate(LLM_OUTPUT)
    # 打印完整的
    # ticket。
    print(ticket)
    # 分别打印它的
    # priority、status
    print(ticket.priority)
    print(ticket.status)
    print(ticket.assigned_to)
    # 和
    # assigned_to。
    # 使用
    # ticket.model_dump()
    print(ticket.model_dump())
    # 把对象重新转成字典并打印。
if __name__ == '__main__':
    main()
