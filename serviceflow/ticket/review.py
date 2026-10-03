from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from serviceflow.db.database import engine
from serviceflow.db.models import TicketConfirmationTable, TicketReviewTaskTable
from serviceflow.ticket.service import TicketServiceError, requires_manual_review


def enqueue_manual_review(ticket_id: str) -> None:
    """根据已保存的确认快照登记待人工核验任务，同一工单最多登记一次。"""
    if not ticket_id or not ticket_id.strip():
        raise ValueError("工单编号不能为空")

    with Session(engine) as session:
        with session.begin():
            confirmation = session.get(TicketConfirmationTable, ticket_id)
            if confirmation is None:
                raise TicketServiceError("缺少工单确认记录，无法登记人工核验")

            decision = confirmation.application_snapshot.get("decision")
            reason = confirmation.application_snapshot.get("reason")
            if decision is None or not requires_manual_review(decision):
                raise TicketServiceError("该工单的预评估结果不需要人工核验")
            if not reason or not reason.strip():
                raise TicketServiceError("缺少预评估原因，无法登记人工核验")

            statement = insert(TicketReviewTaskTable).values(
                ticket_id=ticket_id, decision=decision, reason=reason,
            ).on_conflict_do_nothing(index_elements=[TicketReviewTaskTable.ticket_id])
            session.execute(statement)
