from datetime import date
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from serviceflow.db.database import engine
from serviceflow.db.models import LogisticsTable, OrderTable, ProductTable, TicketConfirmationTable, TicketTable
from serviceflow.models import Logistics, LogisticsStatus, Order, Product, Ticket, TicketConfirmationCreate, \
    TicketCreate
from serviceflow.ticket.rules import ReturnEligibilityInput, ReturnReason, TicketEligibilityDecision, \
    check_order_ticket_eligibility


class TicketServiceError(Exception):
    pass


class TicketOrderNotFoundError(TicketServiceError):
    pass


class TicketCreateConflictError(TicketServiceError):
    pass


class TicketEligibilityRejectedError(TicketServiceError):
    pass


class TicketProductNotFoundError(TicketServiceError):
    pass


class TicketLogisticsNotFoundError(TicketServiceError):
    pass


def create_ticket(ticket_create: TicketCreate, confirmation_create: TicketConfirmationCreate | None = None) -> Ticket:
    try:
        with Session(engine) as session:
            with session.begin():
                # 检查 order_id 对应的订单是否存在
                order_record = session.get(OrderTable, ticket_create.order_id)
                if order_record is None:
                    raise TicketOrderNotFoundError("订单不存在")
                order = Order.model_validate(order_record, from_attributes=True, )
                # 根据订单本身判断能不能继续申请售后
                eligibility = check_order_ticket_eligibility(order)
                if eligibility.decision != TicketEligibilityDecision.ALLOW:
                    raise TicketEligibilityRejectedError(eligibility.reason)

                ticket_id = f"TICKET-{uuid4().hex}"
                ticket = Ticket(ticket_id=ticket_id, **ticket_create.model_dump(), )
                ticket_record = TicketTable(**ticket.model_dump(mode="json"), )
                session.add(ticket_record)
                session.flush()

                if confirmation_create is not None:
                    confirmation = TicketConfirmationTable(ticket_id=ticket_id,
                                                           **confirmation_create.model_dump(), )
                    session.add(confirmation)
                    session.flush()

                return Ticket.model_validate(ticket_record, from_attributes=True, )

    except IntegrityError as error:
        raise TicketCreateConflictError("工单创建冲突") from error


def load_return_eligibility_input(
        order_id: str,
        issue_description: str,
        requested_action: str,
        return_reason: ReturnReason,
        requested_at: date,
) -> ReturnEligibilityInput:
    # 负责查数据库并组装数据
    # 用 order_id 查订单
    received_at = None
    with Session(engine) as session:
        with session.begin():
            order_record = session.get(OrderTable, order_id)
            if order_record is None:
                raise TicketOrderNotFoundError("订单不存在")

            order = Order.model_validate(order_record, from_attributes=True, )

            # 根据 order.product_id查商品，取category
            product_record = session.get(ProductTable, order.product_id)
            if product_record is None:
                raise TicketProductNotFoundError("商品不存在")
            product = Product.model_validate(product_record, from_attributes=True, )

            # 根据 order_id 查询物流信息
            logistics_record = session.get(LogisticsTable, order_id)
            if logistics_record:
                logistics = Logistics.model_validate(logistics_record, from_attributes=True, )

                if logistics.status == LogisticsStatus.DELIVERED and logistics.signed_at:
                    received_at = logistics.signed_at

    return ReturnEligibilityInput(
        product_category=product.category,
        is_opened=None,
        issue_description=issue_description,
        requested_action=requested_action,
        quality_inspection_passed=None,
        return_reason=return_reason,
        received_at=received_at,
        requested_at=requested_at,
        requires_special_review=product.requires_special_review,
    )


# 人工处理分流
def requires_manual_review(decision: str) -> bool:
    if decision == "need_manual_review" or decision == "reject":
        return True
    if decision == "allow":
        return False

    raise ValueError("错误的decision值")
