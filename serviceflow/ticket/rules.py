from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

from serviceflow.models import Order, OrderStatus


class TicketEligibilityDecision(str, Enum):
    ALLOW = "allow"  # 允许继续创建工单
    REJECT = "reject"  # 明确拒绝，比如订单不存在、订单已取消、未支付
    NEED_MANUAL_REVIEW = "need_manual_review"  # 不能自动判断，需要人工，比如疑似高风险、政策模糊、需要检测


class TicketEligibilityResult(BaseModel):
    decision: TicketEligibilityDecision
    reason: str = Field(min_length=1, max_length=500)


# 退换货原因
class ReturnReason(str, Enum):
    QUALITY_ISSUE = "quality_issue"
    NO_REASON = "no_reason"
    OTHER = "other"


class ReturnEligibilityInput(BaseModel):
    product_category: str | None = None
    is_opened: bool | None = None
    issue_description: str
    requested_action: str
    quality_inspection_passed: bool | None = None
    return_reason: ReturnReason
    received_at: date | None = None  # 没有签收记录时表示未知
    requested_at: date  # 本次申请的日期
    requires_special_review: bool = False  # 标记为人工专门审核


# 只根据订单本身判断能不能继续申请售后
def check_order_ticket_eligibility(order: Order) -> TicketEligibilityResult:
    if not order.is_paid:
        return TicketEligibilityResult(
            decision=TicketEligibilityDecision.REJECT,
            reason="订单未支付，暂不能创建售后工单。",
        )

    if order.status == OrderStatus.CANCELED:
        return TicketEligibilityResult(
            decision=TicketEligibilityDecision.REJECT,
            reason="订单已取消，不能创建售后工单。",
        )

    return TicketEligibilityResult(
        decision=TicketEligibilityDecision.ALLOW,
        reason="订单状态允许创建售后工单。",
    )


# 处理无理由退货的日期规则
def check_no_reason_deadline(
        facts: ReturnEligibilityInput,
) -> TicketEligibilityResult | None:
    if facts.return_reason == ReturnReason.NO_REASON and facts.requested_action == "退货":
        # 无理由退货
        if facts.received_at is None:
            return TicketEligibilityResult(
                decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
                reason="没有收货日期"
            )

        days_since_received = (facts.requested_at - facts.received_at).days
        # 超过 7 天
        if days_since_received > 7:
            return TicketEligibilityResult(
                decision=TicketEligibilityDecision.REJECT,
                reason="超过了7天"
            )

        if days_since_received < 0:
            return TicketEligibilityResult(
                decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
                reason="数据异常，人工核验"
            )

    return None


# 已拆封入耳式耳机不支持无理由退货
def check_product_restriction(
        facts: ReturnEligibilityInput,
) -> TicketEligibilityResult | None:
    # 入耳式耳机已拆封，客户申请无理由退货
    if facts.product_category == "in_ear_earphone" and facts.is_opened == True and facts.return_reason == ReturnReason.NO_REASON and facts.requested_action == "退货":
        return TicketEligibilityResult(
            decision=TicketEligibilityDecision.REJECT,
            reason="已拆封的入耳式耳机不支持无理由退货"
        )
    return None


def check_return_eligibility(
        facts: ReturnEligibilityInput,
) -> TicketEligibilityResult:
    if facts.requires_special_review:
        # 标记为专项人工审核
        return TicketEligibilityResult(
            decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
            reason="该申请需要专项人工审核，暂不能自动确认退换货条件。"
        )
    if facts.requested_action == "换货" and facts.return_reason == ReturnReason.QUALITY_ISSUE and facts.quality_inspection_passed is None:
        return TicketEligibilityResult(
            decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
            reason="疑似质量问题，需要检测，暂不能承诺换货。"
        )
    result = check_product_restriction(facts=facts)
    if result is not None:
        return result

    result = check_no_reason_deadline(facts=facts)
    if result is not None:
        return result

    return TicketEligibilityResult(
        decision=TicketEligibilityDecision.NEED_MANUAL_REVIEW,
        reason="需要人工审核"
    )
