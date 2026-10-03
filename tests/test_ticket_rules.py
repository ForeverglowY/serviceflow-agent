from datetime import date

import pytest

from serviceflow.models import Order, OrderStatus
from serviceflow.ticket.rules import (
    ReturnEligibilityInput,
    ReturnReason,
    TicketEligibilityDecision,
    check_order_ticket_eligibility,
    check_return_eligibility,
)


def make_order(
    *,
    is_paid: bool = True,
    status: OrderStatus = OrderStatus.PAID,
) -> Order:
    return Order(
        order_id="20260721001",
        product_name="蓝牙耳机",
        product_id="PRODUCT-001",
        price=399.0,
        quantity=1,
        is_paid=is_paid,
        status=status,
    )


def test_unpaid_order_rejects_ticket_creation() -> None:
    result = check_order_ticket_eligibility(
        make_order(is_paid=False, status=OrderStatus.PENDING)
    )

    assert result.decision == TicketEligibilityDecision.REJECT
    assert result.reason == "订单未支付，暂不能创建售后工单。"


def test_canceled_order_rejects_ticket_creation() -> None:
    result = check_order_ticket_eligibility(
        make_order(is_paid=True, status=OrderStatus.CANCELED)
    )

    assert result.decision == TicketEligibilityDecision.REJECT
    assert result.reason == "订单已取消，不能创建售后工单。"


def test_paid_non_canceled_order_allows_ticket_creation() -> None:
    result = check_order_ticket_eligibility(
        make_order(is_paid=True, status=OrderStatus.PAID)
    )

    assert result.decision == TicketEligibilityDecision.ALLOW
    assert result.reason == "订单状态允许创建售后工单。"


def test_uninspected_quality_issue_needs_review_regardless_of_wording() -> None:
    facts = ReturnEligibilityInput(
        product_category="earphone",
        is_opened=True,
        issue_description="右边耳机无法充电",
        requested_action="换货",
        quality_inspection_passed=None,
        return_reason=ReturnReason.QUALITY_ISSUE,
        requested_at=date(2026, 10, 2),
    )

    result = check_return_eligibility(facts)

    assert result.decision == TicketEligibilityDecision.NEED_MANUAL_REVIEW
    assert "检测" in result.reason
    assert "暂不能承诺换货" in result.reason


def test_unknown_facts_do_not_automatically_allow_return() -> None:
    facts = ReturnEligibilityInput(
        issue_description="商品有问题",
        requested_action="退货",
        return_reason=ReturnReason.OTHER,
        requested_at=date(2026, 10, 2),
    )

    result = check_return_eligibility(facts)

    assert result.decision == TicketEligibilityDecision.NEED_MANUAL_REVIEW


@pytest.mark.parametrize("reason", list(ReturnReason))
def test_special_review_takes_priority_over_other_return_rules(reason):
    facts = ReturnEligibilityInput(
        product_category="in_ear_earphone", is_opened=True,
        issue_description="申请退货", requested_action="退货",
        return_reason=reason, received_at=date(2026, 9, 1),
        requested_at=date(2026, 10, 3), requires_special_review=True,
    )
    result = check_return_eligibility(facts)
    assert result.decision == TicketEligibilityDecision.NEED_MANUAL_REVIEW
    assert "专项人工审核" in result.reason


def test_opened_in_ear_earphones_reject_no_reason_return() -> None:
    facts = ReturnEligibilityInput(
        product_category="in_ear_earphone",
        is_opened=True,
        issue_description="不想要了",
        requested_action="退货",
        return_reason=ReturnReason.NO_REASON,
        requested_at=date(2026, 10, 2),
    )

    result = check_return_eligibility(facts)

    assert result.decision == TicketEligibilityDecision.REJECT
    assert "已拆封" in result.reason
    assert "无理由退货" in result.reason


@pytest.mark.parametrize("is_opened", [False, None])
def test_no_reason_return_without_confirmed_opening_is_not_rejected(
    is_opened: bool | None,
) -> None:
    facts = ReturnEligibilityInput(
        product_category="in_ear_earphone",
        is_opened=is_opened,
        issue_description="不想要了",
        requested_action="退货",
        return_reason=ReturnReason.NO_REASON,
        requested_at=date(2026, 10, 2),
    )

    result = check_return_eligibility(facts)

    assert result.decision == TicketEligibilityDecision.NEED_MANUAL_REVIEW


@pytest.mark.parametrize(
    "received_at,requested_at,return_reason,expected_decision",
    [
        (date(2026, 10, 1), date(2026, 10, 8), ReturnReason.NO_REASON, TicketEligibilityDecision.NEED_MANUAL_REVIEW),
        (date(2026, 10, 1), date(2026, 10, 9), ReturnReason.NO_REASON, TicketEligibilityDecision.REJECT),
        (None, date(2026, 10, 9), ReturnReason.NO_REASON, TicketEligibilityDecision.NEED_MANUAL_REVIEW),
        (date(2026, 10, 9), date(2026, 10, 8), ReturnReason.NO_REASON, TicketEligibilityDecision.NEED_MANUAL_REVIEW),
        (date(2026, 10, 1), date(2026, 10, 9), ReturnReason.QUALITY_ISSUE, TicketEligibilityDecision.NEED_MANUAL_REVIEW),
    ],
)
def test_seven_day_limit_applies_only_to_no_reason_returns(
    received_at: date | None,
    requested_at: date,
    return_reason: ReturnReason,
    expected_decision: TicketEligibilityDecision,
) -> None:
    facts = ReturnEligibilityInput(
        product_category="keyboard",
        is_opened=False,
        issue_description="申请退货",
        requested_action="退货",
        return_reason=return_reason,
        received_at=received_at,
        requested_at=requested_at,
    )

    result = check_return_eligibility(facts)

    assert result.decision == expected_decision
