from datetime import date
from unittest.mock import Mock

import pytest

from serviceflow.agent import ticket_assessment
from serviceflow.ticket.rules import (
    ReturnEligibilityInput,
    ReturnReason,
    TicketEligibilityDecision,
    TicketEligibilityResult,
)


@pytest.mark.parametrize(
    "reason,decision",
    [
        (ReturnReason.QUALITY_ISSUE, TicketEligibilityDecision.NEED_MANUAL_REVIEW),
        (ReturnReason.NO_REASON, TicketEligibilityDecision.REJECT),
    ],
)
def test_assess_return_request_connects_classification_facts_and_rules(
        monkeypatch: pytest.MonkeyPatch,
        reason: ReturnReason,
        decision: TicketEligibilityDecision,
) -> None:
    requested_at = date(2026, 10, 3)
    facts = ReturnEligibilityInput(
        product_category="in_ear_earphone",
        issue_description="左耳没有声音",
        requested_action="换货",
        return_reason=reason,
        requested_at=requested_at,
    )
    expected_result = TicketEligibilityResult(decision=decision, reason="需要核验")
    classify = Mock(return_value=Mock(return_reason=reason))
    load_facts = Mock(return_value=facts)
    check_rules = Mock(return_value=expected_result)
    monkeypatch.setattr(ticket_assessment, "classify_return_reason", classify)
    monkeypatch.setattr(ticket_assessment, "load_return_eligibility_input", load_facts)
    monkeypatch.setattr(ticket_assessment, "check_return_eligibility", check_rules)
    monkeypatch.setattr(ticket_assessment, "assess_order_ticket_eligibility", Mock(return_value=
        TicketEligibilityResult(decision=TicketEligibilityDecision.ALLOW, reason="订单允许申请")))

    actual = ticket_assessment.assess_return_request(
        order_id="20260721001",
        issue_description="左耳没有声音",
        requested_action="换货",
        requested_at=requested_at,
    )

    assert actual is expected_result
    classify.assert_called_once_with(
        issue_description="左耳没有声音",
        requested_action="换货",
    )
    load_facts.assert_called_once_with(
        order_id="20260721001",
        issue_description="左耳没有声音",
        requested_action="换货",
        return_reason=reason,
        requested_at=requested_at,
    )
    check_rules.assert_called_once_with(facts)


@pytest.mark.parametrize("reason", ["订单未支付", "订单已取消", "订单不存在"])
def test_order_rejection_short_circuits_llm_and_return_rules(monkeypatch, reason):
    rejected = TicketEligibilityResult(decision=TicketEligibilityDecision.REJECT, reason=reason)
    monkeypatch.setattr(ticket_assessment, "assess_order_ticket_eligibility", Mock(return_value=rejected))
    classify = Mock()
    facts = Mock()
    rules = Mock()
    monkeypatch.setattr(ticket_assessment, "classify_return_reason", classify)
    monkeypatch.setattr(ticket_assessment, "load_return_eligibility_input", facts)
    monkeypatch.setattr(ticket_assessment, "check_return_eligibility", rules)
    result = ticket_assessment.assess_return_request("ORDER", "坏了", "换货", date.today())
    assert result is rejected
    classify.assert_not_called()
    facts.assert_not_called()
    rules.assert_not_called()
