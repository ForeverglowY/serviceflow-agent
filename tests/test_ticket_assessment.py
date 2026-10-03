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
