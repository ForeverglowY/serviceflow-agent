from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from serviceflow.models import OrderStatus, TicketCreate, TicketConfirmationCreate
from serviceflow.db.models import LogisticsTable, OrderTable, ProductTable, TicketTable, TicketConfirmationTable
from sqlalchemy.exc import IntegrityError
from serviceflow.ticket import service as ticket_service
from serviceflow.ticket.rules import ReturnReason, check_return_eligibility, TicketEligibilityDecision


def make_ticket_create() -> TicketCreate:
    return TicketCreate(
        order_id="20260721001",
        user_id="USER-DEMO",
        issue_type="product_fault",
        description="左耳没有声音；用户希望处理方式：换货",
    )


@pytest.mark.parametrize("decision,expected", [
    ("need_manual_review", True),
    ("reject", True),
    ("allow", False),
])
def test_manual_review_routing_policy(decision, expected):
    assert ticket_service.requires_manual_review(decision) is expected


def test_manual_review_routing_rejects_unknown_decision():
    with pytest.raises(ValueError):
        ticket_service.requires_manual_review("unexpected")


@pytest.mark.parametrize("fail_confirmation_write", [False, True])
def test_confirmation_is_written_in_ticket_transaction(monkeypatch, fail_confirmation_write):
    order_record = SimpleNamespace(
        order_id="20260721001", product_id="PRODUCT-001", product_name="耳机",
        price=399.0, quantity=1, is_paid=True, status="paid",
        tracking_number=None, customer_note=None,
    )
    session = Mock()
    session.get.return_value = order_record
    transaction = MagicMock()
    session.begin.return_value = transaction
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    monkeypatch.setattr(ticket_service, "Session", Mock(return_value=session_context))
    confirmation = TicketConfirmationCreate(
        confirmed_by="USER-DEMO",
        confirmed_at=datetime(2026, 10, 3, 8, 30, tzinfo=timezone.utc),
        confirmation_message="确认",
        application_snapshot={"order_id": "20260721001", "requested_action": "换货"},
    )
    if fail_confirmation_write:
        error = IntegrityError("insert confirmation", {}, Exception("write failed"))
        session.flush.side_effect = [None, error]
        with pytest.raises(ticket_service.TicketCreateConflictError) as captured:
            ticket_service.create_ticket(make_ticket_create(), confirmation_create=confirmation)
        assert captured.value.__cause__ is error
        assert transaction.__exit__.call_args.args[0] is IntegrityError
    else:
        ticket = ticket_service.create_ticket(make_ticket_create(), confirmation_create=confirmation)
        assert transaction.__exit__.call_args.args[0] is None
        assert ticket.ticket_id == session.add.call_args_list[0].args[0].ticket_id

    ticket_record, confirmation_record = [entry.args[0] for entry in session.add.call_args_list]
    assert isinstance(ticket_record, TicketTable)
    assert isinstance(confirmation_record, TicketConfirmationTable)
    assert confirmation_record.ticket_id == ticket_record.ticket_id
    assert confirmation_record.confirmed_at == confirmation.confirmed_at
    assert confirmation_record.confirmation_message == "确认"
    assert confirmation_record.application_snapshot == confirmation.application_snapshot
    session.begin.assert_called_once()
    assert session.flush.call_count == 2


def test_create_ticket_rejects_when_order_is_not_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    order_record = Mock()
    session = Mock()
    session.get.return_value = order_record
    session.begin.return_value = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    session_context.__exit__.return_value = None
    eligibility = ticket_service.TicketEligibilityDecision.REJECT

    monkeypatch.setattr(ticket_service, "Session", Mock(return_value=session_context))
    monkeypatch.setattr(ticket_service.Order, "model_validate", Mock(return_value=Mock()))
    monkeypatch.setattr(
        ticket_service,
        "check_order_ticket_eligibility",
        Mock(return_value=Mock(
            decision=eligibility,
            reason="订单未支付，暂不能创建售后工单。",
        )),
    )

    with pytest.raises(
        ticket_service.TicketEligibilityRejectedError,
        match="订单未支付，暂不能创建售后工单。",
    ):
        ticket_service.create_ticket(make_ticket_create())

    session.add.assert_not_called()
    session.flush.assert_not_called()


@pytest.mark.parametrize(
    "is_paid,status,expected_reason",
    [
        (False, OrderStatus.PENDING, "订单未支付"),
        (True, OrderStatus.CANCELED, "订单已取消"),
    ],
)
def test_create_ticket_enforces_order_rules_before_database_write(
    monkeypatch: pytest.MonkeyPatch,
    is_paid: bool,
    status: OrderStatus,
    expected_reason: str,
) -> None:
    order_record = SimpleNamespace(
        order_id="20260721001",
        product_name="蓝牙耳机",
        product_id="PRODUCT-001",
        price=399.0,
        quantity=1,
        is_paid=is_paid,
        status=status.value,
        tracking_number=None,
        customer_note=None,
    )
    session = Mock()
    session.get.return_value = order_record
    session.begin.return_value = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    monkeypatch.setattr(ticket_service, "Session", Mock(return_value=session_context))

    with pytest.raises(ticket_service.TicketEligibilityRejectedError, match=expected_reason):
        ticket_service.create_ticket(make_ticket_create())

    session.add.assert_not_called()
    session.flush.assert_not_called()


@pytest.mark.parametrize("special_review", [False, True])
def test_load_return_eligibility_input_uses_catalog_and_signed_date(
    monkeypatch: pytest.MonkeyPatch,
    special_review: bool,
) -> None:
    order_record = SimpleNamespace(
        order_id="20260721017",
        product_id="PRODUCT-017",
        product_name="桌面麦克风",
        price=459.0,
        quantity=1,
        is_paid=True,
        status="delivered",
        tracking_number=None,
        customer_note=None,
    )
    product_record = SimpleNamespace(
        product_id="PRODUCT-017",
        name="桌面麦克风",
        category="electronics",
        price=459.0,
        stock=16,
        is_returnable=True,
        requires_special_review=special_review,
    )
    logistics_record = SimpleNamespace(
        order_id="20260721017",
        tracking_number="JD100000017",
        carrier="京东物流",
        status="delivered",
        latest_event="快件已由本人签收",
        signed_at=date(2026, 7, 25),
    )
    session = Mock()
    session.get.side_effect = [order_record, product_record, logistics_record]
    session.begin.return_value = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    monkeypatch.setattr(ticket_service, "Session", Mock(return_value=session_context))

    facts = ticket_service.load_return_eligibility_input(
        order_id="20260721017",
        issue_description="不想要了",
        requested_action="退货",
        return_reason=ReturnReason.NO_REASON,
        requested_at=date(2026, 8, 3),
    )

    assert facts.product_category == "electronics"
    assert facts.requires_special_review is special_review
    result = check_return_eligibility(facts)
    if special_review:
        assert result.decision == TicketEligibilityDecision.NEED_MANUAL_REVIEW
        assert "专项人工审核" in result.reason
    else:
        assert result.decision == TicketEligibilityDecision.REJECT
    assert facts.received_at == date(2026, 7, 25)
    assert facts.is_opened is None
    assert facts.quality_inspection_passed is None
    assert session.get.call_args_list == [
        call(OrderTable, "20260721017"),
        call(ProductTable, "PRODUCT-017"),
        call(LogisticsTable, "20260721017"),
    ]


def test_load_return_eligibility_input_without_logistics_keeps_date_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_record = SimpleNamespace(
        order_id="20260721001",
        product_id="PRODUCT-001",
        product_name="蓝牙耳机",
        price=399.0,
        quantity=1,
        is_paid=True,
        status="paid",
        tracking_number=None,
        customer_note=None,
    )
    product_record = SimpleNamespace(
        product_id="PRODUCT-001",
        name="蓝牙耳机",
        category="in_ear_earphone",
        price=399.0,
        stock=100,
        is_returnable=True,
    )
    session = Mock()
    session.get.side_effect = [order_record, product_record, None]
    session.begin.return_value = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    monkeypatch.setattr(ticket_service, "Session", Mock(return_value=session_context))

    facts = ticket_service.load_return_eligibility_input(
        order_id="20260721001",
        issue_description="不想要了",
        requested_action="退货",
        return_reason=ReturnReason.NO_REASON,
        requested_at=date(2026, 8, 3),
    )

    assert facts.product_category == "in_ear_earphone"
    assert facts.received_at is None


def test_load_return_eligibility_input_in_transit_keeps_date_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order_record = SimpleNamespace(
        order_id="20260721001",
        product_id="PRODUCT-001",
        product_name="蓝牙耳机",
        price=399.0,
        quantity=1,
        is_paid=True,
        status="shipped",
        tracking_number=None,
        customer_note=None,
    )
    product_record = SimpleNamespace(
        product_id="PRODUCT-001",
        name="蓝牙耳机",
        category="in_ear_earphone",
        price=399.0,
        stock=100,
        is_returnable=True,
    )
    logistics_record = SimpleNamespace(
        order_id="20260721001",
        tracking_number="SF100000001",
        carrier="顺丰速运",
        status="in_transit",
        latest_event="运输中",
        signed_at=None,
    )
    session = Mock()
    session.get.side_effect = [order_record, product_record, logistics_record]
    session.begin.return_value = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = session
    monkeypatch.setattr(ticket_service, "Session", Mock(return_value=session_context))

    facts = ticket_service.load_return_eligibility_input(
        order_id="20260721001",
        issue_description="不想要了",
        requested_action="退货",
        return_reason=ReturnReason.NO_REASON,
        requested_at=date(2026, 8, 3),
    )

    assert facts.received_at is None
