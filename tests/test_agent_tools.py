import json
from unittest.mock import Mock

import pytest

from serviceflow.agent import tools as agent_tools
from serviceflow.agent.tools import ToolNotFoundError, execute_tool
from serviceflow.llm.models import GetLogisticsResult, GetOrderResult


def test_execute_order_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_order_mock = Mock(
        return_value=GetOrderResult(
            order_id="20260721001",
            product_name="蓝牙耳机",
            unit_price=399.0,
            quantity=2,
            total_price=798.0,
            payment_status="已支付",
            order_status="待处理",
            tracking_number=None,
        )
    )
    monkeypatch.setattr(agent_tools, "get_order", get_order_mock)

    content = execute_tool(
        "get_order",
        '{"order_id":"20260721001"}',
    )

    data = json.loads(content)
    assert data["order_id"] == "20260721001"
    assert data["total_price"] == 798.0
    assert data["order_status"] == "待处理"
    assert get_order_mock.call_args.args[0].order_id == "20260721001"


def test_execute_logistics_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_logistics_mock = Mock(
        return_value=GetLogisticsResult(
            order_id="20260721001",
            tracking_number="SF100000001",
            carrier="顺丰速运",
            logistics_status="运输中",
            latest_event="快件已到达上海转运中心",
        )
    )
    monkeypatch.setattr(
        agent_tools,
        "get_logistics",
        get_logistics_mock,
    )

    content = execute_tool(
        "get_logistics",
        '{"order_id":"20260721001"}',
    )

    data = json.loads(content)
    assert data["carrier"] == "顺丰速运"
    assert data["logistics_status"] == "运输中"
    assert data["latest_event"] == "快件已到达上海转运中心"
    assert get_logistics_mock.call_args.args[0].order_id == (
        "20260721001"
    )


def test_execute_tool_serializes_missing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_tools,
        "get_logistics",
        Mock(return_value=None),
    )

    content = execute_tool(
        "get_logistics",
        '{"order_id":"NOT-FOUND"}',
    )

    assert json.loads(content) == {
        "error": "未查询到相关数据",
    }


def test_execute_tool_rejects_unknown_name() -> None:
    with pytest.raises(
        ToolNotFoundError,
        match="未注册的工具：delete_order",
    ):
        execute_tool("delete_order", "{}")
