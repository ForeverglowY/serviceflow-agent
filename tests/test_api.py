from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_existing_order() -> None:
    response = client.get("/orders/20260721001")

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "20260721001"
    assert data["product_name"] == "蓝牙耳机"
    assert data["price"] == 399.0


def test_get_unknown_order_returns_404() -> None:
    response = client.get("/orders/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "订单不存在"}


def test_get_existing_product() -> None:
    response = client.get("/products/PRODUCT-003")

    assert response.status_code == 200

    data = response.json()
    assert data["product_id"] == "PRODUCT-003"
    assert data["stock"] == 0
    assert data["is_returnable"] is False


def test_get_unknown_product_returns_404() -> None:
    response = client.get("/products/PRODUCT-999")

    assert response.status_code == 404
    assert response.json() == {"detail": "商品不存在"}


def test_get_existing_logistics() -> None:
    response = client.get("/logistics/20260721002")

    assert response.status_code == 200

    data = response.json()
    assert data["order_id"] == "20260721002"
    assert data["status"] == "pending"
    assert data["latest_event"] is None


def test_get_unknown_logistics_returns_404() -> None:
    response = client.get("/logistics/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "物流信息不存在"}


def test_preview_valid_ticket() -> None:
    ticket_data = {
        "ticket_id": "TICKET-API-001",
        "order_id": "20260721001",
        "user_id": "USER-001",
        "issue_type": "product_fault",
        "description": "蓝牙耳机左耳没有声音",
        "priority": "high",
    }

    response = client.post("/tickets/preview", json=ticket_data)

    assert response.status_code == 200

    data = response.json()
    assert data["ticket_id"] == "TICKET-API-001"
    assert data["priority"] == "high"
    assert data["status"] == "open"
    assert data["assigned_to"] is None


def test_preview_invalid_ticket_returns_422() -> None:
    ticket_data = {
        "ticket_id": "TICKET-API-002",
        "order_id": "20260721001",
        "user_id": "USER-001",
        "issue_type": "product_fault",
        "description": "",
        "priority": "critical",
    }

    response = client.post("/tickets/preview", json=ticket_data)

    assert response.status_code == 422

    errors = response.json()["detail"]
    error_locations = [error["loc"] for error in errors]

    assert ["body", "description"] in error_locations
    assert ["body", "priority"] in error_locations
