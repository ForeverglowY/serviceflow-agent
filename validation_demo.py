from pydantic import ValidationError

from models import Order, OrderStatus

order_right = {
    "order_id": "1",
    "product_name": "蓝牙耳机",
    "price": "399",
    "quantity": 4,
    "is_paid": True,
}

order_quantity = {
    "order_id": "2",
    "product_name": "不知道",
    "price": 399,
    "is_paid": True,
}

order_error_type = {
    "order_id": "3",
    "product_name": "键盘",
    "price": "未知价格",
    "quantity": 4,
    "is_paid": True,
}

order_zero_price = {
    "order_id": "4",
    "product_name": "鼠标",
    "price": 0,
    "quantity": 1,
    "is_paid": True,
}

order_zero_quantity = {
    "order_id": "5",
    "product_name": "键盘",
    "price": 599.0,
    "quantity": 0,
    "is_paid": False,
}

order_empty_id = {
    "order_id": "",
    "product_name": "显示器",
    "price": 1299.0,
    "quantity": 1,
    "is_paid": True,
}

order_blank_id = {
    "order_id": "   ",
    "product_name": "数据线",
    "price": 29.9,
    "quantity": 1,
    "is_paid": True,
}

order_paid = {
    "order_id": "6",
    "product_name": "键盘",
    "price": 599.0,
    "quantity": 2,
    "is_paid": False,
    "status": "paid",
}
order_unknown = {
    "order_id": "6",
    "product_name": "键盘",
    "price": 599.0,
    "quantity": 2,
    "is_paid": False,
    "status": "unknown",
}

order_without_tracking = {
    "order_id": "7",
    "product_name": "无线鼠标",
    "price": 199.0,
    "quantity": 1,
    "is_paid": True,
    "status": "paid",
}

order_tracking_none = {
    "order_id": "8",
    "product_name": "机械键盘",
    "price": 599.0,
    "quantity": 1,
    "is_paid": True,
    "status": "paid",
    "tracking_number": None,
    "customer_note": None,
}

order_with_tracking = {
    "order_id": "9",
    "product_name": "显示器",
    "price": 1299.0,
    "quantity": 1,
    "is_paid": True,
    "status": "shipped",
    "tracking_number": "SF123456789",
    "customer_note": "工作日送货，送达前请电话联系",
}

order_empty_tracking = {
    "order_id": "10",
    "product_name": "数据线",
    "price": 29.9,
    "quantity": 2,
    "is_paid": True,
    "status": "shipped",
    "tracking_number": "",
    "customer_note": "请尽快发货",
}


def validate_order(order_data: dict) -> None:
    try:
        order = Order.model_validate(order_data)
    except ValidationError as e:
        for error in e.errors():
            print("loc=", error["loc"])
            print("msg=", error["msg"])
            print("type=", error["type"])
            print()
    else:
        print(order)
        print(type(order.status))
        print(order.status.value)
        print()

def main() -> None:
    validate_order(order_without_tracking)
    validate_order(order_tracking_none)
    validate_order(order_with_tracking)
    validate_order(order_empty_tracking)


if __name__ == "__main__":
    main()
