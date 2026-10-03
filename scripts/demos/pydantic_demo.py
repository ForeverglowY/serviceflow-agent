from serviceflow.models import Order
from pydantic import ValidationError


def main() -> None:
    order = Order(order_id="123", product_name="蓝牙耳机", price="399.2", quantity="10", is_paid=True)
    print(order.order_id)
    print(type(order.order_id))
    print(order.price)
    print(type(order.price))
    print(order.quantity)
    print(type(order.quantity))
    print(order.total_price())
    print(order.payment_status())
    print(order.model_dump())
    print(order)

if __name__ == '__main__':
    main()
