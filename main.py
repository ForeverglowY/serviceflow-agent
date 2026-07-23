import json

from pydantic import ValidationError

from models import Order


class product:
    def __init__(self, name: str, price: float, quantity: int) -> None:
        self.name = name
        self.price = price
        self.quantity = quantity
    def total_price(self) -> float:
        return self.price * self.quantity

def find_order(order_id: str, orders: list[Order]) -> Order | None:
    for order in orders:
        if order.order_id == order_id:
            return order
    return None


def print_order(order: Order) -> None:
    print(f"订单号: {order.order_id}")
    print(f"商品: {order.product_name}")
    print(f"单价: {order.price}")
    print(f"数量: {order.quantity}")
    print(f"已支付: {order.payment_status()}")
    print(f"总价: {order.total_price()}")

def load_orders(file_path: str) -> list[Order]:
    orders: list[Order] = []
    with open(file_path, "r", encoding="utf-8") as file:
        datas = json.load(file)
        for data in datas:
            # 把当前字典转换成Order
            # 添加到 orders
            order: Order = Order.model_validate(data)
            orders.append(order)
    return orders

def main() -> None:
    input_order_id: str = input("请输入订单号: ").strip()
    if not input_order_id:
        print("订单号不能为空")
        return

    try:
        orders = load_orders("orders.json")
        # orders = load_orders("invalid_orders.json")
    except FileNotFoundError:
        print("订单文件不存在")
        return
    except json.JSONDecodeError:
        print("订单文件格式错误")
        return
    except ValidationError as error:
        print("订单数据校验失败")
        print(error)
        return

    found_order: Order | None = find_order(input_order_id, orders)
    if found_order is None:
        print("未找到订单")
    else:
        print_order(found_order)


if __name__ == "__main__":
    main()
