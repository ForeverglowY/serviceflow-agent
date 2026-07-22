class Order:
    def __init__(self, order_id: str, product_name: str, price: float, quantity: int, is_paid: bool) -> None:
        self.order_id = order_id
        self.product_name = product_name
        self.price = price
        self.quantity = quantity
        self.is_paid = is_paid

    def total_price(self) -> float:
        return self.price * self.quantity

    def payment_status(self) -> str:
        return "是" if self.is_paid else "否"


def main() -> None:
    order1 = Order("20260721001", "蓝牙耳机", 399.0, 2, True)
    order2 = Order("20260721002", "机械键盘", 599.0, 1, False)

    # 蓝牙耳机，总价：798.0，已支付：是
    # 机械键盘，总价：599.0，已支付：否
    print(f"{order1.product_name}, 总价: {order1.total_price()}, 已支付: {order1.payment_status()}")
    print(f"{order2.product_name}, 总价: {order2.total_price()}, 已支付: {order2.payment_status()}")


if __name__ == "__main__":
    main()
