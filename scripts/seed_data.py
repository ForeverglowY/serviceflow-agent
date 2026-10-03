from sqlalchemy import select
from sqlalchemy.orm import Session

from serviceflow.data_loader import load_logistics, load_orders, load_products
from serviceflow.db.database import engine
from serviceflow.db.models import LogisticsTable, OrderTable, ProductTable


def seed_orders(session: Session) -> int:
    # 获取数据库中所有的订单id
    existing_order_ids: set[str] = set(
        session.scalars(
            select(OrderTable.order_id)
        ).all()
    )
    order_records: list[OrderTable] = []
    orders = load_orders("data/orders.json")

    for order in orders:
        if order.order_id in existing_order_ids:
            continue

        order_data = order.model_dump(mode="json")
        order_records.append(OrderTable(**order_data))
        existing_order_ids.add(order.order_id)

    session.add_all(order_records)
    return len(order_records)


def seed_products(session: Session) -> int:
    existing_product_ids: set[str] = set(
        session.scalars(
            select(ProductTable.product_id)
        ).all()
    )

    product_records: list[ProductTable] = []

    products = load_products("data/products.json")
    for product in products:
        if product.product_id in existing_product_ids:
            continue
        product_data = product.model_dump(mode="json")
        product_records.append(ProductTable(**product_data))
        existing_product_ids.add(product.product_id)

    session.add_all(product_records)
    return len(product_records)


def seed_logistics(session: Session) -> int:
    existing_logistics_ids: set[str] = set(
        session.scalars(
            select(LogisticsTable.order_id)
        ).all()
    )
    logistics_records: list[LogisticsTable] = []

    logistics_list = load_logistics("data/logistics.json")
    for logistics in logistics_list:
        if logistics.order_id in existing_logistics_ids:
            continue

        logistics_data = logistics.model_dump()
        logistics_records.append(LogisticsTable(**logistics_data))
        existing_logistics_ids.add(logistics.order_id)

    session.add_all(logistics_records)
    return len(logistics_records)


def main() -> None:
    with Session(engine) as session:
        with session.begin():
            product_count = seed_products(session)
            order_count = seed_orders(session)
            logistics_count = seed_logistics(session)

    print(f"新增商品：{product_count} 条")
    print(f"新增订单：{order_count} 条")
    print(f"新增物流：{logistics_count} 条")


if __name__ == "__main__":
    main()
