"""从本地 JSON 文件加载演示和初始化数据。"""

import json

from serviceflow.models import Logistics, Order, Product


def load_orders(file_path: str) -> list[Order]:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_orders = json.load(file)

    return [Order.model_validate(data) for data in raw_orders]


def load_products(file_path: str) -> list[Product]:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_products = json.load(file)

    return [Product.model_validate(data) for data in raw_products]


def load_logistics(file_path: str) -> list[Logistics]:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_logistics = json.load(file)

    return [Logistics.model_validate(data) for data in raw_logistics]
