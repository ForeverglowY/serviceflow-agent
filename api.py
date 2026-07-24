from fastapi import FastAPI, HTTPException, status

from main import find_logistics, find_order, find_product, load_logistics, load_orders, load_products
from models import Logistics, Order, Product, Ticket

app = FastAPI(title="智能客服与工单处理 Agent")


@app.get("/health", tags=["系统"], summary="服务健康检查", )
def health() -> dict:
    return {"status": "ok"}


@app.get("/orders/{order_id}", response_model=Order, tags=["订单"], summary="根据订单号查询订单", )
def get_order(order_id: str) -> Order:
    orders: list[Order] = load_orders("orders.json")
    order = find_order(order_id, orders)
    if order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="订单不存在", )

    return order


@app.get("/products/{product_id}", response_model=Product, tags=["商品"], summary="根据商品编号查询商品", )
def get_product(product_id: str) -> Product:
    products: list[Product] = load_products("products.json")
    product: Product | None = find_product(product_id, products)
    if product is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="商品不存在", )
    return product


@app.get("/logistics/{order_id}", response_model=Logistics, tags=["物流"], summary="根据订单号查询物流", )
def get_logistic(order_id: str) -> Logistics:
    logistics_list: list[Logistics] = load_logistics("logistics.json")
    logistics_info: Logistics | None = find_logistics(order_id, logistics_list)
    if logistics_info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="物流信息不存在", )
    return logistics_info


@app.post("/tickets/preview", response_model=Ticket, tags=["工单"], summary="校验并预览工单", )
def preview_ticket(ticket: Ticket) -> Ticket:
    return ticket
