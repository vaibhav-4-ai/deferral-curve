import copy
import re

# A tiny fake order database. QuickKart is a simulated store, so the "backend" is simulated too.
# Never changed in place: every question works on its own copy (see fresh_orders), so cancelling an order
# in one question can't change the answer to the next one.
ORDERS = {
    "4821": {"status": "shipped", "eta": "2 days", "refund": "none", "cancellable": False},
    "5530": {"status": "processing", "eta": "5 days", "refund": "none", "cancellable": True},
    "6102": {"status": "returned", "eta": "-", "refund": "initiated, 5 to 7 business days", "cancellable": False},
}

ORDER_ID = re.compile(r"#?\b(\d{4,6})\b")


def fresh_orders():
    return copy.deepcopy(ORDERS)


def order_status(orders, order_id):
    order = orders.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."
    return f"Order {order_id} is {order['status']}. Expected delivery in {order['eta']}."


def refund_status(orders, order_id):
    order = orders.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."
    return f"Refund for order {order_id}: {order['refund']}."


def cancel_order(orders, order_id):
    order = orders.get(order_id)
    if order is None:
        return f"Order {order_id} was not found."
    if not order["cancellable"]:
        return f"Order {order_id} is {order['status']} and can no longer be cancelled."
    order["status"] = "cancelled"
    order["cancellable"] = False
    return f"Order {order_id} has been cancelled."


TOOLS = {
    "order_status": order_status,
    "refund_status": refund_status,
    "cancel_order": cancel_order,
}


def pick_tool(query):
    # Rule-based tool choice: needs an order number plus a recognisable intent word.
    match = ORDER_ID.search(query)
    if match is None:
        return None
    q = query.lower()
    if "cancel" in q:
        return "cancel_order", match.group(1)
    if "refund" in q:
        return "refund_status", match.group(1)
    if any(word in q for word in ("kahan", "where", "track", "status")):
        return "order_status", match.group(1)
    return None
