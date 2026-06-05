"""Synthetic e-commerce dataset generator.

Deterministic given (seed, n_orders, end_date): the same arguments always
produce byte-identical data, so eval results are reproducible. Works against
DuckDB (default) and Postgres (pass a psycopg connection).
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

DDL = """
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name          VARCHAR NOT NULL,
    email         VARCHAR NOT NULL,
    region        VARCHAR NOT NULL,
    segment       VARCHAR NOT NULL,
    created_at    DATE NOT NULL
);
CREATE TABLE suppliers (
    supplier_id   INTEGER PRIMARY KEY,
    name          VARCHAR NOT NULL,
    country       VARCHAR NOT NULL
);
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    name          VARCHAR NOT NULL,
    line          VARCHAR NOT NULL
);
CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    name          VARCHAR NOT NULL,
    line          VARCHAR NOT NULL,
    category_id   INTEGER NOT NULL,
    supplier_id   INTEGER NOT NULL,
    unit_cost     DOUBLE PRECISION NOT NULL
);
CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL,
    order_date    DATE NOT NULL,
    status        VARCHAR NOT NULL,
    channel       VARCHAR NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    product_id    INTEGER NOT NULL,
    quantity      INTEGER NOT NULL,
    unit_price    DOUBLE PRECISION NOT NULL,
    discount      DOUBLE PRECISION NOT NULL
);
CREATE TABLE payments (
    payment_id    INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    amount        DOUBLE PRECISION NOT NULL,
    method        VARCHAR NOT NULL,
    paid_at       DATE NOT NULL
);
CREATE TABLE shipments (
    shipment_id   INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    shipped_at    DATE NOT NULL,
    carrier       VARCHAR NOT NULL,
    shipping_cost DOUBLE PRECISION NOT NULL
);
"""

REGIONS = ["Northeast", "Southeast", "Midwest", "West"]
SEGMENTS = ["Consumer", "Corporate", "Enterprise"]
CHANNELS = ["web", "mobile", "store"]
STATUSES = ["completed", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.62, 0.18, 0.08, 0.07, 0.05]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "gift_card"]
CARRIERS = ["UPS", "FedEx", "USPS", "DHL"]
COUNTRIES = ["USA", "China", "Germany", "Vietnam", "Mexico", "India"]

LINES: dict[str, list[str]] = {
    "Electronics": ["Phones", "Laptops", "Audio", "Wearables"],
    "Home": ["Kitchen", "Furniture", "Decor"],
    "Outdoors": ["Camping", "Cycling", "Fitness"],
    "Apparel": ["Mens", "Womens", "Kids"],
    "Beauty": ["Skincare", "Fragrance"],
}

FIRST_NAMES = ["Ava", "Liam", "Maya", "Noah", "Zoe", "Ethan", "Iris", "Owen",
               "Ruby", "Caleb", "Nina", "Felix", "Lena", "Hugo", "Tara", "Dev"]
LAST_NAMES = ["Patel", "Nguyen", "Garcia", "Smith", "Chen", "Okafor", "Kim",
              "Rossi", "Khan", "Brown", "Sato", "Muller", "Lopez", "Singh"]
PRODUCT_ADJ = ["Pro", "Lite", "Max", "Mini", "Ultra", "Eco", "Prime", "Classic"]


def seed_database(
    conn,
    *,
    n_orders: int = 50_000,
    n_customers: int = 5_000,
    n_products: int = 400,
    seed: int = 42,
    end_date: date | None = None,
    span_days: int = 730,
) -> dict[str, int]:
    """Create the schema and load synthetic rows into an open DBAPI connection.

    Returns row counts per table.
    """
    rng = random.Random(seed)
    end = end_date or date.today()
    start = end - timedelta(days=span_days)

    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(statement)

    # -- customers ----------------------------------------------------------
    customers = []
    for cid in range(1, n_customers + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        customers.append((
            cid,
            f"{first} {last}",
            f"{first.lower()}.{last.lower()}{cid}@example.com",
            rng.choice(REGIONS),
            rng.choices(SEGMENTS, weights=[0.6, 0.25, 0.15])[0],
            start - timedelta(days=rng.randint(0, 365)),
        ))

    # -- suppliers / categories / products -----------------------------------
    suppliers = [
        (sid, f"Supplier {sid:03d}", rng.choice(COUNTRIES))
        for sid in range(1, 41)
    ]
    categories = []
    cat_id = 0
    cat_by_line: dict[str, list[int]] = {}
    for line, cats in LINES.items():
        cat_by_line[line] = []
        for cat in cats:
            cat_id += 1
            categories.append((cat_id, cat, line))
            cat_by_line[line].append(cat_id)

    products = []
    line_names = list(LINES)
    for pid in range(1, n_products + 1):
        line = rng.choices(line_names, weights=[0.3, 0.25, 0.15, 0.2, 0.1])[0]
        base = {"Electronics": 220, "Home": 90, "Outdoors": 70,
                "Apparel": 40, "Beauty": 30}[line]
        cost = round(rng.uniform(0.3, 1.8) * base, 2)
        products.append((
            pid,
            f"{line} {rng.choice(PRODUCT_ADJ)} {pid:04d}",
            line,
            rng.choice(cat_by_line[line]),
            rng.choice(suppliers)[0],
            cost,
        ))

    # -- orders / order_items / payments / shipments -------------------------
    orders, items, payments, shipments = [], [], [], []
    item_id = 0
    for oid in range(1, n_orders + 1):
        # Mild seasonality: weight later days a bit heavier (growth trend).
        day_offset = int(span_days * (rng.random() ** 0.85))
        order_date = start + timedelta(days=day_offset)
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        orders.append((
            oid,
            rng.randint(1, n_customers),
            order_date,
            status,
            rng.choices(CHANNELS, weights=[0.5, 0.35, 0.15])[0],
        ))

        order_total = 0.0
        for _ in range(rng.choices([1, 2, 3, 4], weights=[0.45, 0.3, 0.17, 0.08])[0]):
            item_id += 1
            product = rng.choice(products)
            quantity = rng.choices([1, 2, 3, 5], weights=[0.6, 0.25, 0.1, 0.05])[0]
            unit_price = round(product[5] * rng.uniform(1.25, 1.9), 2)
            discount = rng.choices([0.0, 0.05, 0.1, 0.2], weights=[0.7, 0.12, 0.12, 0.06])[0]
            items.append((item_id, oid, product[0], quantity, unit_price, discount))
            order_total += quantity * unit_price * (1 - discount)

        if status != "cancelled":
            payments.append((
                len(payments) + 1, oid, round(order_total, 2),
                rng.choices(PAYMENT_METHODS, weights=[0.55, 0.2, 0.18, 0.07])[0],
                order_date + timedelta(days=rng.randint(0, 2)),
            ))
        if status in ("completed", "shipped", "returned"):
            shipments.append((
                len(shipments) + 1, oid,
                order_date + timedelta(days=rng.randint(1, 5)),
                rng.choice(CARRIERS),
                round(rng.uniform(3.5, 24.0), 2),
            ))

    _bulk_insert(conn, "customers", customers, 6)
    _bulk_insert(conn, "suppliers", suppliers, 3)
    _bulk_insert(conn, "categories", categories, 3)
    _bulk_insert(conn, "products", products, 6)
    _bulk_insert(conn, "orders", orders, 5)
    _bulk_insert(conn, "order_items", items, 6)
    _bulk_insert(conn, "payments", payments, 5)
    _bulk_insert(conn, "shipments", shipments, 5)
    if hasattr(conn, "commit"):
        conn.commit()

    return {
        "customers": len(customers), "suppliers": len(suppliers),
        "categories": len(categories), "products": len(products),
        "orders": len(orders), "order_items": len(items),
        "payments": len(payments), "shipments": len(shipments),
    }


def _bulk_insert(conn, table: str, rows: list[tuple], width: int) -> None:
    placeholders = ", ".join(["?"] * width)
    sql = f"INSERT INTO {table} VALUES ({placeholders})"
    try:
        conn.executemany(sql, rows)
    except Exception:
        # psycopg uses %s placeholders and cursor-level executemany
        sql_pg = f"INSERT INTO {table} VALUES ({', '.join(['%s'] * width)})"
        with conn.cursor() as cur:
            cur.executemany(sql_pg, rows)


def seed_duckdb(path: str | Path, **kwargs) -> dict[str, int]:
    """Create (or overwrite) a DuckDB file with the synthetic dataset."""
    import duckdb

    db_path = Path(path)
    if db_path.exists():
        db_path.unlink()
    conn = duckdb.connect(str(db_path))
    try:
        return seed_database(conn, **kwargs)
    finally:
        conn.close()
