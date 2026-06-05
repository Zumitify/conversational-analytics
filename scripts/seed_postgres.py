"""Load the same deterministic synthetic dataset into Postgres.

Usage:
    docker compose up -d
    pip install -e ".[postgres]"
    python scripts/seed_postgres.py [--dsn postgresql://cae:cae@localhost:5433/cae]
"""

from __future__ import annotations

import argparse

import psycopg

from cae.data.seed import seed_database

TABLES = [
    "shipments", "payments", "order_items", "orders",
    "products", "categories", "suppliers", "customers",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsn", default="postgresql://cae:cae@localhost:5433/cae"
    )
    parser.add_argument("--orders", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            for table in TABLES:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        conn.commit()
        counts = seed_database(conn, n_orders=args.orders, seed=args.seed)

    print("seeded postgres:")
    for table, count in counts.items():
        print(f"  {table:<12} {count:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
