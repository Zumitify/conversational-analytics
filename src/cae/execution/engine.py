"""Execution engines: run validated SQL and return a typed QueryResult.

Both engines enforce timeouts, cap fetched rows, and expose the live
database catalog so the validator can check every referenced table/column.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from cae.config import AppConfig
from cae.exceptions import ExecutionError
from cae.models import ColumnMeta, QueryResult


class ExecutionEngine(Protocol):
    dialect: str

    def execute(self, sql: str, timeout_s: int = 30, max_rows: int = 10_000) -> QueryResult: ...
    def catalog(self) -> dict[str, set[str]]: ...
    def close(self) -> None: ...


def _normalize(value: Any) -> Any:
    """Make DB values JSON-friendly and arithmetic-friendly."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


class DuckDBEngine:
    """In-process DuckDB engine. Opens the database file read-only."""

    dialect = "duckdb"

    def __init__(self, path: str, read_only: bool = True) -> None:
        import duckdb

        self._conn = duckdb.connect(path, read_only=read_only)
        self._lock = threading.Lock()

    def execute(self, sql: str, timeout_s: int = 30, max_rows: int = 10_000) -> QueryResult:
        result_holder: dict[str, Any] = {}

        def run() -> None:
            try:
                with self._lock:
                    cursor = self._conn.execute(sql)
                    rows = cursor.fetchmany(max_rows + 1)
                    result_holder["rows"] = rows
                    result_holder["description"] = cursor.description
            except Exception as exc:  # noqa: BLE001 — surfaced as ExecutionError
                result_holder["error"] = exc

        started = time.perf_counter()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            self._conn.interrupt()
            worker.join(5)
            raise ExecutionError(f"query timed out after {timeout_s}s")
        if "error" in result_holder:
            raise ExecutionError(str(result_holder["error"]))

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        raw_rows = result_holder.get("rows", [])
        truncated = len(raw_rows) > max_rows
        raw_rows = raw_rows[:max_rows]
        description = result_holder.get("description") or []
        columns = [
            ColumnMeta(name=d[0], sql_type=str(d[1]) if len(d) > 1 else "")
            for d in description
        ]
        rows = [[_normalize(v) for v in row] for row in raw_rows]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    def catalog(self) -> dict[str, set[str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns"
            ).fetchall()
        catalog: dict[str, set[str]] = {}
        for table, column in rows:
            catalog.setdefault(table.lower(), set()).add(column.lower())
        return catalog

    def close(self) -> None:
        self._conn.close()


class PostgresEngine:
    """Postgres engine via psycopg 3. Read-only, statement-timeout enforced."""

    dialect = "postgres"

    def __init__(self, dsn: str) -> None:
        import psycopg

        self._conn = psycopg.connect(dsn)
        self._conn.read_only = True
        self._conn.autocommit = True

    def execute(self, sql: str, timeout_s: int = 30, max_rows: int = 10_000) -> QueryResult:
        started = time.perf_counter()
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {timeout_s * 1000}")
                cur.execute(sql)
                raw_rows = cur.fetchmany(max_rows + 1)
                description = cur.description or []
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(str(exc)) from exc

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        truncated = len(raw_rows) > max_rows
        raw_rows = raw_rows[:max_rows]
        columns = [
            ColumnMeta(name=d.name, sql_type=str(d.type_code)) for d in description
        ]
        rows = [[_normalize(v) for v in row] for row in raw_rows]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    def estimate_rows(self, sql: str) -> int:
        """Planner row estimate via EXPLAIN — used as a pre-execution cost gate."""
        import json

        with self._conn.cursor() as cur:
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            plan = cur.fetchone()[0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        return int(plan[0]["Plan"]["Plan Rows"])

    def catalog(self) -> dict[str, set[str]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
            )
            rows = cur.fetchall()
        catalog: dict[str, set[str]] = {}
        for table, column in rows:
            catalog.setdefault(table.lower(), set()).add(column.lower())
        return catalog

    def close(self) -> None:
        self._conn.close()


def make_engine(config: AppConfig) -> ExecutionEngine:
    if config.database.dialect == "duckdb":
        return DuckDBEngine(config.database.path)
    if config.database.dialect == "postgres":
        if not config.database.postgres_dsn:
            raise ValueError("postgres dialect requires database.postgres_dsn")
        return PostgresEngine(config.database.postgres_dsn)
    raise ValueError(f"unknown dialect: {config.database.dialect}")
