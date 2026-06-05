"""QueryPlan -> SQL string. Deterministic Jinja rendering, no LLM.

The plan is already fully specified, so "translating" it with a model would
only reintroduce hallucination risk. Templates are boring and correct.

DuckDB and Postgres share the base template (both support DATE_TRUNC,
DATE '...' literals, and named WINDOW clauses); each dialect class owns its
hook points for divergence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from cae.models import QueryPlan

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class SQLGenerator(Protocol):
    dialect: str

    def render(self, plan: QueryPlan) -> str: ...


class BaseGenerator:
    dialect = "ansi"
    template_name = "select.sql.j2"

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(_TEMPLATE_DIR),
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )

    def _order_sql(self, plan: QueryPlan) -> str:
        return ", ".join(
            f"{clause.expr} {clause.direction.upper()}" for clause in plan.order_by
        )

    def render(self, plan: QueryPlan) -> str:
        template = self._env.get_template(self.template_name)
        sql = template.render(plan=plan, order_sql=self._order_sql(plan))
        # Normalize whitespace artifacts from template control flow.
        lines = [line.rstrip() for line in sql.splitlines() if line.strip()]
        return "\n".join(lines)


class DuckDBGenerator(BaseGenerator):
    dialect = "duckdb"


class PostgresGenerator(BaseGenerator):
    dialect = "postgres"


def make_generator(dialect: str) -> SQLGenerator:
    if dialect == "duckdb":
        return DuckDBGenerator()
    if dialect == "postgres":
        return PostgresGenerator()
    raise ValueError(f"unknown SQL dialect: {dialect}")
