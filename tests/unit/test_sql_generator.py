"""Unit tests: plan -> SQL rendering. SQL is compared by AST (sqlglot), not
string equality, and every rendered query must parse in its target dialect."""

from __future__ import annotations

from datetime import date

import pytest
import sqlglot

from cae.models import Filter, QueryIntent, TimeRange
from cae.query_ir import Planner
from cae.sql_generator import DuckDBGenerator, PostgresGenerator, make_generator

TODAY = date(2026, 6, 4)


@pytest.fixture()
def planner(layer):
    return Planner(layer)


def normalize(sql: str, dialect: str) -> str:
    return sqlglot.parse_one(sql, read=dialect).sql(dialect=dialect, normalize=True)


class TestRendering:
    def test_simple_select_parses_in_both_dialects(self, planner):
        plan = planner.plan(
            QueryIntent(metrics=["revenue"], dimensions=["region"]), today=TODAY
        )
        for generator, dialect in ((DuckDBGenerator(), "duckdb"),
                                   (PostgresGenerator(), "postgres")):
            sql = generator.render(plan)
            tree = sqlglot.parse_one(sql, read=dialect)
            assert tree is not None

    def test_expected_structure(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                dimensions=["region"],
                time_range=TimeRange(relative="last_month"),
            ),
            today=TODAY,
        )
        sql = DuckDBGenerator().render(plan)
        tree = sqlglot.parse_one(sql, read="duckdb")
        tables = {t.name for t in tree.find_all(sqlglot.exp.Table)}
        assert tables == {"orders", "customers", "order_items"}
        assert tree.args.get("group") is not None
        assert tree.args.get("limit") is not None

    def test_ast_equivalence_to_handwritten(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["orders_count"],
                filters=[Filter(dimension="channel", op="=", values=["web"])],
                time_range=TimeRange(relative="q1_2026"),
            ),
            today=TODAY,
        )
        rendered = DuckDBGenerator().render(plan)
        expected = """
            SELECT COUNT(DISTINCT orders.order_id) AS orders_count
            FROM orders AS orders
            WHERE orders.order_date >= DATE '2026-01-01'
              AND orders.order_date <= DATE '2026-03-31'
              AND orders.channel = 'web'
            ORDER BY orders_count DESC
            LIMIT 1000
        """
        assert normalize(rendered, "duckdb") == normalize(expected, "duckdb")

    def test_join_conditions_rendered_verbatim(self, planner):
        plan = planner.plan(
            QueryIntent(metrics=["revenue"], dimensions=["category"]), today=TODAY
        )
        sql = DuckDBGenerator().render(plan)
        assert "JOIN order_items AS order_items ON orders.order_id = order_items.order_id" in sql
        assert "JOIN categories AS categories ON products.category_id = categories.category_id" in sql


class TestComparisonRendering:
    @pytest.fixture()
    def comparison_sql(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                dimensions=["region"],
                comparison="wow",
                time_range=TimeRange(grain="week", relative="last_12_weeks"),
            ),
            today=TODAY,
        )
        return DuckDBGenerator().render(plan)

    def test_cte_wrapper(self, comparison_sql):
        assert comparison_sql.startswith("WITH base AS (")

    def test_lag_window_columns(self, comparison_sql):
        assert "LAG(revenue) OVER w AS revenue_prev" in comparison_sql
        assert "revenue - LAG(revenue) OVER w AS revenue_delta" in comparison_sql
        assert "revenue_pct_change" in comparison_sql

    def test_window_partitioned_by_dimension(self, comparison_sql):
        assert "WINDOW w AS (PARTITION BY region ORDER BY period)" in comparison_sql

    def test_comparison_parses_in_both_dialects(self, comparison_sql):
        for dialect in ("duckdb", "postgres"):
            assert sqlglot.parse_one(comparison_sql, read=dialect) is not None


class TestFactory:
    def test_make_generator(self):
        assert make_generator("duckdb").dialect == "duckdb"
        assert make_generator("postgres").dialect == "postgres"
        with pytest.raises(ValueError):
            make_generator("oracle")
