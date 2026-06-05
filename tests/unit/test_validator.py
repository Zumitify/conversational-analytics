"""Validator corpus: malicious / malformed vs valid SQL, with expected verdicts."""

from __future__ import annotations

import pytest

from cae.sql_validator import SQLValidator

GOOD_TIME_BOUND = "WHERE orders.order_date >= DATE '2026-01-01' AND orders.order_date <= DATE '2026-06-04'"


@pytest.fixture()
def validator(layer, engine):
    return SQLValidator(layer, catalog=engine.catalog(), max_rows=10_000)


def check(validator, sql, dialect="duckdb"):
    return validator.validate(sql, dialect=dialect)


class TestSafetyRejections:
    @pytest.mark.parametrize("sql", [
        "DROP TABLE orders",
        "DELETE FROM orders WHERE 1=1",
        "UPDATE orders SET status = 'x'",
        "INSERT INTO orders VALUES (1)",
        "CREATE TABLE evil (x INT)",
        "TRUNCATE TABLE orders",
    ])
    def test_ddl_dml_rejected(self, validator, sql):
        verdict = check(validator, sql)
        assert not verdict.ok

    def test_multiple_statements_rejected(self, validator):
        verdict = check(validator, "SELECT 1; SELECT 2")
        assert not verdict.ok
        assert "exactly one statement" in verdict.errors[0]

    def test_unknown_table_rejected(self, validator):
        verdict = check(validator, "SELECT * FROM secrets LIMIT 10")
        assert not verdict.ok
        assert any("allow-list" in e for e in verdict.errors)

    def test_schema_qualified_access_rejected(self, validator):
        verdict = check(
            validator, "SELECT * FROM information_schema.tables LIMIT 10"
        )
        assert not verdict.ok

    def test_parse_error_rejected(self, validator):
        verdict = check(validator, "SELEKT revenue FROM orders")
        assert not verdict.ok
        assert "parse error" in verdict.errors[0]

    def test_cartesian_join_rejected(self, validator):
        sql = (
            "SELECT COUNT(*) FROM orders AS orders "
            "JOIN customers AS customers ON 1 = 1 "
            f"{GOOD_TIME_BOUND} LIMIT 10"
        )
        verdict = check(validator, sql)
        assert not verdict.ok
        assert any("undeclared join" in e for e in verdict.errors)

    def test_missing_time_bound_rejected(self, validator):
        verdict = check(validator, "SELECT COUNT(*) FROM orders AS orders LIMIT 10")
        assert not verdict.ok
        assert any("time" in e.lower() for e in verdict.errors)

    def test_unknown_column_rejected(self, validator):
        sql = (
            "SELECT orders.password FROM orders AS orders "
            f"{GOOD_TIME_BOUND} LIMIT 10"
        )
        verdict = check(validator, sql)
        assert not verdict.ok
        assert any("column not in catalog" in e for e in verdict.errors)


class TestLimitHandling:
    def test_limit_injected_when_missing(self, validator):
        sql = f"SELECT COUNT(*) FROM orders AS orders {GOOD_TIME_BOUND}"
        verdict = check(validator, sql)
        assert verdict.ok
        assert verdict.rewritten_sql is not None
        assert "LIMIT 10000" in verdict.rewritten_sql.replace("\n", " ")

    def test_oversized_limit_capped(self, validator):
        sql = f"SELECT COUNT(*) FROM orders AS orders {GOOD_TIME_BOUND} LIMIT 999999"
        verdict = check(validator, sql)
        assert verdict.ok
        assert "LIMIT 10000" in verdict.rewritten_sql.replace("\n", " ")
        assert any("capped" in w for w in verdict.warnings)

    def test_reasonable_limit_untouched(self, validator):
        sql = f"SELECT COUNT(*) FROM orders AS orders {GOOD_TIME_BOUND} LIMIT 50"
        verdict = check(validator, sql)
        assert verdict.ok
        assert verdict.rewritten_sql is None


class TestValidQueries:
    def test_select_one_allowed(self, validator):
        verdict = check(validator, "SELECT 1")
        assert verdict.ok

    def test_declared_join_allowed(self, validator):
        sql = (
            "SELECT customers.region AS region, COUNT(DISTINCT orders.order_id) AS n "
            "FROM orders AS orders "
            "JOIN customers AS customers ON orders.customer_id = customers.customer_id "
            f"{GOOD_TIME_BOUND} "
            "GROUP BY customers.region LIMIT 100"
        )
        verdict = check(validator, sql)
        assert verdict.ok, verdict.errors

    def test_generated_comparison_query_passes(self, layer, engine, planner_sql):
        validator = SQLValidator(layer, catalog=engine.catalog())
        verdict = validator.validate(planner_sql, dialect="duckdb")
        assert verdict.ok, verdict.errors


@pytest.fixture()
def planner_sql(layer):
    """A full comparison query as the pipeline would generate it."""
    from datetime import date

    from cae.models import QueryIntent, TimeRange
    from cae.query_ir import Planner
    from cae.sql_generator import DuckDBGenerator

    plan = Planner(layer).plan(
        QueryIntent(
            metrics=["revenue"],
            dimensions=["region"],
            comparison="wow",
            time_range=TimeRange(grain="week", relative="last_12_weeks"),
        ),
        today=date(2026, 6, 4),
    )
    return DuckDBGenerator().render(plan)
