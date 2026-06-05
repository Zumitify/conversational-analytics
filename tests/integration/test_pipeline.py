"""End-to-end pipeline tests on the seeded DuckDB dataset with a mock LLM.

The mock provider is programmed with intents; everything downstream of the
LLM (validation, planning, SQL, execution, postprocessing, persistence) runs
for real.
"""

from __future__ import annotations

import pytest

from cae.exceptions import ClarificationNeeded
from cae.models import Filter, QueryIntent, TimeRange


def program(pipeline, question: str, intent: dict) -> None:
    pipeline.provider.program_intent(question, intent)


class TestEndToEnd:
    def test_simple_aggregate(self, pipeline):
        program(pipeline, "total revenue last month",
                {"metrics": ["revenue"], "time_range": {"relative": "last_month"}})
        session = pipeline.create_session()
        response = pipeline.ask(session, "total revenue last month")

        assert response.result.row_count == 1
        assert response.result.columns[0].name == "revenue"
        assert response.result.columns[0].role == "metric"
        assert response.result.rows[0][0] > 0
        assert response.chart_spec["chart_type"] == "kpi"
        assert "LIMIT" in response.sql

    def test_group_by_region_matches_hand_sql(self, pipeline, engine):
        program(pipeline, "revenue by region this year",
                {"metrics": ["revenue"], "dimensions": ["region"],
                 "time_range": {"relative": "ytd"}})
        session = pipeline.create_session()
        response = pipeline.ask(session, "revenue by region this year")

        assert response.result.row_count == 4
        assert response.chart_spec["chart_type"] == "bar"

        # Cross-check the top value against independent hand-written SQL.
        check = engine.execute(
            "SELECT SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS rev "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN order_items oi ON o.order_id = oi.order_id "
            "WHERE o.order_date >= DATE '2026-01-01' "
            "  AND o.order_date <= DATE '2026-06-04' "
            "GROUP BY c.region ORDER BY rev DESC LIMIT 1"
        )
        assert response.result.rows[0][1] == pytest.approx(check.rows[0][0])

    def test_weekly_trend_has_time_role(self, pipeline):
        program(pipeline, "weekly revenue trend this year",
                {"metrics": ["revenue"],
                 "time_range": {"grain": "week", "relative": "ytd"}})
        session = pipeline.create_session()
        response = pipeline.ask(session, "weekly revenue trend this year")

        roles = {c.name: c.role for c in response.result.columns}
        assert roles["period"] == "time"
        assert response.result.row_count >= 15
        assert response.chart_spec["chart_type"] == "line"

    def test_comparison_wow(self, pipeline):
        program(pipeline, "wow revenue last 12 weeks",
                {"metrics": ["revenue"], "comparison": "wow",
                 "time_range": {"relative": "last_12_weeks"}})
        session = pipeline.create_session()
        response = pipeline.ask(session, "wow revenue last 12 weeks")

        names = [c.name for c in response.result.columns]
        assert {"period", "revenue", "revenue_prev", "revenue_delta",
                "revenue_pct_change"} <= set(names)
        # first row has no previous period
        first = dict(zip(names, response.result.rows[0]))
        assert first["revenue_prev"] is None
        # subsequent rows have deltas that are consistent
        second = dict(zip(names, response.result.rows[1]))
        assert second["revenue_prev"] == pytest.approx(first["revenue"])
        assert second["revenue_delta"] == pytest.approx(
            second["revenue"] - first["revenue"]
        )

    def test_top_n_with_filter(self, pipeline):
        program(pipeline, "top 3 categories by units in the west",
                {"metrics": ["units_sold"], "dimensions": ["category"],
                 "filters": [{"dimension": "region", "op": "=", "values": ["West"]}],
                 "time_range": {"relative": "ytd"},
                 "sort": [{"field": "units_sold", "direction": "desc"}],
                 "limit": 3})
        session = pipeline.create_session()
        response = pipeline.ask(session, "top 3 categories by units in the west")

        assert response.result.row_count == 3
        values = [row[1] for row in response.result.rows]
        assert values == sorted(values, reverse=True)


class TestFollowUps:
    def test_active_intent_passed_as_context(self, pipeline):
        program(pipeline, "revenue by region this year",
                {"metrics": ["revenue"], "dimensions": ["region"],
                 "time_range": {"relative": "ytd"}})
        program(pipeline, "now just enterprise customers",
                {"metrics": ["revenue"], "dimensions": ["region"],
                 "filters": [{"dimension": "segment", "op": "=",
                              "values": ["Enterprise"]}],
                 "time_range": {"relative": "ytd"}})

        session = pipeline.create_session()
        first = pipeline.ask(session, "revenue by region this year")
        second = pipeline.ask(session, "now just enterprise customers")

        assert second.result.row_count == 4
        assert second.result.rows[0][1] < first.result.rows[0][1]

        # slot memory: active intent is the latest one
        stored = pipeline.store.get_session(session)
        assert stored.active_intent.filters[0].values == ["Enterprise"]
        assert len(stored.turns) == 2

    def test_edited_intent_rerun_without_llm(self, pipeline):
        session = pipeline.create_session()
        intent = QueryIntent(
            metrics=["revenue"],
            dimensions=["channel"],
            filters=[Filter(dimension="region", op="=", values=["midwest"])],
            time_range=TimeRange(relative="last_month"),
        )
        response = pipeline.ask_intent(session, intent)
        assert response.result.row_count == 3
        # enum value was canonicalized during validation
        assert response.intent.filters[0].values == ["Midwest"]


class TestFailureModes:
    def test_unknown_metric_clarification(self, pipeline):
        program(pipeline, "show me the profit margin",
                {"metrics": ["profit_margin"]})
        session = pipeline.create_session()
        with pytest.raises(ClarificationNeeded):
            pipeline.ask(session, "show me the profit margin")

    def test_unknown_session(self, pipeline):
        with pytest.raises(KeyError):
            pipeline.ask("ghost-session", "anything")

    def test_summary_passes_faithfulness_by_default(self, pipeline):
        program(pipeline, "total revenue last month",
                {"metrics": ["revenue"], "time_range": {"relative": "last_month"}})
        session = pipeline.create_session()
        response = pipeline.ask(session, "total revenue last month")
        assert not response.summary_dropped
        assert response.summary  # mock default summary

    def test_unfaithful_summary_dropped(self, pipeline):
        program(pipeline, "total revenue last month",
                {"metrics": ["revenue"], "time_range": {"relative": "last_month"}})
        pipeline.provider.program_text("Revenue hit 123456789.99 — a record!")
        session = pipeline.create_session()
        response = pipeline.ask(session, "total revenue last month")
        assert response.summary_dropped
        assert response.summary == ""
        assert any("faithfulness" in w for w in response.warnings)

    def test_timings_and_audit_recorded(self, pipeline):
        program(pipeline, "total revenue last month",
                {"metrics": ["revenue"], "time_range": {"relative": "last_month"}})
        session = pipeline.create_session()
        response = pipeline.ask(session, "total revenue last month")
        for stage_name in ("parse_intent", "plan", "generate_sql",
                           "validate_sql", "execute", "postprocess"):
            assert stage_name in response.stage_timings_ms
