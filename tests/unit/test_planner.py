"""Unit tests for the deterministic planner — the keystone of correctness."""

from __future__ import annotations

from datetime import date

import pytest

from cae.exceptions import PlanningError
from cae.models import Filter, QueryIntent, SortSpec, TimeRange
from cae.query_ir import Planner

TODAY = date(2026, 6, 4)


@pytest.fixture()
def planner(layer) -> Planner:
    return Planner(layer, default_limit=1000, lookback_years=5)


class TestSelectShape:
    def test_simple_aggregate(self, planner):
        plan = planner.plan(QueryIntent(metrics=["revenue"]), today=TODAY)
        assert [s.alias for s in plan.select] == ["revenue"]
        assert plan.select[0].role == "metric"
        assert plan.group_by == []
        assert plan.comparison is None

    def test_dimension_added_to_group_by(self, planner, layer):
        plan = planner.plan(
            QueryIntent(metrics=["revenue"], dimensions=["region"]), today=TODAY
        )
        aliases = [s.alias for s in plan.select]
        assert aliases == ["region", "revenue"]
        assert layer.resolve_dimension("region").expr in plan.group_by

    def test_time_grain_adds_period_first(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                time_range=TimeRange(grain="week", relative="ytd"),
            ),
            today=TODAY,
        )
        assert plan.select[0].alias == "period"
        assert plan.select[0].role == "time"
        assert "DATE_TRUNC('week'" in plan.select[0].expr
        assert plan.order_by[0].expr == "period"


class TestDeterminism:
    def test_same_intent_same_plan(self, planner):
        intent = QueryIntent(
            metrics=["revenue", "units_sold"],
            dimensions=["region", "product_line"],
            filters=[Filter(dimension="channel", op="=", values=["web"])],
            time_range=TimeRange(grain="month", relative="ytd"),
        )
        plan_a = planner.plan(intent, today=TODAY)
        plan_b = planner.plan(intent, today=TODAY)
        assert plan_a.model_dump() == plan_b.model_dump()


class TestTimeBound:
    def test_mandatory_time_bound_when_no_range(self, planner):
        plan = planner.plan(QueryIntent(metrics=["revenue"]), today=TODAY)
        bound = plan.where[0].expr
        assert "orders.order_date >= DATE '2021-06-04'" in bound
        assert "orders.order_date <= DATE '2026-06-04'" in bound

    def test_relative_range_resolved(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"], time_range=TimeRange(relative="last_month")
            ),
            today=TODAY,
        )
        assert "DATE '2026-05-01'" in plan.where[0].expr
        assert "DATE '2026-05-31'" in plan.where[0].expr


class TestFilters:
    def test_equality_filter(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                filters=[Filter(dimension="region", op="=", values=["Northeast"])],
            ),
            today=TODAY,
        )
        assert "customers.region = 'Northeast'" in plan.where[1].expr

    def test_multi_value_equality_promoted_to_in(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                filters=[Filter(dimension="region", op="=", values=["West", "Midwest"])],
            ),
            today=TODAY,
        )
        assert "customers.region IN ('West', 'Midwest')" in plan.where[1].expr

    def test_string_values_escaped(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["units_sold"],
                filters=[Filter(dimension="category", op="=", values=["Kid's"])],
            ),
            today=TODAY,
        )
        assert "'Kid''s'" in plan.where[1].expr

    def test_between_requires_two_values(self, planner):
        with pytest.raises(PlanningError):
            planner.plan(
                QueryIntent(
                    metrics=["revenue"],
                    filters=[Filter(dimension="region", op="between", values=["A"])],
                ),
                today=TODAY,
            )

    def test_filter_only_dimension_still_joined(self, planner):
        """Filtering on segment must pull in the customers join even when
        segment isn't in the group-by."""
        plan = planner.plan(
            QueryIntent(
                metrics=["orders_count"],
                filters=[Filter(dimension="segment", op="=", values=["Enterprise"])],
            ),
            today=TODAY,
        )
        assert "customers" in [j.table for j in plan.joins]


class TestComparisonDesugaring:
    def test_wow_produces_comparison_spec(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                comparison="wow",
                time_range=TimeRange(grain="week", relative="last_12_weeks"),
            ),
            today=TODAY,
        )
        assert plan.comparison is not None
        assert plan.comparison.kind == "wow"
        assert plan.comparison.value_aliases == ["revenue"]
        assert plan.comparison.partition_by == []

    def test_comparison_partitions_by_dimensions(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["revenue"],
                dimensions=["region"],
                comparison="mom",
                time_range=TimeRange(grain="month", relative="ytd"),
            ),
            today=TODAY,
        )
        assert plan.comparison.partition_by == ["region"]
        # comparison ordering: dims first, then period
        assert [o.expr for o in plan.order_by] == ["region", "period"]


class TestOrderAndLimit:
    def test_default_limit_injected(self, planner):
        plan = planner.plan(QueryIntent(metrics=["revenue"]), today=TODAY)
        assert plan.limit == 1000

    def test_explicit_sort_and_limit(self, planner):
        plan = planner.plan(
            QueryIntent(
                metrics=["units_sold"],
                dimensions=["category"],
                sort=[SortSpec(field="units_sold", direction="desc")],
                limit=5,
            ),
            today=TODAY,
        )
        assert plan.limit == 5
        assert plan.order_by[0].expr == "units_sold"
        assert plan.order_by[0].direction == "desc"

    def test_default_order_first_metric_desc(self, planner):
        plan = planner.plan(
            QueryIntent(metrics=["revenue"], dimensions=["region"]), today=TODAY
        )
        assert plan.order_by[0].expr == "revenue"
        assert plan.order_by[0].direction == "desc"
