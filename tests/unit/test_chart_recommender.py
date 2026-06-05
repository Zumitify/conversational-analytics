"""Decision-table tests for the rule-based chart recommender."""

from __future__ import annotations

from cae.models import ColumnMeta, QueryIntent, QueryResult
from cae.postprocessing import recommend_chart


def result(columns: list[tuple[str, str]], rows: list[list]) -> QueryResult:
    return QueryResult(
        columns=[ColumnMeta(name=n, role=r) for n, r in columns],
        rows=rows,
        row_count=len(rows),
        elapsed_ms=1,
    )


class TestShapes:
    def test_single_value_is_kpi(self):
        r = result([("revenue", "metric")], [[1234.5]])
        spec = recommend_chart(r)
        assert spec["chart_type"] == "kpi"
        assert spec["value"] == 1234.5

    def test_metric_over_time_is_line(self):
        r = result(
            [("period", "time"), ("revenue", "metric")],
            [["2026-01-01", 10], ["2026-02-01", 12]],
        )
        assert recommend_chart(r)["chart_type"] == "line"

    def test_metric_by_category_is_bar(self):
        r = result(
            [("region", "dimension"), ("revenue", "metric")],
            [["West", 10], ["Midwest", 8]],
        )
        spec = recommend_chart(r)
        assert spec["chart_type"] == "bar"
        assert spec["encoding"]["x"]["field"] == "region"

    def test_high_cardinality_becomes_top_n(self):
        rows = [[f"cat{i}", float(i)] for i in range(30)]
        r = result([("category", "dimension"), ("revenue", "metric")], rows)
        spec = recommend_chart(r)
        assert spec["chart_type"] == "bar_top_n"
        values = spec["data"]["values"]
        assert len(values) == 21  # top 20 + "(other)"
        assert values[-1]["category"] == "(other)"

    def test_time_plus_category_is_multi_line(self):
        r = result(
            [("period", "time"), ("region", "dimension"), ("revenue", "metric")],
            [["2026-01-01", "West", 1]],
        )
        spec = recommend_chart(r)
        assert spec["chart_type"] == "multi_line"
        assert spec["encoding"]["color"]["field"] == "region"

    def test_two_categoricals_is_heatmap(self):
        r = result(
            [("region", "dimension"), ("segment", "dimension"), ("revenue", "metric")],
            [["West", "Consumer", 5]],
        )
        assert recommend_chart(r)["chart_type"] == "heatmap"

    def test_two_metrics_one_dim_is_grouped_bar(self):
        r = result(
            [("channel", "dimension"), ("revenue", "metric"), ("units_sold", "metric")],
            [["web", 10, 3]],
        )
        assert recommend_chart(r)["chart_type"] == "grouped_bar"

    def test_comparison_is_line_with_delta_tooltip(self):
        r = result(
            [
                ("period", "time"),
                ("revenue", "metric"),
                ("revenue_prev", "metric"),
                ("revenue_delta", "metric"),
                ("revenue_pct_change", "metric"),
            ],
            [["2026-01-05", 10, None, None, None]],
        )
        intent = QueryIntent(metrics=["revenue"], comparison="wow")
        spec = recommend_chart(r, intent)
        assert spec["chart_type"] == "comparison_line"
        tooltip_fields = [t["field"] for t in spec["encoding"]["tooltip"]]
        assert "revenue_pct_change" in tooltip_fields

    def test_no_metrics_falls_back_to_table(self):
        r = result([("region", "dimension")], [["West"]])
        assert recommend_chart(r)["chart_type"] == "table"
