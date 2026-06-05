"""Rule-based chart recommender. No LLM — a decision tree over result shape.

| Shape                                         | Chart                        |
|-----------------------------------------------|------------------------------|
| 1 metric, 0 dims, 1 row                       | KPI card                     |
| 1 metric, 1 time dim                          | Line chart                   |
| 1 metric, 1 categorical dim (<=20 categories) | Bar chart                    |
| 1 metric, 1 categorical dim (>20 categories)  | Bar chart, top-N + "other"   |
| 1 metric, time + categorical                  | Multi-line chart             |
| 1 metric, 2 categorical dims                  | Heatmap                      |
| 2+ metrics, 1 dim                             | Grouped bar / multi-line     |
| Comparison query (wow/mom/yoy)                | Line with delta tooltip      |

Returns a Vega-Lite spec so any frontend can render it without extra logic.
"""

from __future__ import annotations

from typing import Any

from cae.models import QueryIntent, QueryResult

TOP_N = 20


def _values(result: QueryResult) -> list[dict[str, Any]]:
    names = [c.name for c in result.columns]
    return [dict(zip(names, row)) for row in result.rows]


def _base_spec(result: QueryResult, mark: str | dict) -> dict[str, Any]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": _values(result)},
        "mark": mark,
        "width": "container",
    }


def recommend_chart(result: QueryResult, intent: QueryIntent | None = None) -> dict[str, Any]:
    metrics = [c.name for c in result.columns if c.role == "metric"]
    dims = [c.name for c in result.columns if c.role == "dimension"]
    time_cols = [c.name for c in result.columns if c.role == "time"]
    # Comparison helper columns (_prev/_delta/_pct_change) shouldn't drive shape.
    base_metrics = [
        m for m in metrics
        if not m.endswith(("_prev", "_delta", "_pct_change"))
    ]
    is_comparison = intent is not None and intent.comparison != "none"

    # KPI card: single number.
    if len(base_metrics) >= 1 and not dims and not time_cols and result.row_count == 1:
        return {
            "chart_type": "kpi",
            "metric": base_metrics[0],
            "value": _values(result)[0].get(base_metrics[0]),
        }

    if not base_metrics:
        return {"chart_type": "table"}

    metric = base_metrics[0]

    # Comparison: line with delta in tooltip.
    if is_comparison and time_cols:
        spec = _base_spec(result, {"type": "line", "point": True})
        spec["encoding"] = {
            "x": {"field": time_cols[0], "type": "temporal"},
            "y": {"field": metric, "type": "quantitative"},
            "tooltip": [
                {"field": time_cols[0], "type": "temporal"},
                {"field": metric, "type": "quantitative"},
                {"field": f"{metric}_delta", "type": "quantitative"},
                {"field": f"{metric}_pct_change", "type": "quantitative", "format": ".1%"},
            ],
        }
        if dims:
            spec["encoding"]["color"] = {"field": dims[0], "type": "nominal"}
        spec["chart_type"] = "comparison_line"
        return spec

    # Time series.
    if time_cols and not dims:
        if len(base_metrics) == 1:
            spec = _base_spec(result, {"type": "line", "point": True})
            spec["encoding"] = {
                "x": {"field": time_cols[0], "type": "temporal"},
                "y": {"field": metric, "type": "quantitative"},
            }
            spec["chart_type"] = "line"
            return spec
        # 2+ metrics over time -> fold into a multi-line.
        spec = _base_spec(result, {"type": "line", "point": True})
        spec["transform"] = [{"fold": base_metrics, "as": ["metric", "value"]}]
        spec["encoding"] = {
            "x": {"field": time_cols[0], "type": "temporal"},
            "y": {"field": "value", "type": "quantitative"},
            "color": {"field": "metric", "type": "nominal"},
        }
        spec["chart_type"] = "multi_line"
        return spec

    # Time + one categorical -> multi-line.
    if time_cols and len(dims) >= 1:
        spec = _base_spec(result, {"type": "line", "point": True})
        spec["encoding"] = {
            "x": {"field": time_cols[0], "type": "temporal"},
            "y": {"field": metric, "type": "quantitative"},
            "color": {"field": dims[0], "type": "nominal"},
        }
        spec["chart_type"] = "multi_line"
        return spec

    # One categorical dimension -> bar (top-N when high-cardinality).
    if len(dims) == 1:
        cardinality = len({row[0] for row in result.rows}) if result.rows else 0
        if len(base_metrics) >= 2:
            spec = _base_spec(result, "bar")
            spec["transform"] = [{"fold": base_metrics, "as": ["metric", "value"]}]
            spec["encoding"] = {
                "x": {"field": dims[0], "type": "nominal", "sort": "-y"},
                "y": {"field": "value", "type": "quantitative"},
                "xOffset": {"field": "metric"},
                "color": {"field": "metric", "type": "nominal"},
            }
            spec["chart_type"] = "grouped_bar"
            return spec
        spec = _base_spec(result, "bar")
        spec["encoding"] = {
            "x": {"field": dims[0], "type": "nominal", "sort": "-y"},
            "y": {"field": metric, "type": "quantitative"},
        }
        if cardinality > TOP_N:
            # Keep top-N by metric; everything else collapses to "other".
            rows = sorted(
                _values(result),
                key=lambda r: (r.get(metric) is None, r.get(metric)),
                reverse=True,
            )
            top = rows[:TOP_N]
            other_total = sum(
                r.get(metric) or 0 for r in rows[TOP_N:]
            )
            top.append({dims[0]: "(other)", metric: other_total})
            spec["data"] = {"values": top}
            spec["chart_type"] = "bar_top_n"
        else:
            spec["chart_type"] = "bar"
        return spec

    # Two categorical dimensions -> heatmap.
    if len(dims) >= 2:
        spec = _base_spec(result, "rect")
        spec["encoding"] = {
            "x": {"field": dims[0], "type": "nominal"},
            "y": {"field": dims[1], "type": "nominal"},
            "color": {"field": metric, "type": "quantitative"},
        }
        spec["chart_type"] = "heatmap"
        return spec

    # Aggregate without grouping but multiple rows — fall back to table.
    return {"chart_type": "table"}
