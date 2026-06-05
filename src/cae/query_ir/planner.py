"""QueryIntent -> QueryPlan. Pure Python, no LLM, fully deterministic.

This is where comparisons (WoW / MoM / YoY) get desugared: the plan carries a
ComparisonSpec that the SQL generator renders as a LAG window over the base
aggregate — far more reliable than asking an LLM to write window functions.
"""

from __future__ import annotations

from datetime import date

from cae.exceptions import PlanningError
from cae.intent_parser.timeparse import resolve_time_range
from cae.models import (
    ComparisonSpec,
    Filter,
    FilterClause,
    OrderClause,
    QueryIntent,
    QueryPlan,
    SelectItem,
)
from cae.semantic_layer import SemanticLayer


def _sql_literal(value: str | int | float) -> str:
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return str(value)


def _render_filter(expr: str, f: Filter) -> str:
    values = f.values
    if f.op in ("in", "not_in"):
        rendered = ", ".join(_sql_literal(v) for v in values)
        keyword = "IN" if f.op == "in" else "NOT IN"
        return f"{expr} {keyword} ({rendered})"
    if f.op == "between":
        if len(values) != 2:
            raise PlanningError("'between' filter needs exactly two values")
        return f"{expr} BETWEEN {_sql_literal(values[0])} AND {_sql_literal(values[1])}"
    if len(values) != 1:
        # "region = [A, B]" is what the user means by "A or B" — promote to IN.
        if f.op == "=":
            rendered = ", ".join(_sql_literal(v) for v in values)
            return f"{expr} IN ({rendered})"
        raise PlanningError(f"operator '{f.op}' needs exactly one value")
    return f"{expr} {f.op} {_sql_literal(values[0])}"


class Planner:
    def __init__(
        self,
        layer: SemanticLayer,
        default_limit: int = 1000,
        lookback_years: int = 5,
    ) -> None:
        self.layer = layer
        self.default_limit = default_limit
        self.lookback_years = lookback_years

    def plan(self, intent: QueryIntent, today: date | None = None) -> QueryPlan:
        today = today or date.today()
        layer = self.layer

        metric_defs = [layer.resolve_metric(m) for m in intent.metrics]
        dim_defs = [layer.resolve_dimension(d) for d in intent.dimensions]
        if any(m is None for m in metric_defs) or any(d is None for d in dim_defs):
            raise PlanningError("intent must be validated before planning")

        select: list[SelectItem] = []
        group_by: list[str] = []

        # Time axis first (when the user wants a trend or a comparison).
        grain = intent.time_range.grain if intent.time_range else None
        if grain:
            period_expr = f"DATE_TRUNC('{grain}', {layer.time_expr})"
            select.append(SelectItem(alias="period", expr=period_expr, role="time"))
            group_by.append(period_expr)

        for d in dim_defs:
            select.append(SelectItem(alias=d.name, expr=d.expr, role="dimension"))
            group_by.append(d.expr)

        for m in metric_defs:
            select.append(SelectItem(alias=m.name, expr=m.expr, role="metric"))

        # Joins: whatever the metrics/dims/filters require, connected by BFS.
        filter_dims = [f.dimension for f in intent.filters]
        joins = layer.required_joins(
            intent.metrics, intent.dimensions + filter_dims
        )

        # WHERE: mandatory time bound + user filters.
        where: list[FilterClause] = []
        start, end = resolve_time_range(
            intent.time_range, today, self.lookback_years
        )
        where.append(
            FilterClause(
                expr=(
                    f"{layer.time_expr} >= DATE '{start.isoformat()}' "
                    f"AND {layer.time_expr} <= DATE '{end.isoformat()}'"
                )
            )
        )
        for f in intent.filters:
            dim = layer.resolve_dimension(f.dimension)
            assert dim is not None
            where.append(FilterClause(expr=_render_filter(dim.expr, f)))

        # ORDER BY: explicit sort > period asc > first metric desc.
        order_by: list[OrderClause] = []
        if intent.comparison != "none":
            for d in dim_defs:
                order_by.append(OrderClause(expr=d.name, direction="asc"))
            order_by.append(OrderClause(expr="period", direction="asc"))
        elif intent.sort:
            for s in intent.sort:
                order_by.append(OrderClause(expr=s.field, direction=s.direction))
        elif grain:
            order_by.append(OrderClause(expr="period", direction="asc"))
        elif metric_defs:
            order_by.append(
                OrderClause(expr=metric_defs[0].name, direction="desc")
            )

        comparison: ComparisonSpec | None = None
        if intent.comparison != "none":
            comparison = ComparisonSpec(
                kind=intent.comparison,
                value_aliases=[m.name for m in metric_defs],
                partition_by=[d.name for d in dim_defs],
            )

        fact = layer.tables[layer.fact_table]
        return QueryPlan(
            select=select,
            from_table=fact.name,
            from_physical=fact.physical,
            joins=joins,
            where=where,
            group_by=group_by,
            order_by=order_by,
            limit=intent.limit if intent.limit is not None else self.default_limit,
            comparison=comparison,
        )
