"""Core typed contracts that flow between pipeline stages.

Three objects matter most (see design doc §4):

1. ``QueryIntent``  - what the user wants, in semantic-layer terms.
2. ``QueryPlan``    - how to compute it, in physical terms (dialect-free).
3. ``QueryResult``  - what came back, with column roles attached.

Everything is a Pydantic v2 model so each stage boundary is validated.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

FilterOp = Literal["=", "!=", "in", "not_in", ">", "<", ">=", "<=", "between"]
Grain = Literal["day", "week", "month", "quarter", "year"]
Comparison = Literal["wow", "mom", "yoy", "none"]
ColumnRole = Literal["metric", "dimension", "time", "other"]


# ---------------------------------------------------------------------------
# 1. QueryIntent — output of the intent parser (LLM call #1)
# ---------------------------------------------------------------------------

class Filter(BaseModel):
    dimension: str
    op: FilterOp = "="
    values: list[str | int | float]


class TimeRange(BaseModel):
    grain: Grain | None = None
    start: date | None = None
    end: date | None = None
    relative: str | None = None  # e.g. "last_4_weeks", "ytd", "q3_2024"


class SortSpec(BaseModel):
    # The design doc sketches sort as list[tuple[str, Literal]]; a model is
    # used instead because tuples don't round-trip through JSON-schema
    # structured outputs.
    field: str
    direction: Literal["asc", "desc"] = "desc"


class QueryIntent(BaseModel):
    metrics: list[str] = Field(min_length=1)
    dimensions: list[str] = []
    filters: list[Filter] = []
    time_range: TimeRange | None = None
    comparison: Comparison = "none"
    sort: list[SortSpec] = []
    limit: int | None = None
    explain: bool = False


# ---------------------------------------------------------------------------
# 2. QueryPlan — output of the deterministic planner (no LLM)
# ---------------------------------------------------------------------------

class SelectItem(BaseModel):
    alias: str
    expr: str
    role: ColumnRole


class JoinEdge(BaseModel):
    table: str      # logical table name (used as SQL alias)
    physical: str   # physical relation name
    on: str         # join condition referencing logical aliases


class FilterClause(BaseModel):
    expr: str  # fully rendered boolean SQL fragment


class OrderClause(BaseModel):
    expr: str  # alias or expression
    direction: Literal["asc", "desc"] = "asc"


class ComparisonSpec(BaseModel):
    """Desugared comparison: rendered as a LAG window over the base query."""

    kind: Literal["wow", "mom", "yoy"]
    value_aliases: list[str]      # metric aliases to diff
    partition_by: list[str] = []  # dimension aliases
    order_alias: str = "period"


class QueryPlan(BaseModel):
    select: list[SelectItem]
    from_table: str   # logical name of the fact table
    from_physical: str
    joins: list[JoinEdge] = []
    where: list[FilterClause] = []
    group_by: list[str] = []
    order_by: list[OrderClause] = []
    limit: int | None = None
    comparison: ComparisonSpec | None = None

    def column_roles(self) -> dict[str, ColumnRole]:
        roles: dict[str, ColumnRole] = {s.alias: s.role for s in self.select}
        if self.comparison:
            for alias in self.comparison.value_aliases:
                roles[f"{alias}_prev"] = "metric"
                roles[f"{alias}_delta"] = "metric"
                roles[f"{alias}_pct_change"] = "metric"
        return roles


# ---------------------------------------------------------------------------
# 3. QueryResult — output of the execution engine
# ---------------------------------------------------------------------------

class ColumnMeta(BaseModel):
    name: str
    sql_type: str = ""
    role: ColumnRole = "other"


class QueryResult(BaseModel):
    columns: list[ColumnMeta]
    rows: list[list[Any]]
    row_count: int
    elapsed_ms: int
    truncated: bool = False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    ok: bool
    errors: list[str] = []
    warnings: list[str] = []
    rewritten_sql: str | None = None  # e.g. with LIMIT injected


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

class ResultDigest(BaseModel):
    """Compact view of a result — stored per turn instead of full rows."""

    columns: list[str]
    row_count: int
    metric_stats: dict[str, dict[str, float]] = {}  # alias -> {min,max,sum}

    @classmethod
    def from_result(cls, result: QueryResult) -> "ResultDigest":
        stats: dict[str, dict[str, float]] = {}
        for idx, col in enumerate(result.columns):
            if col.role != "metric":
                continue
            values = [
                float(r[idx]) for r in result.rows
                if r[idx] is not None and isinstance(r[idx], (int, float))
            ]
            if values:
                stats[col.name] = {
                    "min": min(values), "max": max(values), "sum": sum(values),
                }
        return cls(
            columns=[c.name for c in result.columns],
            row_count=result.row_count,
            metric_stats=stats,
        )


class Turn(BaseModel):
    user_text: str
    intent: QueryIntent
    plan: QueryPlan
    sql: str
    result_digest: ResultDigest
    chart_spec: dict[str, Any] = {}
    summary_text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Session(BaseModel):
    session_id: str
    turns: list[Turn] = []
    active_intent: QueryIntent | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Pipeline response — what /ask returns
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class AskResponse(BaseModel):
    session_id: str
    question: str
    intent: QueryIntent
    sql: str
    result: QueryResult
    chart_spec: dict[str, Any]
    summary: str
    summary_dropped: bool = False  # faithfulness check failed -> summary removed
    warnings: list[str] = []
    usage: Usage = Usage()
    stage_timings_ms: dict[str, int] = {}
