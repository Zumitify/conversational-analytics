"""Intent parser — LLM call #1: natural language -> typed QueryIntent.

Conversation context is passed as the *previous intent object* (not raw chat
history), so follow-ups like "now break that down by region" are resolved
against a real structure. Post-call validation rejects anything not in the
semantic layer with closest-match suggestions.
"""

from __future__ import annotations

import json
from datetime import date

from cae.exceptions import ClarificationNeeded
from cae.intent_parser.timeparse import resolve_time_range
from cae.llm.client import LLMProvider
from cae.models import Filter, QueryIntent, SortSpec, TimeRange, Usage
from cae.semantic_layer import SemanticLayer

SYSTEM_TEMPLATE = """\
You translate analytics questions into a structured QueryIntent object.

Rules:
- Use ONLY metric and dimension names from the vocabulary below (canonical \
names, not synonyms).
- `metrics` is what gets aggregated; `dimensions` is the group-by.
- Filters reference dimensions; enum values must match the allowed values.
- For time, prefer `relative` strings: last_N_days / last_N_weeks / \
last_N_months / last_N_quarters / last_N_years, last_week, last_month, \
last_quarter, last_year, this_week, this_month, this_quarter, ytd, qtd, mtd, \
qN_YYYY (e.g. q3_2024), or a bare year (e.g. 2024). Use explicit start/end \
dates only when the user gives exact dates.
- Set `grain` ONLY when the user wants a trend over time (per day/week/...).
- `comparison`: wow / mom / yoy when the user asks for growth or change \
versus the prior period; otherwise "none".
- If a FOLLOW-UP context intent is provided, return the FULL updated intent \
(start from the previous intent and apply the user's edit), not a patch.
- `limit` only when the user asks for top/bottom N; pair it with `sort`.

VOCABULARY:
{vocabulary}

Examples:

Q: total revenue last month
-> {{"metrics": ["revenue"], "dimensions": [], "filters": [], \
"time_range": {{"relative": "last_month"}}, "comparison": "none"}}

Q: weekly revenue trend for the Northeast this year
-> {{"metrics": ["revenue"], "dimensions": [], "filters": [{{"dimension": \
"region", "op": "=", "values": ["Northeast"]}}], "time_range": {{"grain": \
"week", "relative": "ytd"}}, "comparison": "none"}}

Q: week-over-week revenue growth in Q3 2024 by product line
-> {{"metrics": ["revenue"], "dimensions": ["product_line"], "filters": [], \
"time_range": {{"grain": "week", "relative": "q3_2024"}}, "comparison": "wow"}}

Q: top 5 categories by units sold last quarter
-> {{"metrics": ["units_sold"], "dimensions": ["category"], "filters": [], \
"time_range": {{"relative": "last_quarter"}}, "comparison": "none", \
"sort": [{{"field": "units_sold", "direction": "desc"}}], "limit": 5}}

Follow-up example — previous intent had metrics=["revenue"], filters on \
region=Northeast; user says "now just enterprise customers":
-> same intent with filters = [region=Northeast, segment=Enterprise]
"""


def build_system_prompt(layer: SemanticLayer) -> str:
    return SYSTEM_TEMPLATE.format(vocabulary=layer.to_prompt_context())


def build_user_prompt(question: str, previous_intent: QueryIntent | None) -> str:
    parts = []
    if previous_intent is not None:
        parts.append(
            "FOLLOW-UP CONTEXT — the user is iterating on this previous intent:\n"
            + json.dumps(previous_intent.model_dump(mode="json"))
        )
    parts.append(f"QUESTION: {question}")
    return "\n\n".join(parts)


class IntentParser:
    def __init__(self, provider: LLMProvider, layer: SemanticLayer, max_tokens: int = 2048):
        self.provider = provider
        self.layer = layer
        self.max_tokens = max_tokens
        self._system = build_system_prompt(layer)

    def parse(
        self,
        question: str,
        previous_intent: QueryIntent | None = None,
        today: date | None = None,
    ) -> tuple[QueryIntent, Usage]:
        intent, usage = self.provider.parse(
            system=self._system,
            user=build_user_prompt(question, previous_intent),
            output_model=QueryIntent,
            max_tokens=self.max_tokens,
        )
        validated = validate_intent(intent, self.layer, today=today or date.today())
        return validated, usage


def validate_intent(
    intent: QueryIntent, layer: SemanticLayer, today: date | None = None
) -> QueryIntent:
    """Reject unknown names / illegal values; normalize to canonical names.

    Raises ClarificationNeeded with closest-match suggestions so the UI can
    ask one useful question instead of failing opaquely.
    """
    today = today or date.today()

    metrics: list[str] = []
    for name in intent.metrics:
        metric = layer.resolve_metric(name)
        if metric is None:
            raise ClarificationNeeded(
                f"I don't know a metric called '{name}'.",
                suggestions=layer.suggest(name, kind="metric"),
            )
        if metric.name not in metrics:
            metrics.append(metric.name)

    dimensions: list[str] = []
    for name in intent.dimensions:
        dim = layer.resolve_dimension(name)
        if dim is None:
            raise ClarificationNeeded(
                f"I don't know a dimension called '{name}'.",
                suggestions=layer.suggest(name, kind="dimension"),
            )
        if dim.pii:
            raise ClarificationNeeded(
                f"'{dim.name}' contains personal data and can't be used for grouping."
            )
        if dim.name not in dimensions:
            dimensions.append(dim.name)

    filters: list[Filter] = []
    for f in intent.filters:
        dim = layer.resolve_dimension(f.dimension)
        if dim is None:
            raise ClarificationNeeded(
                f"I can't filter on '{f.dimension}' — it's not a known dimension.",
                suggestions=layer.suggest(f.dimension, kind="dimension"),
            )
        values: list[str | int | float] = []
        for value in f.values:
            resolved = layer.resolve_enum_value(dim, value)
            if resolved is None:
                raise ClarificationNeeded(
                    f"'{value}' is not a valid value for {dim.name}.",
                    suggestions=dim.values,
                )
            values.append(resolved)
        filters.append(Filter(dimension=dim.name, op=f.op, values=values))

    sort: list[SortSpec] = []
    valid_sort_fields = set(metrics) | set(dimensions) | {"period"}
    for s in intent.sort:
        metric = layer.resolve_metric(s.field)
        dim = layer.resolve_dimension(s.field)
        field = metric.name if metric else (dim.name if dim else s.field)
        if field not in valid_sort_fields:
            raise ClarificationNeeded(
                f"I can't sort by '{s.field}' — it isn't selected in this query."
            )
        sort.append(SortSpec(field=field, direction=s.direction))

    time_range = intent.time_range
    comparison = intent.comparison
    if comparison != "none":
        # Comparisons need a period axis; align grain with the comparison kind.
        grain = {"wow": "week", "mom": "month", "yoy": "year"}[comparison]
        time_range = (time_range or TimeRange()).model_copy(update={"grain": grain})
    if time_range and time_range.relative:
        # Fail fast on unparseable relative strings (raises ClarificationNeeded).
        resolve_time_range(time_range, today)

    if intent.limit is not None and intent.limit <= 0:
        raise ClarificationNeeded("Limit must be a positive number.")

    return QueryIntent(
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        time_range=time_range,
        comparison=comparison,
        sort=sort,
        limit=intent.limit,
        explain=intent.explain,
    )
