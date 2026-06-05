"""Semantic layer: the single source of truth about the data.

Loads YAML definitions of tables, metrics, dimensions, joins and synonyms.
Resolves logical names ("revenue", "northeast region") to physical SQL
fragments, and computes the join path needed by any metric/dimension subset
via BFS over the declared join graph.
"""

from __future__ import annotations

import difflib
from collections import deque
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from cae.exceptions import UnreachableJoinError
from cae.models import JoinEdge


class TableJoin(BaseModel):
    to: str
    on: str


class TableDef(BaseModel):
    name: str
    physical: str
    primary_key: str | None = None
    time_column: str | None = None
    joins: list[TableJoin] = []


class MetricDef(BaseModel):
    name: str
    expr: str
    requires: list[str] = []
    synonyms: list[str] = []
    format: str = "number"
    description: str = ""


class DimensionDef(BaseModel):
    name: str
    expr: str
    requires: list[str] = []
    values: list[str] = []
    synonyms: list[str] = []
    pii: bool = False


class SemanticLayer(BaseModel):
    fact_table: str
    tables: dict[str, TableDef]
    metrics: dict[str, MetricDef]
    dimensions: dict[str, DimensionDef]

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticLayer":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "SemanticLayer":
        tables = {}
        for name, spec in (raw.get("tables") or {}).items():
            # YAML 1.1 parses a bare `on:` key as boolean True — normalize so
            # hand-authored layers don't need to quote it.
            for join in spec.get("joins") or []:
                if True in join:
                    join["on"] = join.pop(True)
            tables[name] = TableDef(name=name, **spec)
        metrics = {
            name: MetricDef(name=name, **spec)
            for name, spec in (raw.get("metrics") or {}).items()
        }
        dimensions = {
            name: DimensionDef(name=name, **spec)
            for name, spec in (raw.get("dimensions") or {}).items()
        }
        layer = cls(
            fact_table=raw["fact_table"],
            tables=tables,
            metrics=metrics,
            dimensions=dimensions,
        )
        layer._check_references()
        return layer

    def _check_references(self) -> None:
        """Fail fast at load time if a definition references an unknown table."""
        for kind, defs in (("metric", self.metrics), ("dimension", self.dimensions)):
            for d in defs.values():
                for table in d.requires:
                    if table not in self.tables:
                        raise ValueError(
                            f"{kind} '{d.name}' requires unknown table '{table}'"
                        )
        for table in self.tables.values():
            for join in table.joins:
                if join.to not in self.tables:
                    raise ValueError(
                        f"table '{table.name}' joins to unknown table '{join.to}'"
                    )

    # -- resolution --------------------------------------------------------

    def list_metrics(self) -> list[MetricDef]:
        return list(self.metrics.values())

    def list_dimensions(self) -> list[DimensionDef]:
        return list(self.dimensions.values())

    def resolve_metric(self, name: str) -> MetricDef | None:
        return self._resolve(name, self.metrics)

    def resolve_dimension(self, name: str) -> DimensionDef | None:
        return self._resolve(name, self.dimensions)

    @staticmethod
    def _resolve(name: str, defs: dict) -> MetricDef | DimensionDef | None:
        needle = name.strip().lower().replace("_", " ")
        for d in defs.values():
            candidates = [d.name.lower().replace("_", " ")]
            candidates += [s.lower() for s in d.synonyms]
            if needle in candidates:
                return d
        return None

    def suggest(self, name: str, kind: str = "any", n: int = 3) -> list[str]:
        """Closest canonical names for an unresolved input (for clarifications)."""
        pool: list[tuple[str, str]] = []  # (candidate, canonical)
        sources = []
        if kind in ("metric", "any"):
            sources.append(self.metrics)
        if kind in ("dimension", "any"):
            sources.append(self.dimensions)
        for defs in sources:
            for d in defs.values():
                pool.append((d.name.lower().replace("_", " "), d.name))
                pool.extend((s.lower(), d.name) for s in d.synonyms)
        needle = name.strip().lower().replace("_", " ")
        matches = difflib.get_close_matches(
            needle, [c for c, _ in pool], n=n * 2, cutoff=0.4
        )
        canonical: list[str] = []
        for m in matches:
            for cand, canon in pool:
                if cand == m and canon not in canonical:
                    canonical.append(canon)
        return canonical[:n]

    def resolve_enum_value(self, dim: DimensionDef, value: str | int | float):
        """Case-insensitive match of a filter value against declared enum values."""
        if not dim.values or not isinstance(value, str):
            return value
        for allowed in dim.values:
            if allowed.lower() == value.strip().lower():
                return allowed
        return None

    # -- join graph --------------------------------------------------------

    def required_joins(self, metrics: list[str], dims: list[str]) -> list[JoinEdge]:
        """Ordered JOIN edges connecting the fact table to every required table.

        BFS over the declared join graph; raises UnreachableJoinError if any
        required table cannot be reached from the fact table.
        """
        required: set[str] = set()
        for name in metrics:
            m = self.resolve_metric(name)
            if m:
                required.update(m.requires)
        for name in dims:
            d = self.resolve_dimension(name)
            if d:
                required.update(d.requires)
        required.discard(self.fact_table)
        if not required:
            return []

        # BFS from fact table; record the parent edge for each discovered table.
        parent: dict[str, tuple[str, TableJoin]] = {}
        seen = {self.fact_table}
        queue = deque([self.fact_table])
        while queue:
            current = queue.popleft()
            for join in self.tables[current].joins:
                if join.to not in seen:
                    seen.add(join.to)
                    parent[join.to] = (current, join)
                    queue.append(join.to)

        unreachable = required - seen
        if unreachable:
            raise UnreachableJoinError(
                f"No join path from '{self.fact_table}' to: {sorted(unreachable)}"
            )

        # Walk each required table back to the fact table, collecting edges,
        # then emit them in discovery (parent-first) order without duplicates.
        needed: set[str] = set()
        for table in required:
            cursor = table
            while cursor != self.fact_table:
                needed.add(cursor)
                cursor = parent[cursor][0]

        edges: list[JoinEdge] = []
        emitted: set[str] = set()

        def emit(table: str) -> None:
            if table in emitted or table == self.fact_table:
                return
            source, join = parent[table]
            emit(source)  # parent join must come first
            emitted.add(table)
            edges.append(
                JoinEdge(table=table, physical=self.tables[table].physical, on=join.on)
            )

        for table in sorted(needed):  # deterministic order
            emit(table)
        return edges

    # -- helpers used by the planner / validator ---------------------------

    @property
    def time_table(self) -> TableDef:
        return self.tables[self.fact_table]

    @property
    def time_expr(self) -> str:
        table = self.time_table
        if not table.time_column:
            raise ValueError(f"fact table '{table.name}' has no time_column")
        return f"{table.name}.{table.time_column}"

    def allowed_physical_tables(self) -> set[str]:
        return {t.physical for t in self.tables.values()}

    def declared_join_conditions(self) -> set[str]:
        conditions = set()
        for table in self.tables.values():
            for join in table.joins:
                conditions.add(" ".join(join.on.lower().split()))
        return conditions

    # -- LLM prompt view ----------------------------------------------------

    def to_prompt_context(self) -> str:
        """Compact, curated vocabulary the intent parser is constrained to.

        PII-flagged dimensions are excluded entirely.
        """
        lines = ["METRICS (use these exact names):"]
        for m in self.metrics.values():
            syn = f" (synonyms: {', '.join(m.synonyms)})" if m.synonyms else ""
            desc = f" — {m.description}" if m.description else ""
            lines.append(f"- {m.name}{syn}{desc}")
        lines.append("\nDIMENSIONS (use these exact names):")
        for d in self.dimensions.values():
            if d.pii:
                continue
            syn = f" (synonyms: {', '.join(d.synonyms)})" if d.synonyms else ""
            vals = f" — allowed values: {', '.join(d.values)}" if d.values else ""
            lines.append(f"- {d.name}{syn}{vals}")
        lines.append(
            "\nTIME: data is organized by order date; grains: day, week, month, "
            "quarter, year. Comparisons: wow (week over week), mom, yoy."
        )
        return "\n".join(lines)
