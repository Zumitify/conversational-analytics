"""SQL validator: refuse to execute anything unsafe or wrong.

Layers (design doc §3.5):
1. Parse        — sqlglot, dialect-aware; reject on any parse error.
2. Safety       — single SELECT only; no DDL/DML anywhere; allow-listed
                  tables only; LIMIT injected/capped; mandatory time bound
                  on queries that touch the fact table.
3. Semantic     — every referenced table/column exists in the live catalog;
                  every JOIN condition is one declared in the semantic layer
                  (no Cartesian products from a hallucinated ON 1=1).
"""

from __future__ import annotations

import sqlglot
from sqlglot import expressions as exp

from cae.models import ValidationResult
from cae.semantic_layer import SemanticLayer

_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Merge,
    exp.TruncateTable,
    exp.Command,  # PRAGMA / SET / COPY / arbitrary commands
)


class SQLValidator:
    def __init__(
        self,
        layer: SemanticLayer,
        catalog: dict[str, set[str]] | None = None,
        max_rows: int = 10_000,
    ) -> None:
        self.layer = layer
        self.catalog = catalog  # table -> columns, from the live database
        self.max_rows = max_rows

    def validate(self, sql: str, dialect: str = "duckdb") -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # -- 1. parse -------------------------------------------------------
        try:
            statements = sqlglot.parse(sql, read=dialect)
        except sqlglot.errors.ParseError as e:
            return ValidationResult(ok=False, errors=[f"parse error: {e}"])
        statements = [s for s in statements if s is not None]
        if len(statements) != 1:
            return ValidationResult(
                ok=False,
                errors=[f"expected exactly one statement, got {len(statements)}"],
            )
        tree = statements[0]

        # -- 2. safety ------------------------------------------------------
        if not isinstance(tree, exp.Select):
            errors.append(f"only SELECT statements are allowed, got {tree.key.upper()}")
        for node_type in _FORBIDDEN_NODES:
            if list(tree.find_all(node_type)):
                errors.append(f"forbidden construct: {node_type.__name__.upper()}")
        if errors:
            return ValidationResult(ok=False, errors=errors)

        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        allowed = {t.lower() for t in self.layer.allowed_physical_tables()}

        referenced: dict[str, str] = {}  # alias -> physical
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            if name in cte_names:
                continue
            if table.db:  # schema-qualified access is outside the sandbox
                errors.append(f"schema-qualified table access not allowed: {table.sql()}")
                continue
            if name not in allowed:
                errors.append(f"table not in allow-list: {name}")
                continue
            referenced[(table.alias_or_name or name).lower()] = name

        # -- 3. semantic: joins must be declared ----------------------------
        declared = self.layer.declared_join_conditions()
        for join in tree.find_all(exp.Join):
            on = join.args.get("on")
            if on is None:
                errors.append(f"JOIN without ON condition: {join.sql()[:80]}")
                continue
            normalized = " ".join(on.sql().lower().replace('"', "").split())
            if normalized not in declared:
                errors.append(f"undeclared join condition: {on.sql()}")

        # -- 3. semantic: catalog check --------------------------------------
        if self.catalog is not None:
            for name in referenced.values():
                if name not in self.catalog:
                    errors.append(f"table missing from database catalog: {name}")
            for column in tree.find_all(exp.Column):
                table_ref = (column.table or "").lower()
                if not table_ref or table_ref in cte_names or table_ref == "base":
                    continue
                physical = referenced.get(table_ref)
                if physical and physical in self.catalog:
                    if column.name.lower() not in self.catalog[physical]:
                        errors.append(
                            f"column not in catalog: {table_ref}.{column.name}"
                        )

        # -- 2. safety: mandatory time bound ---------------------------------
        fact = self.layer.time_table
        if fact.time_column and fact.physical.lower() in referenced.values():
            if not self._has_time_bound(tree, fact.time_column):
                errors.append(
                    f"query touches '{fact.name}' but has no time bound on "
                    f"'{fact.time_column}' (mandatory lookback cap)"
                )

        # -- 2. safety: LIMIT injection / capping -----------------------------
        rewritten_sql: str | None = None
        limit_node = tree.args.get("limit")
        if limit_node is None:
            tree = tree.limit(self.max_rows)
            rewritten_sql = tree.sql(dialect=dialect, pretty=True)
            warnings.append(f"LIMIT {self.max_rows} injected")
        else:
            try:
                current = int(limit_node.expression.this)
                if current > self.max_rows:
                    tree.args["limit"] = None
                    tree = tree.limit(self.max_rows)
                    rewritten_sql = tree.sql(dialect=dialect, pretty=True)
                    warnings.append(f"LIMIT capped at {self.max_rows}")
            except (TypeError, ValueError, AttributeError):
                errors.append("non-literal LIMIT is not allowed")

        return ValidationResult(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            rewritten_sql=rewritten_sql if not errors else None,
        )

    @staticmethod
    def _has_time_bound(tree: exp.Select, time_column: str) -> bool:
        """True if any WHERE clause constrains the fact table's time column."""
        for where in tree.find_all(exp.Where):
            for column in where.find_all(exp.Column):
                if column.name.lower() == time_column.lower():
                    return True
        return False
