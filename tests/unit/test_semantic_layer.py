"""Unit tests: resolution, synonym matching, join-graph correctness."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cae.exceptions import UnreachableJoinError
from cae.semantic_layer import SemanticLayer


class TestResolution:
    def test_resolve_metric_canonical(self, layer):
        assert layer.resolve_metric("revenue").name == "revenue"

    def test_resolve_metric_synonym(self, layer):
        assert layer.resolve_metric("sales").name == "revenue"
        assert layer.resolve_metric("AOV").name == "average_order_value"

    def test_resolve_metric_case_and_underscores(self, layer):
        assert layer.resolve_metric("Units_Sold").name == "units_sold"
        assert layer.resolve_metric("GROSS REVENUE").name == "revenue"

    def test_resolve_unknown_metric(self, layer):
        assert layer.resolve_metric("profit margin") is None

    def test_resolve_dimension_synonym(self, layer):
        assert layer.resolve_dimension("vendor").name == "supplier"
        assert layer.resolve_dimension("sales channel").name == "channel"

    def test_suggest_close_match(self, layer):
        suggestions = layer.suggest("revenu", kind="metric")
        assert "revenue" in suggestions

    def test_suggest_dimension(self, layer):
        suggestions = layer.suggest("regin", kind="dimension")
        assert "region" in suggestions

    def test_enum_value_case_insensitive(self, layer):
        dim = layer.resolve_dimension("region")
        assert layer.resolve_enum_value(dim, "northeast") == "Northeast"
        assert layer.resolve_enum_value(dim, "Atlantis") is None

    def test_enum_passthrough_for_free_dims(self, layer):
        dim = layer.resolve_dimension("category")  # no declared values
        assert layer.resolve_enum_value(dim, "Anything") == "Anything"


class TestJoinGraph:
    def test_no_joins_for_fact_only(self, layer):
        assert layer.required_joins(["orders_count"], []) == []

    def test_single_hop(self, layer):
        edges = layer.required_joins(["revenue"], [])
        assert [e.table for e in edges] == ["order_items"]

    def test_multi_hop_parent_first(self, layer):
        edges = layer.required_joins(["revenue"], ["category"])
        tables = [e.table for e in edges]
        # parent joins must come before children
        assert tables.index("order_items") < tables.index("products")
        assert tables.index("products") < tables.index("categories")

    def test_combined_requirements_deduped(self, layer):
        edges = layer.required_joins(["revenue", "units_sold"], ["region", "product_line"])
        tables = [e.table for e in edges]
        assert len(tables) == len(set(tables))
        assert {"order_items", "products", "customers"} <= set(tables)

    def test_unreachable_table_raises(self):
        broken = SemanticLayer.from_dict({
            "fact_table": "orders",
            "tables": {
                "orders": {"physical": "orders", "time_column": "order_date"},
                "island": {"physical": "island"},  # no join path
            },
            "metrics": {
                "stranded": {"expr": "SUM(island.x)", "requires": ["island"]},
            },
            "dimensions": {},
        })
        with pytest.raises(UnreachableJoinError):
            broken.required_joins(["stranded"], [])

    def test_unknown_required_table_fails_at_load(self):
        with pytest.raises(ValueError, match="unknown table"):
            SemanticLayer.from_dict({
                "fact_table": "orders",
                "tables": {"orders": {"physical": "orders"}},
                "metrics": {
                    "bad": {"expr": "SUM(x)", "requires": ["ghost"]},
                },
                "dimensions": {},
            })


class TestJoinGraphProperty:
    @settings(max_examples=50, deadline=None)
    @given(data=st.data())
    def test_any_subset_connects_or_raises(self, data):
        """Property: any subset of metrics+dimensions yields a join list whose
        edges are parent-first connected, or a clear UnreachableJoinError."""
        layer = SemanticLayer.from_yaml(
            __import__("tests.conftest", fromlist=["SEMANTIC_LAYER_PATH"]).SEMANTIC_LAYER_PATH
        )
        metric_names = [m.name for m in layer.list_metrics()]
        dim_names = [d.name for d in layer.list_dimensions()]
        metrics = data.draw(st.lists(st.sampled_from(metric_names), max_size=4))
        dims = data.draw(st.lists(st.sampled_from(dim_names), max_size=4))
        try:
            edges = layer.required_joins(metrics, dims)
        except UnreachableJoinError:
            return  # acceptable, explicit failure mode
        connected = {layer.fact_table}
        for edge in edges:
            # every edge must attach to the already-connected component
            referenced = {t for t in connected if t in edge.on}
            assert referenced or edge.table in connected or any(
                t + "." in edge.on for t in connected
            ), f"edge {edge.table} not connected: {edge.on}"
            connected.add(edge.table)


class TestPromptContext:
    def test_contains_metrics_and_synonyms(self, layer):
        context = layer.to_prompt_context()
        assert "revenue" in context
        assert "synonyms" in context
        assert "Northeast" in context  # enum values surfaced

    def test_excludes_pii(self, layer):
        assert "customer_email" not in layer.to_prompt_context()
