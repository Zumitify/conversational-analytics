"""Unit tests: post-LLM intent validation and normalization."""

from __future__ import annotations

from datetime import date

import pytest

from cae.exceptions import ClarificationNeeded
from cae.intent_parser import validate_intent
from cae.models import Filter, QueryIntent, SortSpec, TimeRange

TODAY = date(2026, 6, 4)


def validate(intent: QueryIntent, layer):
    return validate_intent(intent, layer, today=TODAY)


class TestMetricValidation:
    def test_unknown_metric_raises_with_suggestions(self, layer):
        with pytest.raises(ClarificationNeeded) as exc:
            validate(QueryIntent(metrics=["revenu"]), layer)
        assert "revenue" in exc.value.suggestions

    def test_synonym_normalized_to_canonical(self, layer):
        intent = validate(QueryIntent(metrics=["sales"]), layer)
        assert intent.metrics == ["revenue"]

    def test_duplicate_metrics_deduped(self, layer):
        intent = validate(QueryIntent(metrics=["sales", "revenue"]), layer)
        assert intent.metrics == ["revenue"]


class TestDimensionValidation:
    def test_unknown_dimension_raises(self, layer):
        with pytest.raises(ClarificationNeeded):
            validate(QueryIntent(metrics=["revenue"], dimensions=["continent"]), layer)

    def test_pii_dimension_rejected(self, layer):
        with pytest.raises(ClarificationNeeded, match="personal data"):
            validate(
                QueryIntent(metrics=["revenue"], dimensions=["customer_email"]), layer
            )


class TestFilterValidation:
    def test_enum_value_canonicalized(self, layer):
        intent = validate(
            QueryIntent(
                metrics=["revenue"],
                filters=[Filter(dimension="region", op="=", values=["northeast"])],
            ),
            layer,
        )
        assert intent.filters[0].values == ["Northeast"]

    def test_illegal_enum_value_raises_with_allowed_values(self, layer):
        with pytest.raises(ClarificationNeeded) as exc:
            validate(
                QueryIntent(
                    metrics=["revenue"],
                    filters=[Filter(dimension="region", op="=", values=["Atlantis"])],
                ),
                layer,
            )
        assert "Northeast" in exc.value.suggestions

    def test_filter_dimension_synonym_normalized(self, layer):
        intent = validate(
            QueryIntent(
                metrics=["revenue"],
                filters=[Filter(dimension="area", op="=", values=["West"])],
            ),
            layer,
        )
        assert intent.filters[0].dimension == "region"


class TestComparisonAndTime:
    def test_comparison_pins_grain(self, layer):
        intent = validate(
            QueryIntent(metrics=["revenue"], comparison="wow"), layer
        )
        assert intent.time_range.grain == "week"

    def test_mom_pins_month(self, layer):
        intent = validate(
            QueryIntent(
                metrics=["revenue"],
                comparison="mom",
                time_range=TimeRange(relative="ytd"),
            ),
            layer,
        )
        assert intent.time_range.grain == "month"
        assert intent.time_range.relative == "ytd"

    def test_bad_relative_raises(self, layer):
        with pytest.raises(ClarificationNeeded):
            validate(
                QueryIntent(
                    metrics=["revenue"],
                    time_range=TimeRange(relative="whenever"),
                ),
                layer,
            )


class TestSortAndLimit:
    def test_sort_field_must_be_selected(self, layer):
        with pytest.raises(ClarificationNeeded):
            validate(
                QueryIntent(
                    metrics=["revenue"],
                    sort=[SortSpec(field="units_sold", direction="desc")],
                ),
                layer,
            )

    def test_sort_synonym_normalized(self, layer):
        intent = validate(
            QueryIntent(
                metrics=["units_sold"],
                dimensions=["category"],
                sort=[SortSpec(field="quantity sold", direction="desc")],
            ),
            layer,
        )
        assert intent.sort[0].field == "units_sold"

    def test_non_positive_limit_rejected(self, layer):
        with pytest.raises(ClarificationNeeded):
            validate(QueryIntent(metrics=["revenue"], limit=0), layer)
