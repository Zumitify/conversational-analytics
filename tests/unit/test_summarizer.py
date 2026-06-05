"""Faithfulness check + summarizer behavior (fail-closed on bad numbers)."""

from __future__ import annotations

from cae.llm.client import MockProvider
from cae.models import ColumnMeta, QueryIntent, QueryResult
from cae.postprocessing import Summarizer, check_faithfulness


def result(rows: list[list], metric: str = "revenue") -> QueryResult:
    return QueryResult(
        columns=[ColumnMeta(name="region", role="dimension"),
                 ColumnMeta(name=metric, role="metric")],
        rows=rows,
        row_count=len(rows),
        elapsed_ms=1,
    )


class TestFaithfulness:
    def test_exact_number_passes(self):
        r = result([["West", 1234.56]])
        assert check_faithfulness("West revenue was 1234.56.", r)

    def test_rounded_number_passes(self):
        r = result([["West", 1234.56]])
        assert check_faithfulness("West revenue was about 1,234.6.", r)

    def test_currency_and_commas_pass(self):
        r = result([["West", 1500000.0]])
        assert check_faithfulness("Revenue reached $1,500,000.", r)

    def test_millions_abbreviation_passes(self):
        r = result([["West", 1500000.0]])
        assert check_faithfulness("Revenue reached 1.5 million.", r)

    def test_percentage_from_ratio_passes(self):
        r = result([["West", 0.123]])
        assert check_faithfulness("Growth was 12.3%.", r)

    def test_hallucinated_number_fails(self):
        r = result([["West", 1234.56]])
        assert not check_faithfulness("Revenue was 9999.99.", r)

    def test_row_count_claim_passes(self):
        rows = [[f"r{i}", float(i)] for i in range(15)]
        r = result(rows)
        assert check_faithfulness("There are 15 rows in the result.", r)

    def test_small_prose_integers_skipped(self):
        r = result([["West", 1234.56]])
        # "top 5", "3 regions" — counts in prose, not data claims
        assert check_faithfulness("The top 5 entries span 3 regions.", r)

    def test_empty_summary_passes(self):
        assert check_faithfulness("", result([["West", 1.0]]))


class TestSummarizer:
    def test_unfaithful_summary_dropped(self):
        provider = MockProvider()
        provider.program_text("Revenue exploded to 555555.55 last week.")
        summarizer = Summarizer(provider)
        summary, dropped, _ = summarizer.summarize(
            QueryIntent(metrics=["revenue"]), result([["West", 100.0]]), {}
        )
        assert dropped
        assert summary == ""

    def test_faithful_summary_kept(self):
        provider = MockProvider()
        provider.program_text("West generated 100.0 in revenue.")
        summarizer = Summarizer(provider)
        summary, dropped, _ = summarizer.summarize(
            QueryIntent(metrics=["revenue"]), result([["West", 100.0]]), {}
        )
        assert not dropped
        assert "100.0" in summary

    def test_result_data_is_fenced_in_prompt(self):
        captured = {}

        class SpyProvider(MockProvider):
            def complete(self, *, system, user, max_tokens=512):
                captured["user"] = user
                return super().complete(system=system, user=user,
                                        max_tokens=max_tokens)

        summarizer = Summarizer(SpyProvider())
        summarizer.summarize(
            QueryIntent(metrics=["revenue"]), result([["West", 1.0]]), {}
        )
        assert "<query_result_data>" in captured["user"]
