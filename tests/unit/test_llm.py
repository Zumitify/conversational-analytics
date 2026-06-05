"""LLM layer: mock provider, failover circuit breaker, cost tracking."""

from __future__ import annotations

import pytest

from cae.exceptions import LLMError
from cae.llm.client import (
    CostTracker,
    FailoverProvider,
    MockProvider,
    estimate_cost,
)
from cae.models import QueryIntent, Usage


class TestMockProvider:
    def test_programmed_intent_returned(self):
        provider = MockProvider()
        provider.program_intent("total revenue", {"metrics": ["revenue"]})
        intent, usage = provider.parse(
            system="", user="QUESTION: total revenue", output_model=QueryIntent
        )
        assert intent.metrics == ["revenue"]

    def test_question_extracted_from_context_block(self):
        provider = MockProvider()
        provider.program_intent("follow up", {"metrics": ["revenue"]})
        user = 'FOLLOW-UP CONTEXT: {"metrics": ["x"]}\n\nQUESTION: follow up'
        intent, _ = provider.parse(system="", user=user, output_model=QueryIntent)
        assert intent.metrics == ["revenue"]

    def test_unprogrammed_question_raises(self):
        with pytest.raises(LLMError):
            MockProvider().parse(
                system="", user="QUESTION: mystery", output_model=QueryIntent
            )

    def test_default_text_has_no_digits(self):
        text, _ = MockProvider().complete(system="", user="anything")
        assert not any(c.isdigit() for c in text)


class TestFailover:
    class FailingProvider:
        name = "failing"

        def __init__(self):
            self.calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            raise RuntimeError("provider down")

        def complete(self, **kwargs):
            self.calls += 1
            raise RuntimeError("provider down")

    def test_failover_to_secondary(self):
        primary = self.FailingProvider()
        secondary = MockProvider()
        secondary.program_intent("q", {"metrics": ["revenue"]})
        failover = FailoverProvider([primary, secondary])
        intent, _ = failover.parse(
            system="", user="QUESTION: q", output_model=QueryIntent
        )
        assert intent.metrics == ["revenue"]
        assert primary.calls == 1

    def test_circuit_opens_after_threshold(self):
        primary = self.FailingProvider()
        secondary = MockProvider()
        secondary.program_intent("q", {"metrics": ["revenue"]})
        failover = FailoverProvider([primary, secondary], failure_threshold=2)
        for _ in range(4):
            failover.parse(system="", user="QUESTION: q", output_model=QueryIntent)
        # primary stopped being tried after 2 failures
        assert primary.calls == 2

    def test_all_failing_raises_llm_error(self):
        failover = FailoverProvider([self.FailingProvider()])
        with pytest.raises(LLMError):
            failover.complete(system="", user="hi")


class TestCost:
    def test_known_model_pricing(self):
        # opus 4.8: $5/M in, $25/M out
        assert estimate_cost("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0
        assert estimate_cost("claude-haiku-4-5", 1_000_000, 0) == 1.0

    def test_unknown_model_costs_zero(self):
        assert estimate_cost("mock", 100, 100) == 0.0

    def test_tracker_accumulates(self):
        tracker = CostTracker()
        tracker.record(Usage(input_tokens=100, output_tokens=50, cost_usd=0.01))
        tracker.record(Usage(input_tokens=200, output_tokens=100, cost_usd=0.02))
        snapshot = tracker.snapshot()
        assert snapshot["calls"] == 2
        assert snapshot["input_tokens"] == 300
        assert snapshot["cost_usd"] == pytest.approx(0.03)
