"""Provider-agnostic LLM client layer.

- ``AnthropicProvider`` — primary. Uses the official SDK; structured outputs
  go through ``client.messages.parse`` with a Pydantic model, so intent JSON
  is validated at the API layer (never free-text parsing).
- ``MockProvider``     — deterministic fake for tests/offline eval.
- ``FailoverProvider`` — tries providers in order with a simple circuit
  breaker on repeated failures.
- ``CostTracker``      — tokens and dollars per call, aggregated.

The SDK already retries 429/5xx with exponential backoff (max_retries).
"""

from __future__ import annotations

import threading
from typing import Protocol, TypeVar

from pydantic import BaseModel

from cae.exceptions import LLMError
from cae.models import Usage

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output) — cached from platform.claude.com pricing.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES.get(model)
    if not prices:
        return 0.0
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


class CostTracker:
    """Thread-safe accumulator of tokens/dollars across the process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def record(self, usage: Usage) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
            self.cost_usd += usage.cost_usd

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
            }


class LLMProvider(Protocol):
    name: str

    def parse(
        self, *, system: str, user: str, output_model: type[T], max_tokens: int = 2048
    ) -> tuple[T, Usage]:
        """Structured output: returns a validated instance of output_model."""
        ...

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> tuple[str, Usage]:
        """Plain text completion."""
        ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", max_retries: int = 2) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def _usage(self, response) -> Usage:
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
        )

    def parse(
        self, *, system: str, user: str, output_model: type[T], max_tokens: int = 2048
    ) -> tuple[T, Usage]:
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                # System prompt (semantic-layer context + few-shots) is stable
                # across questions — cache it.
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
        )
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("structured output parsing returned no object")
        return parsed, self._usage(response)

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> tuple[str, Usage]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text, self._usage(response)


class MockProvider:
    """Deterministic fake provider.

    Structured calls are answered from a programmed mapping keyed on the raw
    question (the parser embeds it as a ``QUESTION: ...`` line). Text calls
    return a programmed string or a digit-free default so the summarizer's
    faithfulness check passes trivially.
    """

    name = "mock"
    model = "mock"

    def __init__(self) -> None:
        self._intents: dict[str, dict] = {}
        self._texts: list[str] = []

    def program_intent(self, question: str, intent: dict | BaseModel) -> None:
        payload = intent.model_dump(mode="json") if isinstance(intent, BaseModel) else intent
        self._intents[question.strip().lower()] = payload

    def program_text(self, text: str) -> None:
        self._texts.append(text)

    @staticmethod
    def _extract_question(user: str) -> str:
        for line in reversed(user.splitlines()):
            if line.startswith("QUESTION:"):
                return line[len("QUESTION:"):].strip().lower()
        return user.strip().lower()

    def parse(
        self, *, system: str, user: str, output_model: type[T], max_tokens: int = 2048
    ) -> tuple[T, Usage]:
        question = self._extract_question(user)
        if question not in self._intents:
            raise LLMError(f"MockProvider has no programmed intent for: {question!r}")
        return output_model.model_validate(self._intents[question]), Usage()

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> tuple[str, Usage]:
        if self._texts:
            return self._texts.pop(0), Usage()
        return "See the table and chart for the full breakdown.", Usage()


class FailoverProvider:
    """Try providers in order; open a circuit after repeated failures."""

    name = "failover"

    def __init__(self, providers: list[LLMProvider], failure_threshold: int = 3) -> None:
        if not providers:
            raise ValueError("FailoverProvider needs at least one provider")
        self._providers = providers
        self._failures = {id(p): 0 for p in providers}
        self._threshold = failure_threshold

    def _call(self, method: str, **kwargs):
        last_error: Exception | None = None
        for provider in self._providers:
            if self._failures[id(provider)] >= self._threshold:
                continue  # circuit open
            try:
                result = getattr(provider, method)(**kwargs)
                self._failures[id(provider)] = 0
                return result
            except Exception as exc:  # noqa: BLE001
                self._failures[id(provider)] += 1
                last_error = exc
        raise LLMError(f"all providers failed: {last_error}")

    def parse(self, **kwargs):
        return self._call("parse", **kwargs)

    def complete(self, **kwargs):
        return self._call("complete", **kwargs)


def make_provider(provider: str, model: str) -> LLMProvider:
    if provider == "anthropic":
        return AnthropicProvider(model=model)
    if provider == "mock":
        return MockProvider()
    raise ValueError(f"unknown LLM provider: {provider}")
