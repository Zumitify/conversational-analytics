"""Typed exceptions used as control flow between pipeline stages."""

from __future__ import annotations


class CAEError(Exception):
    """Base class for all engine errors."""


class ClarificationNeeded(CAEError):
    """The user's question cannot be resolved without asking back.

    Carries suggestions (e.g. closest metric/dimension names) so the UI
    can render a useful clarifying question.
    """

    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.suggestions = suggestions or []


class UnreachableJoinError(CAEError):
    """The requested metrics/dimensions cannot be connected by declared joins."""


class IntentValidationError(CAEError):
    """The parsed intent references unknown names or illegal values."""


class PlanningError(CAEError):
    """The planner could not turn a valid intent into a QueryPlan."""


class SQLValidationError(CAEError):
    """Generated SQL failed parse / safety / semantic validation."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or [message]


class ExecutionError(CAEError):
    """The database rejected or timed out on a validated query."""


class LLMError(CAEError):
    """All configured LLM providers failed."""
