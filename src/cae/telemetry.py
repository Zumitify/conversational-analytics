"""Lightweight observability: structured stage logging, timings, counters.

Every pipeline stage runs inside ``stage(...)`` which logs duration with
session/turn context and feeds a per-request timing dict. If OpenTelemetry is
installed, spans are emitted too; otherwise it degrades to logging only.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from contextlib import contextmanager

logger = logging.getLogger("cae")
audit_logger = logging.getLogger("cae.audit")

# Counters for the failure modes in design doc §6 (exposed via /healthz).
counters: Counter[str] = Counter()

try:  # optional dependency
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("cae")
except Exception:  # noqa: BLE001
    _tracer = None


@contextmanager
def stage(name: str, timings: dict[str, int] | None = None, **context):
    started = time.perf_counter()
    span_cm = _tracer.start_as_current_span(name) if _tracer else None
    if span_cm:
        span_cm.__enter__()
    try:
        yield
    except Exception:
        counters[f"{name}_error"] += 1
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if timings is not None:
            timings[name] = elapsed_ms
        logger.info("stage=%s duration_ms=%d %s", name, elapsed_ms,
                    " ".join(f"{k}={v}" for k, v in context.items()))
        if span_cm:
            span_cm.__exit__(None, None, None)


def audit(session_id: str, question: str, sql: str, outcome: str) -> None:
    """Audit log of every NL question, generated SQL, and execution outcome."""
    audit_logger.info(
        "session=%s outcome=%s question=%r sql=%r", session_id, outcome, question, sql
    )
