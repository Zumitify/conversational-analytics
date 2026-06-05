"""FastAPI service exposing the pipeline over HTTP.

Responses include the SQL, the parsed intent, the table, the chart spec and
the summary — transparency is a feature.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cae import telemetry
from cae.config import AppConfig
from cae.exceptions import ClarificationNeeded, ExecutionError, SQLValidationError
from cae.models import AskResponse, QueryIntent
from cae.pipeline import Pipeline


class AskRequest(BaseModel):
    question: str


class IntentRequest(BaseModel):
    intent: QueryIntent


class RateLimiter:
    """Sliding-window limiter: per-session and global request caps."""

    def __init__(self, per_session: int = 30, global_cap: int = 120, window_s: int = 60):
        self.per_session = per_session
        self.global_cap = global_cap
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        now = time.monotonic()
        for bucket_key, cap in ((key, self.per_session), ("__global__", self.global_cap)):
            bucket = self._hits[bucket_key]
            while bucket and now - bucket[0] > self.window_s:
                bucket.popleft()
            if len(bucket) >= cap:
                return False
            bucket.append(now)
        return True


def create_app(config: AppConfig | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    app = FastAPI(
        title="Conversational Analytics Engine",
        version="0.1.0",
        description="Natural-language interface to analytical databases.",
    )
    pipe = pipeline or Pipeline(config)
    limiter = RateLimiter()

    @app.exception_handler(ClarificationNeeded)
    async def _clarification(_: Request, exc: ClarificationNeeded):
        return JSONResponse(
            status_code=422,
            content={
                "error": "clarification_needed",
                "message": exc.message,
                "suggestions": exc.suggestions,
            },
        )

    @app.exception_handler(SQLValidationError)
    async def _sql_invalid(_: Request, exc: SQLValidationError):
        return JSONResponse(
            status_code=400,
            content={"error": "sql_validation_failed", "details": exc.errors},
        )

    @app.exception_handler(ExecutionError)
    async def _exec_error(_: Request, exc: ExecutionError):
        return JSONResponse(
            status_code=502,
            content={"error": "execution_failed", "message": str(exc)},
        )

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "llm_cost": pipe.cost.snapshot(),
            "failure_counters": dict(telemetry.counters),
        }

    @app.get("/schema")
    def schema():
        layer = pipe.layer
        return {
            "fact_table": layer.fact_table,
            "metrics": [
                {"name": m.name, "description": m.description,
                 "synonyms": m.synonyms, "format": m.format}
                for m in layer.list_metrics()
            ],
            "dimensions": [
                {"name": d.name, "values": d.values, "synonyms": d.synonyms}
                for d in layer.list_dimensions() if not d.pii
            ],
        }

    @app.post("/sessions", status_code=201)
    def create_session():
        return {"session_id": pipe.create_session()}

    @app.post("/sessions/{session_id}/ask", response_model=AskResponse)
    def ask(session_id: str, body: AskRequest):
        if not limiter.check(session_id):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        try:
            return pipe.ask(session_id, body.question)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown session")

    @app.post("/sessions/{session_id}/ask_intent", response_model=AskResponse)
    def ask_intent(session_id: str, body: IntentRequest):
        """Editable-intent path: re-run a user-corrected structured intent."""
        if not limiter.check(session_id):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        try:
            return pipe.ask_intent(session_id, body.intent)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown session")

    @app.get("/sessions/{session_id}/history")
    def history(session_id: str):
        session = pipe.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return {
            "session_id": session_id,
            "turns": [
                {
                    "index": i,
                    "question": t.user_text,
                    "summary": t.summary_text,
                    "row_count": t.result_digest.row_count,
                    "created_at": t.created_at,
                }
                for i, t in enumerate(session.turns)
            ],
        }

    @app.get("/sessions/{session_id}/turns/{n}/sql")
    def turn_sql(session_id: str, n: int):
        session = pipe.store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        if n < 0 or n >= len(session.turns):
            raise HTTPException(status_code=404, detail="unknown turn")
        turn = session.turns[n]
        return {
            "sql": turn.sql,
            "intent": turn.intent,
            "question": turn.user_text,
        }

    return app
