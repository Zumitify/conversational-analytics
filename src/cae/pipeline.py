"""The orchestrator: wires all modules into one pipeline of small stages.

parse -> validate intent -> plan -> generate -> validate SQL -> execute ->
chart -> summarize -> persist turn.

Each stage has a typed input/output, so failures are debuggable and each
stage is unit-testable in isolation.
"""

from __future__ import annotations

from datetime import date

from cae import telemetry
from cae.config import AppConfig, load_config
from cae.conversation import SessionStore
from cae.exceptions import SQLValidationError
from cae.execution import ExecutionEngine, make_engine
from cae.intent_parser import IntentParser, validate_intent
from cae.llm.client import CostTracker, LLMProvider, make_provider
from cae.models import (
    AskResponse,
    QueryIntent,
    QueryResult,
    ResultDigest,
    Turn,
    Usage,
)
from cae.postprocessing import Summarizer, recommend_chart
from cae.query_ir import Planner
from cae.semantic_layer import SemanticLayer
from cae.sql_generator import make_generator
from cae.sql_validator import SQLValidator


class Pipeline:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        provider: LLMProvider | None = None,
        engine: ExecutionEngine | None = None,
        layer: SemanticLayer | None = None,
        store: SessionStore | None = None,
        today: date | None = None,
    ) -> None:
        self.config = config or load_config()
        self.layer = layer or SemanticLayer.from_yaml(self.config.semantic_layer_path)
        self.engine = engine or make_engine(self.config)
        self.store = store or SessionStore(self.config.sessions_db_path)
        self.provider = provider or make_provider(
            self.config.llm.provider, self.config.llm.model
        )
        self.today = today  # injectable for reproducible tests/evals

        limits = self.config.limits
        self.parser = IntentParser(
            self.provider, self.layer, max_tokens=self.config.llm.max_tokens
        )
        self.planner = Planner(
            self.layer,
            default_limit=limits.default_limit,
            lookback_years=limits.lookback_years,
        )
        self.generator = make_generator(self.engine.dialect)
        self.validator = SQLValidator(
            self.layer, catalog=self.engine.catalog(), max_rows=limits.max_rows
        )
        self.summarizer = Summarizer(
            self.provider, max_tokens=self.config.llm.summary_max_tokens
        )
        self.cost = CostTracker()

    # -- public API -----------------------------------------------------------

    def create_session(self) -> str:
        return self.store.create_session().session_id

    def ask(self, session_id: str, question: str) -> AskResponse:
        """Full path: NL question -> answer. Uses the session's active intent
        as follow-up context."""
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")

        timings: dict[str, int] = {}
        usage_total = Usage()

        with telemetry.stage("parse_intent", timings, session_id=session_id):
            intent, usage = self.parser.parse(
                question,
                previous_intent=session.active_intent,
                today=self.today,
            )
            usage_total = _add_usage(usage_total, usage)

        return self._run(session_id, question, intent, timings, usage_total)

    def ask_intent(self, session_id: str, intent: QueryIntent) -> AskResponse:
        """Editable-intent path: the user fixed the structured intent directly
        (the trust feature) — skip LLM call #1, validate and run."""
        if self.store.get_session(session_id) is None:
            raise KeyError(f"unknown session: {session_id}")
        validated = validate_intent(intent, self.layer, today=self.today)
        return self._run(session_id, "(edited intent)", validated, {}, Usage())

    # -- internals --------------------------------------------------------------

    def _run(
        self,
        session_id: str,
        question: str,
        intent: QueryIntent,
        timings: dict[str, int],
        usage_total: Usage,
    ) -> AskResponse:
        warnings: list[str] = []

        with telemetry.stage("plan", timings):
            plan = self.planner.plan(intent, today=self.today)

        with telemetry.stage("generate_sql", timings):
            sql = self.generator.render(plan)

        with telemetry.stage("validate_sql", timings):
            verdict = self.validator.validate(sql, dialect=self.engine.dialect)
            if not verdict.ok:
                telemetry.audit(session_id, question, sql, "rejected")
                raise SQLValidationError(
                    "generated SQL failed validation", errors=verdict.errors
                )
            warnings.extend(verdict.warnings)
            if verdict.rewritten_sql:
                sql = verdict.rewritten_sql

        with telemetry.stage("execute", timings):
            try:
                result = self.engine.execute(
                    sql,
                    timeout_s=self.config.limits.query_timeout_s,
                    max_rows=self.config.limits.max_rows,
                )
            except Exception:
                telemetry.audit(session_id, question, sql, "error")
                raise
        result = _attach_roles(result, plan.column_roles())
        if result.truncated:
            warnings.append("result truncated at row cap")

        with telemetry.stage("postprocess", timings):
            chart_spec = recommend_chart(result, intent)
            summary, dropped, usage = self.summarizer.summarize(
                intent, result, chart_spec
            )
            usage_total = _add_usage(usage_total, usage)
            if dropped:
                telemetry.counters["summary_dropped_unfaithful"] += 1
                warnings.append(
                    "summary failed the faithfulness check and was dropped"
                )

        turn = Turn(
            user_text=question,
            intent=intent,
            plan=plan,
            sql=sql,
            result_digest=ResultDigest.from_result(result),
            chart_spec=chart_spec,
            summary_text=summary,
        )
        self.store.append_turn(session_id, turn)
        self.cost.record(usage_total)
        telemetry.audit(session_id, question, sql, "ok")

        return AskResponse(
            session_id=session_id,
            question=question,
            intent=intent,
            sql=sql,
            result=result,
            chart_spec=chart_spec,
            summary=summary,
            summary_dropped=dropped,
            warnings=warnings,
            usage=usage_total,
            stage_timings_ms=timings,
        )


def _attach_roles(result: QueryResult, roles: dict[str, str]) -> QueryResult:
    for column in result.columns:
        column.role = roles.get(column.name, "other")  # type: ignore[assignment]
    return result


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cost_usd=a.cost_usd + b.cost_usd,
    )
