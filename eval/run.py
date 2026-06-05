"""Eval harness: run a labeled question suite through the full pipeline.

    python -m eval.run --suite basics                 # offline (mock LLM)
    python -m eval.run --suite basics --provider anthropic

Reports:
- intent accuracy (field-by-field vs gold intents),
- execution success rate,
- result correctness (row bounds + independent check-SQL values),
- latency p50/p95 per stage,
- token cost.

Writes a JSON report plus a markdown diff against the previous run of the
same suite (eval/reports/).

Note on providers: with --provider mock the parser is programmed with the
gold intents, so intent accuracy is trivially 100% and the eval measures the
*deterministic* half (planner -> SQL -> execution -> postprocessing). With
--provider anthropic the full pipeline, including the LLM, is measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from cae.config import load_config
from cae.data.seed import seed_duckdb
from cae.execution import DuckDBEngine
from cae.llm.client import MockProvider, make_provider
from cae.models import QueryIntent
from cae.pipeline import Pipeline

from eval.report import write_reports

SUITES_DIR = Path(__file__).parent / "suites"
REPORTS_DIR = Path(__file__).parent / "reports"
EVAL_DB = Path(__file__).parent / ".eval.duckdb"


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[index]


def intent_accuracy(parsed: QueryIntent, gold: dict) -> tuple[float, list[str]]:
    """Field-by-field match; phrasing-only fields are ignored."""
    gold_intent = QueryIntent.model_validate(gold)
    mismatches: list[str] = []
    checks = {
        "metrics": (sorted(parsed.metrics), sorted(gold_intent.metrics)),
        "dimensions": (sorted(parsed.dimensions), sorted(gold_intent.dimensions)),
        "comparison": (parsed.comparison, gold_intent.comparison),
        "limit": (parsed.limit, gold_intent.limit),
        "filters": (
            sorted((f.dimension, f.op, tuple(map(str, f.values))) for f in parsed.filters),
            sorted((f.dimension, f.op, tuple(map(str, f.values))) for f in gold_intent.filters),
        ),
        "time.relative": (
            parsed.time_range.relative if parsed.time_range else None,
            gold_intent.time_range.relative if gold_intent.time_range else None,
        ),
        "time.grain": (
            parsed.time_range.grain if parsed.time_range else None,
            gold_intent.time_range.grain if gold_intent.time_range else None,
        ),
    }
    matched = 0
    for field, (got, expected) in checks.items():
        if got == expected:
            matched += 1
        else:
            mismatches.append(f"{field}: got {got!r}, expected {expected!r}")
    return matched / len(checks), mismatches


def run_suite(suite_name: str, provider_name: str, n_orders: int) -> dict:
    suite = yaml.safe_load((SUITES_DIR / f"{suite_name}.yaml").read_text())
    today = date.fromisoformat(str(suite["today"]))

    # Deterministic dataset anchored to the suite's frozen "today".
    seed_duckdb(EVAL_DB, n_orders=n_orders, end_date=today)

    if provider_name == "mock":
        provider = MockProvider()
        for item in suite["items"]:
            provider.program_intent(item["question"], item["gold_intent"])
    else:
        config = load_config()
        provider = make_provider(provider_name, config.llm.model)

    config = load_config()
    config.database.dialect = "duckdb"
    config.database.path = str(EVAL_DB)
    config.sessions_db_path = str(Path(__file__).parent / ".eval_sessions.db")
    Path(config.sessions_db_path).unlink(missing_ok=True)

    pipeline = Pipeline(config, provider=provider, today=today)
    check_engine = DuckDBEngine(str(EVAL_DB))

    results = []
    session_id = pipeline.create_session()
    for item in suite["items"]:
        if not item.get("follow_up", False):
            # Fresh context unless the item is explicitly a follow-up.
            session_id = pipeline.create_session()

        record: dict = {"id": item["id"], "question": item["question"]}
        started = time.perf_counter()
        try:
            response = pipeline.ask(session_id, item["question"])
            record["ok"] = True
            record["row_count"] = response.result.row_count
            record["timings_ms"] = response.stage_timings_ms
            record["tokens_in"] = response.usage.input_tokens
            record["tokens_out"] = response.usage.output_tokens
            score, mismatches = intent_accuracy(response.intent, item["gold_intent"])
            record["intent_score"] = score
            record["intent_mismatches"] = mismatches

            expect = item.get("expect", {})
            failures: list[str] = []
            if "min_rows" in expect and response.result.row_count < expect["min_rows"]:
                failures.append(
                    f"row_count {response.result.row_count} < min {expect['min_rows']}"
                )
            if "max_rows" in expect and response.result.row_count > expect["max_rows"]:
                failures.append(
                    f"row_count {response.result.row_count} > max {expect['max_rows']}"
                )
            if "check_sql" in expect:
                # Independent, hand-written SQL — the ground truth value.
                check = check_engine.execute(expect["check_sql"], max_rows=1)
                expected_value = check.rows[0][0]
                column = expect.get("check_column", response.intent.metrics[0])
                names = [c.name for c in response.result.columns]
                got_value = response.result.rows[0][names.index(column)]
                tolerance = float(expect.get("tolerance", 1e-6))
                if expected_value is None or got_value is None or (
                    abs(float(got_value) - float(expected_value))
                    > tolerance * max(1.0, abs(float(expected_value)))
                ):
                    failures.append(
                        f"value check: got {got_value}, expected {expected_value}"
                    )
            record["value_failures"] = failures
            record["correct"] = not failures
        except Exception as exc:  # noqa: BLE001 — eval must finish
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["intent_score"] = 0.0
            record["correct"] = False
        record["latency_ms"] = int((time.perf_counter() - started) * 1000)
        results.append(record)

    check_engine.close()
    pipeline.engine.close()

    ok = [r for r in results if r["ok"]]
    stage_latencies: dict[str, list[int]] = {}
    for record in ok:
        for stage_name, ms in record.get("timings_ms", {}).items():
            stage_latencies.setdefault(stage_name, []).append(ms)

    summary = {
        "suite": suite_name,
        "provider": provider_name,
        "model": getattr(provider, "model", provider_name),
        "today": str(today),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(results),
        "execution_success_rate": round(len(ok) / len(results), 4) if results else 0,
        "intent_accuracy_mean": round(
            statistics.mean(r["intent_score"] for r in results), 4
        ) if results else 0,
        "intent_exact_match_rate": round(
            sum(1 for r in results if r["intent_score"] == 1.0) / len(results), 4
        ) if results else 0,
        "result_correct_rate": round(
            sum(1 for r in results if r.get("correct")) / len(results), 4
        ) if results else 0,
        "latency_p50_ms": _percentile([r["latency_ms"] for r in results], 0.5),
        "latency_p95_ms": _percentile([r["latency_ms"] for r in results], 0.95),
        "stage_latency_p50_ms": {
            s: _percentile(v, 0.5) for s, v in stage_latencies.items()
        },
        "tokens_in_total": sum(r.get("tokens_in", 0) for r in results),
        "tokens_out_total": sum(r.get("tokens_out", 0) for r in results),
        "results": results,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CAE eval harness")
    parser.add_argument("--suite", default="basics")
    parser.add_argument("--provider", default="mock", choices=["mock", "anthropic"])
    parser.add_argument("--orders", type=int, default=20_000)
    args = parser.parse_args(argv)

    summary = run_suite(args.suite, args.provider, args.orders)
    json_path, md_path = write_reports(summary, REPORTS_DIR)

    print(f"suite={summary['suite']} provider={summary['provider']}")
    print(f"  execution success : {summary['execution_success_rate']:.0%}")
    print(f"  intent accuracy   : {summary['intent_accuracy_mean']:.0%}"
          f" (exact {summary['intent_exact_match_rate']:.0%})")
    print(f"  result correct    : {summary['result_correct_rate']:.0%}")
    print(f"  latency p50/p95   : {summary['latency_p50_ms']}ms / {summary['latency_p95_ms']}ms")
    print(f"  tokens in/out     : {summary['tokens_in_total']} / {summary['tokens_out_total']}")
    print(f"  report            : {json_path}")
    print(f"  markdown          : {md_path}")
    failed = [r for r in summary["results"] if not r.get("correct")]
    if failed:
        print("\nfailed items:")
        for record in failed:
            reason = record.get("error") or "; ".join(
                record.get("value_failures", []) + record.get("intent_mismatches", [])
            )
            print(f"  - {record['id']}: {reason}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
