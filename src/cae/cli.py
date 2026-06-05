"""Command-line interface.

    cae seed                  # build the synthetic DuckDB dataset
    cae schema                # print the loaded semantic layer
    cae query "SELECT 1"      # validate + run raw SQL (read-only sandbox)
    cae ask "question"        # one-shot NL question
    cae repl                  # multi-turn session in the terminal
    cae serve                 # run the FastAPI service
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from cae.config import load_config
from cae.exceptions import ClarificationNeeded, SQLValidationError


def _print_table(columns: list[str], rows: list[list], max_rows: int = 25) -> None:
    if not rows:
        print("(no rows)")
        return
    display = [[("" if v is None else str(v)) for v in row] for row in rows[:max_rows]]
    widths = [
        max(len(columns[i]), *(len(r[i]) for r in display))
        for i in range(len(columns))
    ]
    line = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for row in display:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))))
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows)")


def _print_response(response) -> None:
    print("\nINTENT")
    print(json.dumps(response.intent.model_dump(mode="json"), indent=2, default=str))
    print("\nSQL")
    print(response.sql)
    print("\nRESULT")
    _print_table([c.name for c in response.result.columns], response.result.rows)
    print(f"\nCHART: {response.chart_spec.get('chart_type', 'table')}")
    if response.summary:
        print(f"\nSUMMARY\n{response.summary}")
    if response.warnings:
        print(f"\nwarnings: {'; '.join(response.warnings)}")
    if response.usage.cost_usd:
        print(
            f"\ntokens: {response.usage.input_tokens} in / "
            f"{response.usage.output_tokens} out  "
            f"(~${response.usage.cost_usd:.4f})"
        )


def cmd_seed(args) -> int:
    from cae.data.seed import seed_duckdb

    config = load_config(args.config)
    path = args.db or config.database.path
    counts = seed_duckdb(path, n_orders=args.orders, seed=args.seed)
    print(f"seeded {path}:")
    for table, count in counts.items():
        print(f"  {table:<12} {count:>8,}")
    return 0


def cmd_schema(args) -> int:
    from cae.semantic_layer import SemanticLayer

    config = load_config(args.config)
    layer = SemanticLayer.from_yaml(config.semantic_layer_path)
    print(f"fact table: {layer.fact_table}\n")
    print("metrics:")
    for m in layer.list_metrics():
        print(f"  {m.name:<22} {m.expr}")
    print("\ndimensions:")
    for d in layer.list_dimensions():
        pii = "  [PII]" if d.pii else ""
        print(f"  {d.name:<22} {d.expr}{pii}")
    return 0


def cmd_query(args) -> int:
    from cae.execution import make_engine
    from cae.semantic_layer import SemanticLayer
    from cae.sql_validator import SQLValidator

    config = load_config(args.config)
    engine = make_engine(config)
    layer = SemanticLayer.from_yaml(config.semantic_layer_path)
    validator = SQLValidator(layer, catalog=engine.catalog(),
                             max_rows=config.limits.max_rows)
    verdict = validator.validate(args.sql, dialect=engine.dialect)
    if not verdict.ok:
        print("rejected by validator:", file=sys.stderr)
        for error in verdict.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    sql = verdict.rewritten_sql or args.sql
    result = engine.execute(sql, timeout_s=config.limits.query_timeout_s,
                            max_rows=config.limits.max_rows)
    _print_table([c.name for c in result.columns], result.rows)
    print(f"\n{result.row_count} rows in {result.elapsed_ms}ms")
    return 0


def _build_pipeline(args):
    from cae.pipeline import Pipeline

    return Pipeline(load_config(args.config))


def cmd_ask(args) -> int:
    pipeline = _build_pipeline(args)
    session_id = args.session or pipeline.create_session()
    try:
        response = pipeline.ask(session_id, args.question)
    except ClarificationNeeded as e:
        print(f"need clarification: {e.message}")
        if e.suggestions:
            print(f"did you mean: {', '.join(e.suggestions)}?")
        return 1
    except SQLValidationError as e:
        print("query rejected:", "; ".join(e.errors), file=sys.stderr)
        return 1
    print(f"session: {session_id}")
    _print_response(response)
    return 0


def cmd_repl(args) -> int:
    pipeline = _build_pipeline(args)
    session_id = pipeline.create_session()
    print(f"session {session_id} — ask questions, 'exit' to quit")
    while True:
        try:
            question = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", ""):
            break
        try:
            response = pipeline.ask(session_id, question)
            _print_response(response)
        except ClarificationNeeded as e:
            print(f"need clarification: {e.message}")
            if e.suggestions:
                print(f"did you mean: {', '.join(e.suggestions)}?")
        except Exception as e:  # noqa: BLE001 — REPL must not die
            print(f"error: {e}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from cae.api.app import create_app

    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(app, host=config.api.host, port=config.api.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    parser = argparse.ArgumentParser(prog="cae", description=__doc__)
    parser.add_argument("--config", default=None, help="path to app.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="build the synthetic dataset")
    p.add_argument("--db", default=None)
    p.add_argument("--orders", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("schema", help="print the semantic layer")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("query", help="validate + run raw SQL")
    p.add_argument("sql")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("ask", help="ask a natural-language question")
    p.add_argument("question")
    p.add_argument("--session", default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("repl", help="interactive multi-turn session")
    p.set_defaults(func=cmd_repl)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
