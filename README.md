# Conversational Analytics Engine

**A natural-language interface to analytical databases, with a real semantic layer, validated SQL generation, and conversational follow-ups.**

Ask:

> *"What was week-over-week revenue growth in the Northeast for Q3, broken down by product line?"*

…and get back a validated SQL query, a result table, an auto-chosen chart, and a written summary — then follow up conversationally (*"now filter that to just enterprise customers"*) without losing context.

## Why this isn't a toy NL→SQL demo

A toy demo is ~100 lines of prompt engineering. This project solves the hard problems instead:

| Problem | Solution here |
|---|---|
| **LLMs hallucinate columns and join keys** | A semantic layer (`config/semantic_layer.yaml`) is the only vocabulary the model ever sees. Unknown names → a clarifying question with closest matches, not bad SQL. |
| **Generated SQL can be unsafe or wrong** | Every query is parsed with `sqlglot`, checked against an allow-list + live catalog, joins must match declared join keys (no `ON 1=1`), LIMIT and a time bound are mandatory — *before* execution. |
| **Follow-ups lose context** | Structured conversation memory: each session tracks the *active intent* (a typed object), and follow-ups are parsed as edits to it — not raw chat history. |
| **Charts and summaries drift from the data** | The chart is picked by a rule-based decision tree over result *shape* (no LLM). The summary is LLM-written but every number in it is regex-extracted and verified against the result rows — fail closed, drop the summary. |
| **Same question → different SQL** | Intent → plan → SQL is fully deterministic. The LLM only produces a small, validated `QueryIntent`; everything after that is pure Python + Jinja templates. |

## Architecture

```
question ──▶ Intent Parser (LLM #1, structured output)
                 │  QueryIntent  ◀── conversation memory (active intent)
                 ▼
             Planner (pure Python — desugars WoW/MoM/YoY into LAG windows)
                 │  QueryPlan
                 ▼
             SQL Generator (Jinja templates per dialect — no LLM)
                 │  SQL
                 ▼
             SQL Validator (sqlglot: parse / safety / semantic checks)
                 │  validated SQL (LIMIT injected, capped)
                 ▼
             Execution Engine (DuckDB / Postgres, read-only, timeouts)
                 │  QueryResult (columns carry metric/dimension/time roles)
                 ▼
             Chart Recommender (rule-based → Vega-Lite spec)
             Summarizer (LLM #3 + numeric faithfulness post-check)
```

The flow is a pipeline of small, typed, unit-testable stages — not one giant agent loop. The three contracts between stages (`QueryIntent`, `QueryPlan`, `QueryResult`) are Pydantic models in `src/cae/models.py`.

## Quickstart

Requires Python 3.11+.

```bash
git clone <this repo> && cd conversational-analytics
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic,ui]"

# 1. Build the synthetic e-commerce dataset (~50k orders, deterministic)
cae seed

# 2. Inspect the semantic layer
cae schema

# 3. Sanity-check the engine with raw SQL (validated + read-only)
cae query "SELECT COUNT(*) FROM orders WHERE order_date >= DATE '2025-01-01'"

# 4. Ask questions (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
cae ask "weekly revenue trend in the Northeast this year"
cae repl                       # multi-turn session in the terminal

# 5. Web UI / API
streamlit run ui/streamlit_app.py   # chat + table + chart + editable intent
cae serve                           # FastAPI at http://127.0.0.1:8000/docs
```

No API key? Set `CAE_LLM_PROVIDER=mock` — the deterministic half of the pipeline (planner → SQL → execution → charting) still runs, and the whole test suite + offline eval work without any network access.

## The trust feature: editable intent

Every answer shows *how the question was understood* — the parsed `QueryIntent` — alongside the generated SQL. In the Streamlit UI the intent is an editable form: if the model misreads you, fix the structure directly and re-run (`POST /sessions/{id}/ask_intent`). No prompt fighting, and the user can never inject raw SQL.

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/healthz` | GET | Liveness + LLM cost counters + failure-mode counters |
| `/schema` | GET | Semantic layer summary (PII dimensions excluded) |
| `/sessions` | POST | Create session |
| `/sessions/{id}/ask` | POST | `{question}` → intent + SQL + table + chart + summary |
| `/sessions/{id}/ask_intent` | POST | Re-run a user-edited structured intent (skips LLM) |
| `/sessions/{id}/history` | GET | List turns |
| `/sessions/{id}/turns/{n}/sql` | GET | Inspect SQL + intent for a past turn |

Errors are structured: `422 clarification_needed` (with suggestions), `400 sql_validation_failed`, `502 execution_failed`, `429` rate limit.

## Semantic layer

Authored in YAML (Cube.js / dbt-metrics style): tables + declared joins, metrics with SQL expressions + synonyms, dimensions with allowed enum values and PII flags. The join path for any metric/dimension subset is computed by BFS over the declared graph — an unreachable combination is an explicit `UnreachableJoinError`, never a guessed join.

```yaml
metrics:
  revenue:
    expr: "SUM(order_items.quantity * order_items.unit_price * (1 - order_items.discount))"
    requires: [order_items]
    synonyms: [sales, gross revenue, total sales]
    format: currency_usd
```

Point the engine at your own data by editing `config/semantic_layer.yaml` and `config/app.yaml`.

## Postgres

```bash
docker compose up -d                      # postgres on :5433
pip install -e ".[postgres]"
python scripts/seed_postgres.py           # load the same synthetic dataset
CAE_DB_DIALECT=postgres CAE_POSTGRES_DSN=postgresql://cae:cae@localhost:5433/cae cae ask "revenue by region this year"
```

The Postgres adapter additionally runs an `EXPLAIN`-based row estimate before execution.

## Testing & evaluation

```bash
pytest                          # ~120 unit + integration tests, no network needed
pytest --cov=cae                # coverage

# Eval harness — labeled question suites through the full pipeline
python -m eval.run --suite basics                 # offline: measures the deterministic half
python -m eval.run --suite comparisons
python -m eval.run --suite followups
python -m eval.run --suite basics --provider anthropic   # full pipeline incl. LLM
```

The harness reports **intent accuracy** (field-by-field vs gold intents), **execution success rate**, **result correctness** (independent hand-written check-SQL compared with tolerance), **latency p50/p95 per stage**, and **token cost** — and writes a JSON artifact plus a markdown diff against the previous run (`eval/reports/`). Use it to compare prompt variants, models, and semantic-layer changes.

> With `--provider mock` the parser is programmed with the gold intents, so intent accuracy is trivially 100% — mock mode measures planner → SQL → execution → postprocessing. Use `--provider anthropic` to measure the LLM too.

## Failure modes (each has a defined behavior)

| Failure | Behavior |
|---|---|
| Unknown metric/dimension | `ClarificationNeeded` with top-3 closest names |
| Illegal enum filter value | `ClarificationNeeded` listing allowed values |
| Generated SQL fails validation | Rejected with itemized errors; audited |
| Query timeout | Cancelled (DuckDB interrupt / PG statement_timeout); friendly error |
| Result too large | LIMIT injected/capped, `truncated=true`, warning surfaced |
| Summary hallucinates a number | Summary dropped (fail closed), warning + counter |
| Follow-up references nothing | Parsed as a fresh question |
| LLM provider down | `FailoverProvider` tries secondary; circuit-breaker on repeated failures |

## Security

- Read-only DB connections (DuckDB read-only flag; PG read-only role + `default_transaction_read_only`).
- Table allow-list enforced in the validator *and* by DB permissions (defense in depth).
- No raw SQL passthrough from users — only structured intents.
- PII-flagged columns excluded from LLM context and grouping.
- Result data passed to the summarizer is fenced and declared as data (standard prompt-injection mitigation — not claimed bulletproof).
- Per-session + global rate limiting; audit log of every question, SQL, and outcome.

## Repository layout

```
├── config/
│   ├── app.yaml                  # runtime config (CAE_* env overrides)
│   └── semantic_layer.yaml       # the single source of truth about the data
├── src/cae/
│   ├── models.py                 # QueryIntent / QueryPlan / QueryResult contracts
│   ├── semantic_layer/           # YAML loader, resolution, BFS join graph
│   ├── intent_parser/            # LLM call #1 + validation + time-range parser
│   ├── query_ir/                 # deterministic planner (comparison desugaring)
│   ├── sql_generator/            # Jinja templates per dialect
│   ├── sql_validator/            # sqlglot parse/safety/semantic checks
│   ├── execution/                # DuckDB + Postgres engines
│   ├── postprocessing/           # chart recommender + summarizer + faithfulness
│   ├── conversation/             # SQLite session store, slot memory
│   ├── llm/                      # provider abstraction, failover, cost tracking
│   ├── api/                      # FastAPI service
│   ├── pipeline.py               # the orchestrator
│   └── cli.py                    # cae seed | schema | query | ask | repl | serve
├── ui/streamlit_app.py
├── eval/                         # eval harness + labeled suites + reports
├── tests/                        # unit + integration
└── docker-compose.yml            # Postgres for the second adapter
```

## Design notes & deviations from the original spec

- `QueryIntent.sort` is a list of `{field, direction}` objects rather than tuples — tuples don't round-trip through JSON-schema structured outputs.
- The SQL generator is deterministic-only in the MVP (the design doc allows an LLM fallback for unsupported plan features; no such feature exists yet).
- Comparisons use a `LAG` window over the period-grained base aggregate with the grain pinned to the comparison kind (wow→week, mom→month, yoy→year).
- Relative-date parsing is a small purpose-built parser (`intent_parser/timeparse.py`) instead of `dateparser` — the LLM is constrained to a closed vocabulary, so a dependency wasn't warranted.
- OpenTelemetry is optional: if installed, every stage emits a span; otherwise stage logging + timing dicts still work.

## License

MIT
