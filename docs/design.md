# Conversational Analytics Engine

**A natural-language interface to analytical databases, with a real semantic layer, validated SQL generation, and conversational follow-ups.**

---

## 1. Project Overview

### 1.1 What it is

A system that lets a non-technical user ask questions like:

> *"What was week-over-week revenue growth in the Northeast for Q3, broken down by product line?"*

…and get back a validated SQL query, a result table, an auto-chosen chart, and a written summary — with the ability to follow up conversationally (*"now filter that to just enterprise customers"*) without losing context.

### 1.2 Why it's non-trivial

A toy "NL→SQL" demo is ~100 lines of prompt engineering. This project is explicitly *not* that. The hard problems being solved:

1. **Schema grounding** — LLMs hallucinate columns and join keys. A semantic layer with explicit metric/dimension definitions is required to make output trustworthy.
2. **SQL validation** — Generated SQL must be parsed, dialect-checked, and safety-checked (no writes, no cross-schema access, row/time-bound enforced) *before* execution.
3. **Conversational state** — Follow-ups depend on the prior turn's query, filters, and result schema. This requires a structured conversation memory, not just chat history.
4. **Result interpretation** — Picking the right chart and writing a faithful summary requires understanding the *shape* of the result (time series vs. categorical breakdown vs. single metric).
5. **Determinism where it matters** — Same question → same SQL. Achieved through a deterministic post-LLM rewriter and a query cache keyed on canonicalized intent.

### 1.3 Scope (in / out)

**In scope (MVP):**
- DuckDB as the primary execution engine; Postgres adapter as the second target.
- Semantic layer authored in YAML (Cube.js / dbt-metrics style).
- LLM-based intent parsing → structured query IR → SQL generation.
- SQL validation (parse, dialect, safety, semantic).
- Conversation memory with explicit slot tracking.
- Chart recommender (rule-based, not LLM).
- Streamlit UI for chat + table + chart.
- FastAPI backend exposing the same capability via HTTP.
- Sample dataset: TPC-H or a synthetic e-commerce schema (~8 tables).

**Out of scope (explicitly):**
- Writeback / dashboards / scheduled reports.
- Multi-tenant auth.
- Fine-tuning. The whole point is to make a frontier model work *without* fine-tuning by giving it the right scaffolding.
- Real-time / streaming sources.

---

## 2. High-Level Architecture

```
                ┌────────────────────────────────────────────────────┐
                │                   Streamlit UI                     │
                │     chat │ table │ chart │ SQL inspector           │
                └────────────────────────┬───────────────────────────┘
                                         │ HTTP (JSON)
                ┌────────────────────────▼───────────────────────────┐
                │                FastAPI Service                     │
                │  /ask  /sessions/{id}  /healthz  /schema           │
                └────────────────────────┬───────────────────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
   ┌────────▼────────┐         ┌─────────▼─────────┐         ┌────────▼────────┐
   │ Conversation     │         │  Query Pipeline   │         │  Result         │
   │  Manager         │         │  (orchestrator)   │         │  Postprocessor  │
   │  - session store │         │                   │         │  - chart picker │
   │  - slot memory   │         │                   │         │  - summarizer   │
   └────────┬─────────┘         └─────┬──────┬──────┘         └────────┬────────┘
            │                         │      │                         │
            │           ┌─────────────┘      └─────────────┐           │
            │           │                                  │           │
            │  ┌────────▼────────┐                ┌────────▼────────┐  │
            │  │  Intent Parser  │                │ SQL Generator   │  │
            │  │   (LLM call 1)  │                │  (LLM call 2)   │  │
            │  └────────┬────────┘                └────────┬────────┘  │
            │           │                                  │           │
            │  ┌────────▼────────┐                ┌────────▼────────┐  │
            │  │ Semantic Layer  │                │ SQL Validator   │  │
            │  │  - YAML loader  │                │  - sqlglot      │  │
            │  │  - resolver     │                │  - safety       │  │
            │  └────────┬────────┘                └────────┬────────┘  │
            │           │                                  │           │
            │           └─────────────┬────────────────────┘           │
            │                         │                                │
            │                ┌────────▼────────┐                       │
            └────────────────▶  Execution      ◀───────────────────────┘
                             │   Engine        │
                             │ (DuckDB/PG)     │
                             └─────────────────┘
```

The flow is intentionally a **pipeline of small, testable stages**, not one giant agent loop. Each stage has a typed input and output, so failures are debuggable and the whole thing is unit-testable.

---

## 3. Module Breakdown

Nine modules. Each gets its own subsection: responsibility, interface, key implementation notes, and how it's tested.

### 3.1 `semantic_layer/` — The single source of truth about the data

**Responsibility:** Load YAML definitions of tables, columns, metrics, dimensions, joins, and synonyms. Resolve logical names ("revenue", "northeast region") to physical SQL fragments.

**Why this exists first:** Without it, the LLM has to guess column names from raw `CREATE TABLE` statements. With it, the LLM is constrained to a small, curated vocabulary.

**YAML shape (example):**

```yaml
tables:
  orders:
    physical: raw.orders
    primary_key: order_id
    time_column: order_date
    joins:
      - to: customers
        on: orders.customer_id = customers.customer_id
      - to: order_items
        on: orders.order_id = order_items.order_id

metrics:
  revenue:
    expr: "SUM(order_items.quantity * order_items.unit_price)"
    requires: [order_items]
    synonyms: [sales, gross revenue, total sales]
    format: currency_usd

dimensions:
  region:
    expr: "customers.region"
    requires: [customers]
    values: [Northeast, Southeast, Midwest, West]
  product_line:
    expr: "products.line"
    requires: [products]
```

**Public interface:**

```python
class SemanticLayer:
    def list_metrics(self) -> list[MetricDef]: ...
    def list_dimensions(self) -> list[DimensionDef]: ...
    def resolve_metric(self, name: str) -> MetricDef | None: ...
    def resolve_dimension(self, name: str) -> DimensionDef | None: ...
    def required_joins(self, metrics: list[str], dims: list[str]) -> list[Join]: ...
    def to_prompt_context(self) -> str:  # compact view for the LLM
        ...
```

**Testing:** Unit tests on resolution, synonym matching, join-graph correctness. Property test: any subset of metrics+dimensions produces a connected join graph or a clear `UnreachableError`.

---

### 3.2 `intent_parser/` — LLM call #1: NL → structured intent

**Responsibility:** Convert the user's question (plus conversation context) into a typed `QueryIntent` object. No SQL yet.

**Why split intent from SQL:** The intent is small, structured, and easy to validate, log, cache, and diff across turns. SQL is large and dialect-specific. Separating them is the single most important architectural decision in this project.

**Data model:**

```python
class Filter(BaseModel):
    dimension: str
    op: Literal["=", "!=", "in", "not_in", ">", "<", ">=", "<=", "between"]
    values: list[str | int | float]

class TimeRange(BaseModel):
    grain: Literal["day", "week", "month", "quarter", "year"]
    start: date | None
    end: date | None
    relative: str | None  # e.g. "last_4_weeks", "ytd", "q3_2024"

class QueryIntent(BaseModel):
    metrics: list[str]                # logical names from semantic layer
    dimensions: list[str]             # group-by
    filters: list[Filter]
    time_range: TimeRange | None
    comparison: Literal["wow", "mom", "yoy", "none"] = "none"
    sort: list[tuple[str, Literal["asc", "desc"]]] = []
    limit: int | None = None
    explain: bool = False             # user asked "why"
```

**Prompting strategy:**
- System prompt includes the compact semantic-layer view (metrics, dimensions, synonyms, allowed values for low-cardinality dims).
- Few-shot examples covering: simple aggregation, time grain, comparisons, filter list, follow-up turn.
- **Structured output** via the LLM provider's JSON-mode / tool-call API. Never free-text parsing.
- Conversation context passed as the *previous intent* (not raw chat), so the model resolves "filter that to last month" against a real object.

**Validation after the call:**
- Every metric/dimension name must resolve in the semantic layer. Unknown names → `ClarificationNeeded` exception with suggested closest matches (Levenshtein over names + synonyms).
- Filter values for enum dimensions must match allowed values (case-insensitive).
- TimeRange.relative strings parsed by a small dedicated parser (`dateparser` + custom rules for "q3 2024", "last quarter", "ytd").

**Testing:** Golden-file tests on a fixed set of ~40 NL questions → expected `QueryIntent`. Run against the model in CI with a snapshot tolerance (key fields must match exactly; phrasing-only fields ignored).

---

### 3.3 `query_ir/` — Intermediate representation + planning

**Responsibility:** Take a validated `QueryIntent`, expand comparisons (e.g. `wow` = generate two time windows + a delta), and produce a `QueryPlan` that's one step away from SQL.

```python
class QueryPlan(BaseModel):
    select: list[SelectItem]          # metric exprs, dimension exprs, computed
    from_tables: list[str]            # physical names
    joins: list[Join]
    where: list[FilterClause]
    group_by: list[str]
    order_by: list[OrderClause]
    limit: int | None
    ctes: list[CTE] = []              # for comparisons / window calcs
```

This module is **pure Python, no LLM**. It's deterministic. Given the same intent + semantic layer, it always produces the same plan. This is where comparisons (WoW, YoY) get desugared into CTEs with explicit date arithmetic — far more reliable than asking the LLM to write window functions.

**Testing:** Heavy unit testing. This is the keystone of correctness.

---

### 3.4 `sql_generator/` — Plan → SQL string (LLM call #2, optional)

**Responsibility:** Render a `QueryPlan` into dialect-specific SQL.

**Design decision:** For the MVP, this is a **deterministic Jinja-based renderer**, not an LLM call. Templates per dialect (DuckDB, Postgres). The LLM is only re-invoked here as a fallback if the deterministic renderer can't handle a plan feature (rare).

Why not LLM by default: the plan is already fully specified. Asking the LLM to "translate" it just reintroduces hallucination risk. Templates are boring and correct.

**Interface:**

```python
class SQLGenerator(Protocol):
    dialect: str
    def render(self, plan: QueryPlan) -> str: ...
```

Implementations: `DuckDBGenerator`, `PostgresGenerator`. Each owns its own Jinja templates and quoting rules.

---

### 3.5 `sql_validator/` — Parse, safety-check, semantic-check

**Responsibility:** Refuse to execute anything unsafe or wrong before it hits the database.

**Layers of validation:**

1. **Parse** with `sqlglot` (dialect-aware). Reject on any parse error.
2. **Safety:**
   - Statement must be a single `SELECT`.
   - No DDL/DML keywords anywhere (defense in depth — the renderer shouldn't emit them either).
   - No access to schemas outside the allow-list.
   - Mandatory `LIMIT` injected if missing (configurable cap, default 10,000).
   - Mandatory time bound on tables with a `time_column` (configurable lookback cap, default 5 years).
3. **Semantic:**
   - Every referenced table/column exists in the live database catalog.
   - Every join uses a key declared in the semantic layer (no Cartesian products from a hallucinated `ON 1=1`).
4. **Cost estimate** (Postgres adapter only): `EXPLAIN` and reject if estimated rows > threshold.

**Interface:**

```python
class ValidationResult(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    rewritten_sql: str | None  # e.g. with LIMIT injected

class SQLValidator:
    def validate(self, sql: str, dialect: str) -> ValidationResult: ...
```

**Testing:** A corpus of ~50 SQL strings, half malicious / malformed, half valid, with expected verdicts.

---

### 3.6 `execution/` — Run the query

**Responsibility:** Execute validated SQL against the target database and return a typed result.

```python
class QueryResult(BaseModel):
    columns: list[ColumnMeta]   # name, sql_type, semantic_role (metric|dim|time)
    rows: list[tuple]
    row_count: int
    elapsed_ms: int
    truncated: bool

class ExecutionEngine(Protocol):
    def execute(self, sql: str, timeout_s: int) -> QueryResult: ...
```

Implementations: `DuckDBEngine` (in-process), `PostgresEngine` (via `psycopg`). Both enforce timeouts, capture errors, and attach `semantic_role` to each column by looking it up against the `QueryPlan` that produced the SQL (so downstream postprocessing knows which column is the metric).

---

### 3.7 `postprocessing/` — Chart picker + summarizer

**Two submodules:**

**`chart_recommender.py`** — Rule-based, no LLM. Decision tree based on result shape:

| Shape                                          | Chart           |
|------------------------------------------------|-----------------|
| 1 metric, 0 dimensions, 1 row                  | KPI card        |
| 1 metric, 1 time dimension                     | Line chart      |
| 1 metric, 1 categorical dim (≤20 categories)   | Bar chart       |
| 1 metric, 1 categorical dim (>20 categories)   | Bar chart, top-N + "other" |
| 1 metric, 2 dims (1 time, 1 categorical)       | Multi-line chart |
| 1 metric, 2 categorical dims                   | Heatmap or stacked bar |
| 2+ metrics, 1 dim                              | Grouped bar or dual-axis line |
| Comparison query (wow/yoy)                     | Line with delta annotations |

Returns a Vega-Lite spec so the UI can render it without further logic.

**`summarizer.py`** — LLM call #3. Takes the `QueryIntent`, `QueryResult` (downsampled if huge), and chart spec, and produces a 2–4 sentence written summary. Strict prompt: no claims not supported by the result; numbers must match the table; no speculation about causes unless explicitly asked.

Faithfulness check: after generation, extract numeric claims via regex and verify each appears in the result rows. Fail closed (drop the summary, log a warning) if not.

---

### 3.8 `conversation/` — Session and slot memory

**Responsibility:** Track per-session state so follow-ups work.

```python
class Turn(BaseModel):
    user_text: str
    intent: QueryIntent
    plan: QueryPlan
    sql: str
    result_summary: ResultDigest   # column names, row count, key stats — NOT full rows
    chart_spec: dict
    summary_text: str
    created_at: datetime

class Session(BaseModel):
    session_id: str
    turns: list[Turn]
    active_intent: QueryIntent | None  # "current" view the user is iterating on
```

**Storage:** SQLite for the MVP (`sessions.db`), with a simple repository interface so it can be swapped for Redis/Postgres later.

**Follow-up handling:** The intent parser receives `active_intent` as context. The user saying *"now break that down by region"* is parsed as a *delta* on the previous intent — implemented as a separate "intent-edit" prompt path that returns a `QueryIntent` (full, not a patch — easier to validate).

---

### 3.9 `api/` and `ui/` — Service and frontend

**`api/` (FastAPI):**

| Endpoint                          | Method | Purpose                              |
|-----------------------------------|--------|--------------------------------------|
| `/healthz`                        | GET    | Liveness                             |
| `/schema`                         | GET    | Return semantic layer summary        |
| `/sessions`                       | POST   | Create session, return id            |
| `/sessions/{id}/ask`              | POST   | Body: `{question: str}` → full result|
| `/sessions/{id}/history`          | GET    | List turns                           |
| `/sessions/{id}/turns/{n}/sql`    | GET    | Inspect SQL for a past turn          |

Returns include the SQL, the intent (so users can see how their question was understood), the table, the chart spec, and the summary. Transparency is a feature.

**`ui/` (Streamlit):**
- Left pane: chat history.
- Center: current result table + chart.
- Right pane (collapsible): the parsed `QueryIntent` (rendered as a form the user can edit and re-run), and the generated SQL.

The editable intent form is the killer feature for trust — when the model misunderstands, the user fixes the intent directly instead of fighting with prompts.

---

## 4. Data Model Summary

Three core typed objects flow through the system. They are the contracts between modules and must be Pydantic models with full validation.

1. **`QueryIntent`** — what the user wants, in semantic-layer terms.
2. **`QueryPlan`** — how to compute it, in physical terms (still dialect-free).
3. **`QueryResult`** — what came back, with column roles attached.

Plus the configuration/state objects: `SemanticLayer` (loaded from YAML), `Session`/`Turn` (persisted to SQLite), and `ValidationResult`.

---

## 5. Tech Stack & Rationale

| Concern              | Choice                          | Why                                                    |
|----------------------|---------------------------------|--------------------------------------------------------|
| Language             | Python 3.11+                    | Ecosystem; matches your background                     |
| LLM client           | Anthropic + OpenAI, behind protocol | Provider-agnostic; easy A/B                       |
| Structured outputs   | Provider JSON / tool-call mode  | Avoids fragile regex parsing                           |
| Data validation      | Pydantic v2                     | Fast, well-typed, plays well with FastAPI              |
| SQL parsing          | sqlglot                         | Multi-dialect, AST access, transpilation               |
| Templating           | Jinja2                          | Boring and correct for SQL templates                   |
| Local warehouse      | DuckDB                          | Zero-setup, fast on analytical queries                 |
| Remote warehouse     | Postgres via psycopg 3          | Most common starting point in real shops               |
| Web framework        | FastAPI                         | Async, typed, OpenAPI free                             |
| Frontend             | Streamlit                       | Fastest path to a usable demo                          |
| Charts               | Vega-Lite (via Altair / streamlit-vega-lite) | Spec-based, framework-agnostic              |
| Session storage      | SQLite (sqlmodel)               | Zero ops; abstracted for later swap                    |
| Testing              | pytest, hypothesis, syrupy      | Standard + property + snapshot                         |
| Tracing              | OpenTelemetry → console / Jaeger | Each module emits a span; full pipeline visible      |
| Eval harness         | Custom, sits over pytest        | See §8                                                 |

---

## 6. Failure Modes & How They're Handled

This is the bit that separates the project from a demo. Each known failure has a defined behavior.

| Failure                                            | Detection                                | Behavior                                                                 |
|----------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------|
| Intent parser returns unknown metric/dimension     | Semantic layer resolution fails          | Return `ClarificationNeeded` with top-3 suggested names                  |
| Ambiguous question ("show me sales" — which grain?)| Heuristic: missing time grain on time series query | Ask one clarifying question via the UI                         |
| Generated SQL fails parse                          | sqlglot                                  | Retry generator with parse error in prompt (max 2 retries), then fail   |
| SQL fails semantic validation                      | Validator                                | Same: retry with error context, then surface to user                     |
| Query times out                                    | Engine                                   | Cancel; return friendly error; log full SQL                              |
| Result too large                                   | Engine (row count > cap)                 | Auto-inject `LIMIT`, mark `truncated=true`, warn in UI                   |
| Summarizer hallucinates numbers                    | Faithfulness post-check                  | Drop summary, return table+chart only with a note                        |
| Follow-up references something not in prior turn   | Intent diff fails                        | Treat as fresh question, prompt user to confirm                          |
| LLM provider down                                  | Client                                   | Failover to secondary provider; circuit-breaker on repeated failures     |

---

## 7. Security & Safety

- **Read-only DB user** for both DuckDB (file opened read-only) and Postgres (role with `SELECT` only on the analytics schema).
- **Allow-list of schemas/tables** enforced in the validator, not just at the DB level — defense in depth.
- **No raw SQL passthrough** from user input. The user can edit the parsed intent, not the SQL.
- **PII flags** in the semantic layer; columns marked `pii: true` are excluded from being returned to the LLM in either schema context or result samples.
- **Prompt-injection defense:** any content from query results that's passed back into a later prompt (e.g. the summarizer) is wrapped in a delimited block and the system prompt instructs the model to treat it as data, not instructions. Don't claim this is bulletproof — it isn't — but it's the standard mitigation.
- **Rate limiting** at the API layer (per-session and global).
- **Audit log** of every NL question, generated SQL, and execution outcome.

---

## 8. Testing & Evaluation Strategy

Three layers.

**8.1 Unit tests (pytest):** Per-module, mocking neighbors. Coverage target: 80%+ on `semantic_layer`, `query_ir`, `sql_generator`, `sql_validator`. These are deterministic and must be rock-solid.

**8.2 Integration tests:** End-to-end on the bundled sample dataset (DuckDB + TPC-H or e-commerce synthetic). ~30 fixed NL questions → expected (intent, SQL skeleton, result row count). SQL compared by AST equivalence (sqlglot), not string equality.

**8.3 Eval harness (the differentiator):**

A small framework — not a separate product, but a real harness — that runs a labeled question set against the full pipeline and reports:

- **Intent accuracy:** field-by-field match against gold intents.
- **Execution success rate:** % of questions that produce *any* result without error.
- **Result correctness:** for questions with a known answer, exact match on the metric value (with tolerance for float).
- **Latency:** p50/p95 per stage.
- **Cost:** tokens in/out per question, summed.

The harness is runnable as `python -m eval.run --suite=basics` and writes a JSON report plus a markdown diff against the previous run. Used to compare prompt variants, model choices, and semantic-layer changes. This is what makes the project credible as engineering rather than a demo.

---

## 9. Observability

- **Structured logging** (`structlog`) — every stage logs with `session_id`, `turn_id`, `stage`, `duration_ms`.
- **OpenTelemetry spans** for each pipeline stage; a full trace shows: parse → resolve → plan → generate → validate → execute → postprocess.
- **Metrics**: counters for each failure mode in §6, histograms for stage latency, gauge for active sessions.
- **Cost tracking** wired into the LLM client wrapper — tokens and dollars per turn, exposed at `/healthz` aggregate.

---

## 10. Repository Layout

```
conversational-analytics/
├── pyproject.toml
├── README.md
├── docker-compose.yml          # postgres + sample data
├── config/
│   ├── semantic_layer.yaml
│   └── app.yaml
├── src/cae/
│   ├── __init__.py
│   ├── semantic_layer/
│   ├── intent_parser/
│   ├── query_ir/
│   ├── sql_generator/
│   ├── sql_validator/
│   ├── execution/
│   ├── postprocessing/
│   ├── conversation/
│   ├── llm/                    # provider abstraction + retries + cost tracking
│   ├── api/                    # FastAPI app
│   └── pipeline.py             # orchestrator that wires modules together
├── ui/
│   └── streamlit_app.py
├── data/
│   ├── ecommerce_seed.sql
│   └── tpch_loader.py
├── eval/
│   ├── suites/
│   │   ├── basics.yaml
│   │   ├── comparisons.yaml
│   │   └── followups.yaml
│   ├── run.py
│   └── report.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## 11. Milestones (≈ 32 hours, adjust ±)

| # | Milestone                                                       | Hours | Demo-able output                                              |
|---|-----------------------------------------------------------------|-------|---------------------------------------------------------------|
| 1 | Project scaffold, config, semantic layer loader + tests         | 3     | `cae schema` prints loaded metrics/dimensions                 |
| 2 | DuckDB engine + sample dataset loader                           | 2     | `cae query "SELECT 1"` runs end-to-end against DuckDB         |
| 3 | `QueryIntent` model + intent parser (single-turn) with goldens  | 4     | CLI: question → printed intent JSON                           |
| 4 | `QueryPlan` + planner + deterministic SQL generator (DuckDB)    | 5     | CLI: question → SQL → result table                            |
| 5 | SQL validator + safety rules + retry loop                       | 3     | Malicious / malformed SQL rejected with clear errors          |
| 6 | Postprocessor: chart recommender + summarizer + faithfulness    | 3     | CLI prints chart spec + summary; faithfulness tests pass      |
| 7 | Conversation manager + follow-up intent path                    | 3     | Multi-turn CLI session works                                  |
| 8 | FastAPI service + OpenAPI                                       | 2     | `/ask` returns full payload; Swagger UI usable                |
| 9 | Streamlit UI (chat, table, chart, editable intent, SQL pane)    | 3     | Browser demo                                                  |
|10 | Postgres adapter + tests                                        | 2     | Same suite green against Postgres                             |
|11 | Eval harness + initial suite of 30 questions + report           | 3     | `python -m eval.run` produces markdown report                 |
|12 | Observability (logging, tracing, cost tracking) + polish        | 2     | Trace visible in console exporter; cost shown in UI           |

Total: **35 hours**. Realistically 30–40 depending on how much time goes into the eval suite and UI polish.

---

## 12. Stretch Goals (post-MVP, not counted in scope)

- Query result caching keyed on canonicalized intent (huge cost/latency win).
- Semantic-layer authoring UI.
- Snowflake / BigQuery adapters (mostly SQL dialect + auth work).
- "Explain this number" mode — the model walks the user through the SQL and intermediate values.
- A second eval suite of *adversarial* questions (ambiguous, malicious, out-of-scope) to measure refusal quality.
- Replace Streamlit with a React frontend if a real demo audience requires it.

---

## 13. Open Questions to Resolve Before Coding

1. Sample dataset: TPC-H (well-known, dry) vs. synthetic e-commerce (more relatable, more work to generate). Recommendation: **synthetic e-commerce**, ~50k orders, generated by a seeding script — questions feel real and the semantic layer is more interesting.
2. LLM provider for the primary build: which one is your default? The provider abstraction means both work, but goldens and cost numbers will be anchored to whichever you pick.
3. Do you want the editable-intent panel in the UI from day one, or only after milestone 9? Strong recommendation: day one. It's the trust feature.
4. Streamlit vs. a thin React+Tailwind frontend. Streamlit gets to a demo faster; React looks more like a product. For portfolio purposes I'd stay with Streamlit unless the audience is specifically frontend-leaning.

---
