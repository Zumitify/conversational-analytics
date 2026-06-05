"""Report writer: JSON artifact + markdown diff against the previous run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HEADLINE_METRICS = [
    "execution_success_rate",
    "intent_accuracy_mean",
    "intent_exact_match_rate",
    "result_correct_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "tokens_in_total",
    "tokens_out_total",
]


def _previous_report(reports_dir: Path, suite: str) -> dict | None:
    candidates = sorted(reports_dir.glob(f"{suite}_*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def write_reports(summary: dict, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    previous = _previous_report(reports_dir, summary["suite"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    json_path = reports_dir / f"{summary['suite']}_{stamp}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    md_path = reports_dir / f"{summary['suite']}_{stamp}.md"
    md_path.write_text(render_markdown(summary, previous))
    return json_path, md_path


def render_markdown(summary: dict, previous: dict | None) -> str:
    lines = [
        f"# Eval report — `{summary['suite']}`",
        "",
        f"- run at: {summary['run_at']}",
        f"- provider/model: {summary['provider']} / {summary.get('model', '?')}",
        f"- questions: {summary['n_questions']} (frozen today = {summary['today']})",
        "",
        "## Headline metrics",
        "",
        "| metric | value | previous | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in HEADLINE_METRICS:
        value = summary.get(metric, 0)
        prev_value = previous.get(metric) if previous else None
        if prev_value is None:
            delta = "—"
            prev_str = "—"
        else:
            diff = (value or 0) - (prev_value or 0)
            delta = f"{diff:+.4f}" if isinstance(value, float) else f"{diff:+d}"
            prev_str = str(prev_value)
        lines.append(f"| {metric} | {value} | {prev_str} | {delta} |")

    lines += ["", "## Per-question results", "",
              "| id | ok | intent | correct | latency ms | notes |",
              "|---|---|---:|---|---:|---|"]
    for record in summary["results"]:
        notes = record.get("error") or "; ".join(
            record.get("value_failures", []) + record.get("intent_mismatches", [])
        ) or ""
        lines.append(
            f"| {record['id']} | {'✅' if record['ok'] else '❌'} "
            f"| {record.get('intent_score', 0):.2f} "
            f"| {'✅' if record.get('correct') else '❌'} "
            f"| {record.get('latency_ms', 0)} | {notes[:120]} |"
        )
    lines.append("")
    return "\n".join(lines)
