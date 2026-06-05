"""Deterministic relative time-range resolution.

The LLM only emits a small vocabulary of relative strings ("last_4_weeks",
"ytd", "q3_2024"); this module turns them into concrete [start, end] dates
against an injectable "today" so tests and evals are reproducible.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from cae.exceptions import ClarificationNeeded
from cae.models import TimeRange

_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7}

_LAST_N = re.compile(r"^last_(\d+)_(days?|weeks?|months?|quarters?|years?)$")
_QUARTER = re.compile(r"^q([1-4])[_ ]?(\d{4})$")
_YEAR = re.compile(r"^(\d{4})$")
_MONTH = re.compile(r"^(\d{4})[_-](\d{2})$")


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _start_of_week(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def _start_of_month(d: date) -> date:
    return d.replace(day=1)


def _start_of_quarter(d: date) -> date:
    return d.replace(month=3 * ((d.month - 1) // 3) + 1, day=1)


def _shift_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    return date(year, month_index % 12 + 1, 1)


def resolve_time_range(
    time_range: TimeRange | None,
    today: date,
    default_lookback_years: int = 5,
) -> tuple[date, date]:
    """Resolve a TimeRange into a concrete inclusive [start, end] pair.

    Precedence: explicit start/end > relative string > default lookback.
    """
    if time_range and time_range.start and time_range.end:
        return time_range.start, time_range.end
    if time_range and time_range.start:
        return time_range.start, today
    if time_range and time_range.relative:
        return _resolve_relative(_normalize(time_range.relative), today)
    # Mandatory default lookback — the validator requires a time bound.
    return today.replace(year=today.year - default_lookback_years), today


def _resolve_relative(rel: str, today: date) -> tuple[date, date]:
    match = _LAST_N.match(rel)
    if match:
        n, unit = int(match.group(1)), match.group(2).rstrip("s")
        if unit in ("day", "week"):
            days = n * _UNIT_DAYS[unit]
            return today - timedelta(days=days - 1), today
        if unit == "month":
            return _shift_months(_start_of_month(today), -(n - 1)), today
        if unit == "quarter":
            return _shift_months(_start_of_quarter(today), -3 * (n - 1)), today
        if unit == "year":
            return today.replace(year=today.year - n) + timedelta(days=1), today

    if rel in ("today",):
        return today, today
    if rel == "yesterday":
        d = today - timedelta(days=1)
        return d, d
    if rel in ("this_week", "wtd"):
        return _start_of_week(today), today
    if rel in ("this_month", "mtd"):
        return _start_of_month(today), today
    if rel in ("this_quarter", "qtd"):
        return _start_of_quarter(today), today
    if rel in ("this_year", "ytd"):
        return today.replace(month=1, day=1), today
    if rel == "last_week":
        end = _start_of_week(today) - timedelta(days=1)
        return end - timedelta(days=6), end
    if rel == "last_month":
        end = _start_of_month(today) - timedelta(days=1)
        return _start_of_month(end), end
    if rel == "last_quarter":
        end = _start_of_quarter(today) - timedelta(days=1)
        return _start_of_quarter(end), end
    if rel == "last_year":
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31)

    match = _QUARTER.match(rel)
    if match:
        q, year = int(match.group(1)), int(match.group(2))
        start = date(year, 3 * (q - 1) + 1, 1)
        end = _shift_months(start, 3) - timedelta(days=1)
        return start, end

    match = _MONTH.match(rel)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        start = date(year, month, 1)
        return start, _shift_months(start, 1) - timedelta(days=1)

    match = _YEAR.match(rel)
    if match:
        year = int(match.group(1))
        return date(year, 1, 1), date(year, 12, 31)

    raise ClarificationNeeded(
        f"I couldn't understand the time range '{rel}'. "
        "Try something like 'last 4 weeks', 'Q3 2024', or 'year to date'."
    )
