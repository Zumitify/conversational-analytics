"""Unit tests for relative time-range resolution against a frozen today."""

from __future__ import annotations

from datetime import date

import pytest

from cae.exceptions import ClarificationNeeded
from cae.intent_parser.timeparse import resolve_time_range
from cae.models import TimeRange

TODAY = date(2026, 6, 4)  # a Thursday


def rel(text: str) -> tuple[date, date]:
    return resolve_time_range(TimeRange(relative=text), TODAY)


class TestRelative:
    def test_last_month(self):
        assert rel("last_month") == (date(2026, 5, 1), date(2026, 5, 31))

    def test_last_week_is_complete_week(self):
        start, end = rel("last_week")
        assert start == date(2026, 5, 25)  # Monday
        assert end == date(2026, 5, 31)    # Sunday

    def test_last_quarter(self):
        assert rel("last_quarter") == (date(2026, 1, 1), date(2026, 3, 31))

    def test_last_year(self):
        assert rel("last_year") == (date(2025, 1, 1), date(2025, 12, 31))

    def test_ytd(self):
        assert rel("ytd") == (date(2026, 1, 1), TODAY)

    def test_this_quarter(self):
        assert rel("this_quarter") == (date(2026, 4, 1), TODAY)

    def test_mtd(self):
        assert rel("mtd") == (date(2026, 6, 1), TODAY)

    def test_last_n_days(self):
        start, end = rel("last_30_days")
        assert end == TODAY
        assert (end - start).days == 29  # inclusive window of 30 days

    def test_last_n_weeks(self):
        start, end = rel("last_4_weeks")
        assert end == TODAY
        assert (end - start).days == 27

    def test_last_n_months(self):
        start, end = rel("last_6_months")
        assert start == date(2026, 1, 1)
        assert end == TODAY

    def test_last_n_years(self):
        start, end = rel("last_2_years")
        assert start == date(2024, 6, 5)
        assert end == TODAY

    def test_quarter_literal(self):
        assert rel("q3_2024") == (date(2024, 7, 1), date(2024, 9, 30))

    def test_quarter_with_space(self):
        assert rel("Q1 2026") == (date(2026, 1, 1), date(2026, 3, 31))

    def test_bare_year(self):
        assert rel("2025") == (date(2025, 1, 1), date(2025, 12, 31))

    def test_year_month(self):
        assert rel("2025-02") == (date(2025, 2, 1), date(2025, 2, 28))

    def test_yesterday(self):
        assert rel("yesterday") == (date(2026, 6, 3), date(2026, 6, 3))

    def test_normalization_spaces_and_case(self):
        assert rel("Last 4 Weeks") == rel("last_4_weeks")

    def test_unparseable_raises_clarification(self):
        with pytest.raises(ClarificationNeeded):
            rel("during the renaissance")


class TestPrecedence:
    def test_explicit_dates_win(self):
        tr = TimeRange(start=date(2025, 1, 1), end=date(2025, 1, 31), relative="ytd")
        assert resolve_time_range(tr, TODAY) == (date(2025, 1, 1), date(2025, 1, 31))

    def test_start_only_extends_to_today(self):
        tr = TimeRange(start=date(2026, 5, 1))
        assert resolve_time_range(tr, TODAY) == (date(2026, 5, 1), TODAY)

    def test_none_falls_back_to_lookback(self):
        start, end = resolve_time_range(None, TODAY, default_lookback_years=5)
        assert start == date(2021, 6, 4)
        assert end == TODAY
