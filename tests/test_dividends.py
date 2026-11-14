"""Tests for the two dividend models (no internet needed)."""

import math

import pytest

from optionlab.dividends import (project_dividends, trailing_dividends,
                                 trailing_yield, value_of_cash_dividends,
                                 value_of_yield)

# A made-up stock paying 1.00 every quarter.
HISTORY = [("2025-03-20", 1.0), ("2025-06-19", 1.0), ("2025-09-18", 1.0),
           ("2025-12-18", 1.0), ("2026-03-19", 1.0), ("2026-06-18", 1.0),
           ("2026-09-17", 1.0)]
TODAY = "2026-11-13"


def test_trailing_dividends_are_the_last_twelve_months_only():
    recent = trailing_dividends(HISTORY, TODAY)
    assert [day for day, amount in recent] == ["2025-12-18", "2026-03-19",
                                               "2026-06-18", "2026-09-17"]


def test_trailing_yield_is_dividends_over_price():
    assert trailing_yield(HISTORY, spot=200.0, as_of=TODAY) == pytest.approx(4.0 / 200.0)


def test_projection_repeats_last_year_52_weeks_later():
    projected = project_dividends(HISTORY, TODAY, years=1)
    assert projected == [("2026-12-17", 1.0), ("2027-03-18", 1.0),
                         ("2027-06-17", 1.0), ("2027-09-16", 1.0)]


def test_cash_value_is_zero_before_the_first_ex_date():
    projected = project_dividends(HISTORY, TODAY)
    assert value_of_cash_dividends(projected, TODAY, "2026-12-16", r=0.04) == 0.0


def test_cash_value_counts_an_ex_date_on_the_expiry_day():
    projected = project_dividends(HISTORY, TODAY)
    value = value_of_cash_dividends(projected, TODAY, "2026-12-17", r=0.04)
    # One dividend of 1.00, 34 days away, discounted at 4%.
    assert value == pytest.approx(1.0 * math.exp(-0.04 * 34 / 365.0))


def test_cash_value_steps_up_at_each_ex_date():
    projected = project_dividends(HISTORY, TODAY)
    one = value_of_cash_dividends(projected, TODAY, "2027-01-15", r=0.04)
    two = value_of_cash_dividends(projected, TODAY, "2027-03-18", r=0.04)
    assert 0.9 < one < 1.0
    assert 1.9 < two < 2.0


def test_yield_value_grows_smoothly_with_time():
    assert value_of_yield(100.0, 0.02, 0.0) == 0.0
    assert value_of_yield(100.0, 0.02, 1.0) == pytest.approx(100.0 * (1 - math.exp(-0.02)))
    assert value_of_yield(100.0, 0.02, 0.5) < value_of_yield(100.0, 0.02, 1.0)
