"""
Tests for the market-data helpers. None of them touches the internet:
they use small hand-made tables so the expected answers are obvious.
"""

import math

import pandas as pd
import pytest

from optionlab.black_scholes import black_scholes_price
from optionlab.market_check import (add_implied_vols, add_model_prices,
                                    add_time_to_expiry, at_the_money_vol,
                                    clean_quotes, pick_expiry)
from optionlab.market_data import parse_fred_csv, to_continuous_rate


def test_parse_fred_csv_skips_blank_holidays():
    text = "observation_date,DGS3MO\n2026-11-10,4.30\n2026-11-11,\n2026-11-12,4.32\n2026-11-13,\n"
    assert parse_fred_csv(text) == ("2026-11-12", 4.32)


def test_continuous_rate_is_log_of_one_plus_yield():
    assert to_continuous_rate(4.32) == pytest.approx(math.log(1.0432))
    # The continuous rate is always a little below the quoted yield.
    assert to_continuous_rate(4.32) < 0.0432


def make_chain():
    return pd.DataFrame({
        "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
        "bid":    [0.00, 2.00, 5.00, 1.00, 3.00],
        "ask":    [0.50, 1.50, 5.20, 3.00, 3.30],
        # row 0: zero bid, row 1: ask below bid, row 2: good,
        # row 3: spread 2.00 on a mid of 2.00 = 100%, row 4: good
    })


def test_clean_quotes_removes_each_kind_of_bad_quote():
    cleaned, removed = clean_quotes(make_chain(), max_relative_spread=0.25)
    assert removed == {"zero or missing bid": 1, "ask below bid": 1, "spread too wide": 1}
    assert list(cleaned["strike"]) == [100.0, 110.0]


def test_clean_quotes_adds_mid_and_relative_spread():
    cleaned, removed = clean_quotes(make_chain())
    assert cleaned["mid"].iloc[0] == pytest.approx(5.10)
    assert cleaned["relative_spread"].iloc[0] == pytest.approx(0.20 / 5.10)


def test_time_to_expiry_in_years_and_expired_rows_dropped():
    quotes = pd.DataFrame({"expiry": ["2026-11-13", "2026-12-13", "2027-11-13"]})
    result = add_time_to_expiry(quotes, "2026-11-13")
    assert list(result["days"]) == [30, 365]          # the expired one is gone
    assert result["T"].iloc[1] == pytest.approx(1.0)


def test_pick_expiry_chooses_the_closest_one():
    quotes = pd.DataFrame({"expiry": ["a", "b", "c"], "days": [7, 35, 98]})
    assert pick_expiry(quotes, 30) == "b"
    assert pick_expiry(quotes, 90) == "c"


def make_synthetic_quotes(sigma):
    """Quotes whose mid prices come from Black-Scholes with a known sigma."""
    rows = []
    for strike in (90.0, 100.0, 110.0):
        for option_type in ("call", "put"):
            rows.append({
                "strike": strike, "option_type": option_type, "T": 0.5,
                "mid": float(black_scholes_price(100.0, strike, 0.5, 0.04, sigma, option_type)),
            })
    return pd.DataFrame(rows)


def test_implied_vols_recover_the_volatility_that_made_the_prices():
    quotes = add_implied_vols(make_synthetic_quotes(0.25), spot=100.0, r=0.04)
    assert list(quotes["implied_vol"]) == pytest.approx([0.25] * 6, abs=1e-6)


def test_at_the_money_vol_uses_the_strike_nearest_spot():
    quotes = add_implied_vols(make_synthetic_quotes(0.25), spot=101.0, r=0.04)
    vol, strike = at_the_money_vol(quotes, spot=101.0)
    assert strike == 100.0


def test_model_error_is_zero_when_the_market_really_is_black_scholes():
    quotes = add_model_prices(make_synthetic_quotes(0.25), spot=100.0, r=0.04, sigma=0.25)
    assert list(quotes["model_error"]) == pytest.approx([0.0] * 6, abs=1e-9)


def test_at_the_money_vol_copes_with_no_usable_quotes():
    empty = pd.DataFrame({"strike": [100.0], "implied_vol": [float("nan")]})
    assert at_the_money_vol(empty, spot=100.0) == (None, None)


def test_keep_near_the_money_drops_far_strikes():
    from optionlab.market_check import keep_near_the_money

    quotes = pd.DataFrame({"strike": [70.0, 80.0, 100.0, 120.0, 130.0]})
    assert list(keep_near_the_money(quotes, spot=100.0)["strike"]) == [80.0, 100.0, 120.0]


def test_parity_gap_is_zero_for_black_scholes_prices():
    from optionlab.market_check import parity_gap

    strike, gap = parity_gap(make_synthetic_quotes(0.25), spot=100.0, r=0.04)
    assert strike == 100.0
    assert gap == pytest.approx(0.0, abs=1e-9)


def test_parity_gap_needs_a_call_and_a_put_at_the_same_strike():
    from optionlab.market_check import parity_gap

    only_calls = make_synthetic_quotes(0.25)
    only_calls = only_calls[only_calls["option_type"] == "call"]
    assert parity_gap(only_calls, spot=100.0, r=0.04) == (None, None)
