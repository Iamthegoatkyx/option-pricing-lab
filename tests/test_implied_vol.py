"""
Tests for the implied volatility solver.

The main test is a round trip: pick a volatility, turn it into a price with
black_scholes_price(), then ask the solver to turn that price back into a
volatility. We must get the number we started with.
"""

import math

import pytest

from optionlab.black_scholes import black_scholes_price
from optionlab.implied_vol import implied_volatility

S, r = 100.0, 0.05


@pytest.mark.parametrize("sigma", [0.05, 0.10, 0.20, 0.40, 0.80, 1.50])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("years", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_round_trip_recovers_the_input_volatility(sigma, strike, years, option_type):
    price = black_scholes_price(S, strike, years, r, sigma, option_type)
    recovered = implied_volatility(price, S, strike, years, r, option_type)
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_call_and_put_with_the_same_strike_imply_the_same_volatility():
    # Both prices came from one sigma, so both must lead back to it.
    call = black_scholes_price(S, 105.0, 0.5, r, 0.30, "call")
    put = black_scholes_price(S, 105.0, 0.5, r, 0.30, "put")
    assert implied_volatility(call, S, 105.0, 0.5, r, "call") == pytest.approx(
        implied_volatility(put, S, 105.0, 0.5, r, "put"), abs=1e-6)


def test_price_below_the_no_arbitrage_floor_has_no_implied_volatility():
    # An at-the-money 1-year call can never be worth less than
    # S - K * e^(-rT) = 4.877. No volatility gives a price of 4.00.
    floor = S - 100.0 * math.exp(-r * 1.0)
    assert floor == pytest.approx(4.877, abs=1e-3)
    assert implied_volatility(4.00, S, 100.0, 1.0, r, "call") is None


def test_price_above_the_ceiling_has_no_implied_volatility():
    # A call can never be worth more than the stock itself.
    assert implied_volatility(101.0, S, 100.0, 1.0, r, "call") is None


def test_price_exactly_on_the_floor_has_no_implied_volatility():
    floor = S - 100.0 * math.exp(-r * 1.0)
    assert implied_volatility(floor, S, 100.0, 1.0, r, "call") is None


def test_higher_price_means_higher_implied_volatility():
    cheap = implied_volatility(8.0, S, 100.0, 1.0, r, "call")
    dear = implied_volatility(12.0, S, 100.0, 1.0, r, "call")
    assert dear > cheap
