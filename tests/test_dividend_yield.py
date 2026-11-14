"""
Tests for the dividend yield q in binomial_price() and black_scholes_price().

These fail until you use q inside both functions. The formulas are in the
"Adding a dividend yield q" section of each function's docstring.

Run them with:   pytest -v tests/test_dividend_yield.py
"""

import math

import pytest

from optionlab.binomial import binomial_price
from optionlab.black_scholes import black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
q = 0.03   # a 3% dividend yield


# ---------------------------------------------------------------------------
# 1. Black-Scholes hand calculation with q = 3%
# ---------------------------------------------------------------------------
#   d1 = ( ln(100/100) + (0.05 - 0.03 + 0.20^2 / 2) * 1 ) / 0.20
#      = ( 0 + 0.04 ) / 0.20                              = 0.20
#   d2 = 0.20 - 0.20                                      = 0.00
#
#   N(0.20) = 0.579260     N(0.00) = 0.5
#   e^(-0.03) = 0.970446   e^(-0.05) = 0.951229
#
#   call = 100 * 0.970446 * 0.579260 - 100 * 0.951229 * 0.5
#        = 56.2140 - 47.5615                               = 8.6525
#   put  = 100 * 0.951229 * 0.5 - 100 * 0.970446 * (1 - 0.579260)
#        = 47.5615 - 40.8306                               = 6.7309

def test_black_scholes_call_with_dividend_yield_matches_hand_calculation():
    price = black_scholes_price(S, K, T, r, sigma, "call", q=q)
    assert price == pytest.approx(8.6525, abs=1e-4)


def test_black_scholes_put_with_dividend_yield_matches_hand_calculation():
    price = black_scholes_price(S, K, T, r, sigma, "put", q=q)
    assert price == pytest.approx(6.7309, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. One-step binomial hand calculation with q = 3%
# ---------------------------------------------------------------------------
#   u = 1.221403,  d = 0.818731            (unchanged: they do not use q)
#   p = (e^(0.05 - 0.03) - d) / (u - d)
#     = (1.020201 - 0.818731) / 0.402672                  = 0.500334
#   (it was 0.577493 without dividends)
#
#   call = e^-0.05 * (0.500334 * 22.1403)                 = 10.5373
#   put  = e^-0.05 * (0.499666 * 18.1269)                 =  8.6157

def test_one_step_binomial_call_with_dividend_yield():
    price = binomial_price(S, K, T, r, sigma, steps=1, option_type="call", q=q)
    assert price == pytest.approx(10.5373, abs=1e-4)


def test_one_step_binomial_put_with_dividend_yield():
    price = binomial_price(S, K, T, r, sigma, steps=1, option_type="put", q=q)
    assert price == pytest.approx(8.6157, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Leaving q out must give exactly the old prices
# ---------------------------------------------------------------------------

def test_default_q_is_zero_so_old_prices_are_unchanged():
    assert black_scholes_price(S, K, T, r, sigma, "call") == pytest.approx(10.4506, abs=1e-4)
    assert binomial_price(S, K, T, r, sigma, 1, "call") == pytest.approx(12.1623, abs=1e-4)
    assert black_scholes_price(S, K, T, r, sigma, "call", q=0.0) == \
        black_scholes_price(S, K, T, r, sigma, "call")


# ---------------------------------------------------------------------------
# 4. Direction: dividends make calls cheaper and puts dearer
# ---------------------------------------------------------------------------
# Dividends pull the stock price down over time, which hurts a call
# (needs the price high) and helps a put (needs the price low).

def test_dividends_lower_call_prices_and_raise_put_prices():
    assert black_scholes_price(S, K, T, r, sigma, "call", q=q) < \
        black_scholes_price(S, K, T, r, sigma, "call")
    assert black_scholes_price(S, K, T, r, sigma, "put", q=q) > \
        black_scholes_price(S, K, T, r, sigma, "put")
    assert binomial_price(S, K, T, r, sigma, 100, "call", q=q) < \
        binomial_price(S, K, T, r, sigma, 100, "call")
    assert binomial_price(S, K, T, r, sigma, 100, "put", q=q) > \
        binomial_price(S, K, T, r, sigma, 100, "put")


# ---------------------------------------------------------------------------
# 5. Put-call parity with dividends:  call - put = S*e^(-qT) - K*e^(-rT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_put_call_parity_with_dividends_black_scholes(strike):
    call = black_scholes_price(S, strike, T, r, sigma, "call", q=q)
    put = black_scholes_price(S, strike, T, r, sigma, "put", q=q)
    expected = S * math.exp(-q * T) - strike * math.exp(-r * T)
    assert call - put == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("steps", [1, 50, 200])
def test_put_call_parity_with_dividends_binomial(strike, steps):
    call = binomial_price(S, strike, T, r, sigma, steps, "call", q=q)
    put = binomial_price(S, strike, T, r, sigma, steps, "put", q=q)
    expected = S * math.exp(-q * T) - strike * math.exp(-r * T)
    # Looser than the Black-Scholes test: the tree's stock grows at
    # e^((r-q)*dt) per step, which matches S*e^(-qT) exactly, but many
    # multiplications add a little rounding.
    assert call - put == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# 6. The tree still converges to the formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes_with_dividends(strike, option_type):
    exact = black_scholes_price(S, strike, T, r, sigma, option_type, q=q)
    error_10 = abs(binomial_price(S, strike, T, r, sigma, 10, option_type, q=q) - exact)
    error_1000 = abs(binomial_price(S, strike, T, r, sigma, 1000, option_type, q=q) - exact)
    assert error_1000 < 0.01
    assert error_1000 < error_10 / 10


# ---------------------------------------------------------------------------
# 7. With dividends, exercising a call early CAN be worth it
# ---------------------------------------------------------------------------
# Stage 1 showed American call = European call. That relied on there being
# no dividends. Exercising early captures the dividends, so with a big
# enough yield the American call is worth clearly more.

def test_american_call_is_worth_more_than_european_with_high_dividends():
    european = binomial_price(120.0, K, T, r, sigma, 200, "call", american=False, q=0.08)
    american = binomial_price(120.0, K, T, r, sigma, 200, "call", american=True, q=0.08)
    assert american > european + 0.50
