"""
Tests for the Black-Scholes pricer.

Run them with:   pytest -v tests/test_black_scholes.py
"""

import math

import pytest

from optionlab.binomial import binomial_price
from optionlab.black_scholes import black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


# ---------------------------------------------------------------------------
# 1. Hand calculation
# ---------------------------------------------------------------------------
# With S=100, K=100, T=1, r=5%, sigma=20%:
#
#   sigma * sqrt(T) = 0.20
#   d1 = ( ln(100/100) + (0.05 + 0.20^2 / 2) * 1 ) / 0.20
#      = ( 0 + 0.07 ) / 0.20                            = 0.35
#   d2 = 0.35 - 0.20                                    = 0.15
#
#   N(0.35) = 0.636831      N(0.15) = 0.559618     (normal table or norm.cdf)
#   e^(-0.05) = 0.951229
#
#   call = 100 * 0.636831 - 100 * 0.951229 * 0.559618  = 10.4506
#   put  = 100 * 0.951229 * (1 - 0.559618) - 100 * (1 - 0.636831)
#        = 41.8905 - 36.3169                            =  5.5735

def test_call_matches_hand_calculation():
    assert black_scholes_price(S, K, T, r, sigma, "call") == pytest.approx(10.4506, abs=1e-4)


def test_put_matches_hand_calculation():
    assert black_scholes_price(S, K, T, r, sigma, "put") == pytest.approx(5.5735, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Put-call parity:  call - put = S - K * e^(-rT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strike", [60.0, 80.0, 100.0, 120.0, 140.0])
@pytest.mark.parametrize("years", [0.1, 1.0, 3.0])
def test_put_call_parity(strike, years):
    call = black_scholes_price(S, strike, years, r, sigma, "call")
    put = black_scholes_price(S, strike, years, r, sigma, "put")
    assert call - put == pytest.approx(S - strike * math.exp(-r * years), abs=1e-9)


# ---------------------------------------------------------------------------
# 3. No-arbitrage bounds
# ---------------------------------------------------------------------------
#   max(S - K*e^(-rT), 0)  <=  call  <=  S
#   max(K*e^(-rT) - S, 0)  <=  put   <=  K*e^(-rT)

@pytest.mark.parametrize("spot", [50.0, 90.0, 100.0, 110.0, 150.0])
def test_prices_stay_within_no_arbitrage_bounds(spot):
    discounted_strike = K * math.exp(-r * T)
    tiny = 1e-9

    call = black_scholes_price(spot, K, T, r, sigma, "call")
    assert max(spot - discounted_strike, 0.0) - tiny <= call <= spot + tiny

    put = black_scholes_price(spot, K, T, r, sigma, "put")
    assert max(discounted_strike - spot, 0.0) - tiny <= put <= discounted_strike + tiny


# ---------------------------------------------------------------------------
# 4. Sensible behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_rises_with_volatility(option_type):
    # More volatility = more chance of a big payoff, while the loss is
    # still capped at zero. So both calls and puts get more expensive.
    low = black_scholes_price(S, K, T, r, 0.10, option_type)
    high = black_scholes_price(S, K, T, r, 0.30, option_type)
    assert high > low


def test_just_before_expiry_price_is_close_to_payoff():
    # With almost no time left, the option is worth what it would pay now.
    almost_zero = 1e-6
    assert black_scholes_price(120.0, K, almost_zero, r, sigma, "call") == pytest.approx(20.0, abs=1e-3)
    assert black_scholes_price(80.0, K, almost_zero, r, sigma, "call") == pytest.approx(0.0, abs=1e-3)
    assert black_scholes_price(80.0, K, almost_zero, r, sigma, "put") == pytest.approx(20.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. The binomial tree converges to Black-Scholes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes(strike, option_type):
    exact = black_scholes_price(S, strike, T, r, sigma, option_type)

    error_10 = abs(binomial_price(S, strike, T, r, sigma, 10, option_type) - exact)
    error_1000 = abs(binomial_price(S, strike, T, r, sigma, 1000, option_type) - exact)

    # A big tree must be close to the formula (within a cent) ...
    assert error_1000 < 0.01
    # ... and much closer than a small tree.
    assert error_1000 < error_10 / 10


# ---------------------------------------------------------------------------
# 6. Bad input
# ---------------------------------------------------------------------------

def test_unknown_option_type_raises_value_error():
    with pytest.raises(ValueError):
        black_scholes_price(S, K, T, r, sigma, option_type="banana")
