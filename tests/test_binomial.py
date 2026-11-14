"""
Tests for the binomial tree pricer.

Each test checks one fact that must be true of ANY correct option pricer.
None of them needs Black-Scholes, so they can all pass after stage 1.

Run them with:   pytest -v
"""

import math

import pytest

from optionlab.binomial import binomial_price

# One standard set of inputs reused across tests.
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


# ---------------------------------------------------------------------------
# 1. A 1-step tree, checked against a hand calculation
# ---------------------------------------------------------------------------
# With S=100, K=100, T=1, r=5%, sigma=20% and ONE step (so dt = 1):
#
#   u      = e^(0.20 * sqrt(1))            = 1.221403
#   d      = 1 / u                         = 0.818731
#   p      = (e^0.05 - d) / (u - d)
#          = (1.051271 - 0.818731) / (1.221403 - 0.818731) = 0.577493
#   stock up   = 100 * u = 122.1403   -> call pays 22.1403, put pays 0
#   stock down = 100 * d =  81.8731   -> call pays 0,       put pays 18.1269
#
#   call = e^-0.05 * (0.577493 * 22.1403 + 0.422507 * 0)       = 12.1623
#   put  = e^-0.05 * (0.577493 * 0       + 0.422507 * 18.1269) =  7.2852
#
# Work through this with a calculator before you write any code.

def test_one_step_call_matches_hand_calculation():
    price = binomial_price(S, K, T, r, sigma, steps=1, option_type="call")
    assert price == pytest.approx(12.1623, abs=1e-4)


def test_one_step_put_matches_hand_calculation():
    price = binomial_price(S, K, T, r, sigma, steps=1, option_type="put")
    assert price == pytest.approx(7.2852, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Put-call parity (European options only)
# ---------------------------------------------------------------------------
# Holding a call and selling a put with the same strike pays (stock - K) at
# expiry no matter what happens. That is the same as owning the stock and
# owing K at expiry, which costs S - K*e^(-rT) today. So:
#
#       call - put = S - K * e^(-rT)

@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("steps", [1, 5, 50, 200])
def test_put_call_parity_european(strike, steps):
    call = binomial_price(S, strike, T, r, sigma, steps, "call")
    put = binomial_price(S, strike, T, r, sigma, steps, "put")
    assert call - put == pytest.approx(S - strike * math.exp(-r * T), abs=1e-8)


# ---------------------------------------------------------------------------
# 3. American vs European
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_american_call_equals_european_call_without_dividends(strike):
    # With no dividends it is never worth exercising a call early,
    # so the right to do so adds nothing.
    european = binomial_price(S, strike, T, r, sigma, 200, "call", american=False)
    american = binomial_price(S, strike, T, r, sigma, 200, "call", american=True)
    assert american == pytest.approx(european, abs=1e-10)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_american_put_is_worth_at_least_european_put(strike):
    # Extra rights can never make an option cheaper.
    european = binomial_price(S, strike, T, r, sigma, 200, "put", american=False)
    american = binomial_price(S, strike, T, r, sigma, 200, "put", american=True)
    assert american >= european


def test_american_put_is_strictly_more_valuable_when_in_the_money():
    # For a put that is already in the money (K=110 > S=100) early exercise
    # has real value, so the American price must be clearly higher.
    # This catches a pricer that silently ignores american=True.
    european = binomial_price(S, 110.0, T, r, sigma, 200, "put", american=False)
    american = binomial_price(S, 110.0, T, r, sigma, 200, "put", american=True)
    assert american > european + 0.10


# ---------------------------------------------------------------------------
# 4. No-arbitrage bounds
# ---------------------------------------------------------------------------
# If a price ever left these ranges you could lock in a free profit.
#
#   European call:  max(S - K*e^(-rT), 0)  <=  call  <=  S
#   European put:   max(K*e^(-rT) - S, 0)  <=  put   <=  K*e^(-rT)
#   American put:   max(K - S, 0)          <=  put   <=  K

@pytest.mark.parametrize("spot", [50.0, 90.0, 100.0, 110.0, 150.0])
def test_european_prices_stay_within_no_arbitrage_bounds(spot):
    discounted_strike = K * math.exp(-r * T)
    tiny = 1e-9  # allowance for floating-point rounding

    call = binomial_price(spot, K, T, r, sigma, 100, "call")
    assert max(spot - discounted_strike, 0.0) - tiny <= call <= spot + tiny

    put = binomial_price(spot, K, T, r, sigma, 100, "put")
    assert max(discounted_strike - spot, 0.0) - tiny <= put <= discounted_strike + tiny


@pytest.mark.parametrize("spot", [50.0, 90.0, 100.0, 110.0, 150.0])
def test_american_put_stays_within_no_arbitrage_bounds(spot):
    tiny = 1e-9
    put = binomial_price(spot, K, T, r, sigma, 100, "put", american=True)
    assert max(K - spot, 0.0) - tiny <= put <= K + tiny


# ---------------------------------------------------------------------------
# 5. Bad input
# ---------------------------------------------------------------------------

def test_unknown_option_type_raises_value_error():
    with pytest.raises(ValueError):
        binomial_price(S, K, T, r, sigma, 10, option_type="banana")
