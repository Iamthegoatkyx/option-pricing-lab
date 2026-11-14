"""
Tests for the Greeks: each formula must agree with a finite-difference
estimate made by nudging an input to black_scholes_price().
"""

import pytest

from optionlab import greeks

r, sigma = 0.05, 0.20

# A spread of situations: in / at / out of the money, short and long dated.
CASES = [
    # (S, K, T)
    (100.0, 100.0, 1.0),
    (80.0, 100.0, 1.0),
    (120.0, 100.0, 1.0),
    (100.0, 100.0, 0.1),
    (110.0, 100.0, 2.0),
]


@pytest.mark.parametrize("S, K, T", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(S, K, T, option_type):
    formula = greeks.delta(S, K, T, r, sigma, option_type)
    nudge = greeks.fd_delta(S, K, T, r, sigma, option_type)
    assert formula == pytest.approx(nudge, abs=1e-6)


@pytest.mark.parametrize("S, K, T", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_gamma_matches_finite_difference(S, K, T, option_type):
    formula = greeks.gamma(S, K, T, r, sigma, option_type)
    nudge = greeks.fd_gamma(S, K, T, r, sigma, option_type)
    # Looser tolerance: a second difference divides by h squared, which
    # magnifies floating-point rounding in the prices.
    assert formula == pytest.approx(nudge, abs=1e-5)


@pytest.mark.parametrize("S, K, T", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vega_matches_finite_difference(S, K, T, option_type):
    formula = greeks.vega(S, K, T, r, sigma, option_type)
    nudge = greeks.fd_vega(S, K, T, r, sigma, option_type)
    assert formula == pytest.approx(nudge, abs=1e-4)


@pytest.mark.parametrize("S, K, T", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_finite_difference(S, K, T, option_type):
    formula = greeks.theta(S, K, T, r, sigma, option_type)
    nudge = greeks.fd_theta(S, K, T, r, sigma, option_type)
    assert formula == pytest.approx(nudge, abs=1e-4)


@pytest.mark.parametrize("S, K, T", CASES)
def test_call_delta_minus_put_delta_is_one(S, K, T):
    # Differentiate put-call parity (call - put = S - K*e^(-rT)) by S.
    call = greeks.delta(S, K, T, r, sigma, "call")
    put = greeks.delta(S, K, T, r, sigma, "put")
    assert call - put == pytest.approx(1.0)


@pytest.mark.parametrize("S, K, T", CASES)
def test_delta_stays_in_its_range(S, K, T):
    assert 0.0 <= greeks.delta(S, K, T, r, sigma, "call") <= 1.0
    assert -1.0 <= greeks.delta(S, K, T, r, sigma, "put") <= 0.0


@pytest.mark.parametrize("S, K, T", CASES)
def test_gamma_and_vega_are_positive(S, K, T):
    assert greeks.gamma(S, K, T, r, sigma) > 0.0
    assert greeks.vega(S, K, T, r, sigma) > 0.0


# ---------------------------------------------------------------------------
# The same checks with a dividend yield
# ---------------------------------------------------------------------------

Q = 0.03


@pytest.mark.parametrize("S, K, T", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("name, tolerance", [("delta", 1e-6), ("gamma", 1e-5),
                                             ("vega", 1e-4), ("theta", 1e-4)])
def test_greeks_with_dividend_yield_match_finite_differences(S, K, T, option_type, name, tolerance):
    formula = getattr(greeks, name)(S, K, T, r, sigma, option_type, Q)
    nudge = getattr(greeks, "fd_" + name)(S, K, T, r, sigma, option_type, Q)
    assert formula == pytest.approx(nudge, abs=tolerance)


def test_dividend_yield_lowers_call_delta():
    without = greeks.delta(100.0, 100.0, 1.0, r, sigma, "call")
    with_dividends = greeks.delta(100.0, 100.0, 1.0, r, sigma, "call", Q)
    assert with_dividends < without
