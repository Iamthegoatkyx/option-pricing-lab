"""Tests for the convergence table."""

import pytest

from optionlab.binomial import binomial_price
from optionlab.black_scholes import black_scholes_price
from optionlab.convergence import convergence_table

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_table_has_one_row_per_step_count():
    steps, prices, errors, exact = convergence_table(S, K, T, r, sigma, "call", max_steps=20)
    assert steps == list(range(1, 21))
    assert len(prices) == 20
    assert len(errors) == 20


def test_table_values_come_from_the_two_pricers():
    steps, prices, errors, exact = convergence_table(S, K, T, r, sigma, "put", max_steps=20)
    assert exact == pytest.approx(black_scholes_price(S, K, T, r, sigma, "put"))
    assert prices[4] == pytest.approx(binomial_price(S, K, T, r, sigma, 5, "put"))
    assert errors[4] == pytest.approx(prices[4] - exact)
