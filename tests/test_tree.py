"""
Tests for build_tree(), the helper that keeps every node of the tree.

The main safety check: its first node must equal binomial_price(), because
both are the same calculation. If they ever disagree, one of them is wrong.
"""

import pytest

from optionlab.binomial import binomial_price
from optionlab.tree import build_tree

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("american", [False, True])
@pytest.mark.parametrize("steps", [1, 4, 50])
def test_root_of_tree_equals_binomial_price(option_type, american, steps):
    stock, option, exercise = build_tree(S, K, T, r, sigma, steps, option_type, american)
    expected = binomial_price(S, K, T, r, sigma, steps, option_type, american)
    assert option[0][0] == pytest.approx(expected, abs=1e-10)


def test_three_step_tree_has_ten_nodes():
    # 1 + 2 + 3 + 4 = 10, because the tree recombines.
    stock, option, exercise = build_tree(S, K, T, r, sigma, 3)
    assert sum(len(layer) for layer in stock) == 10


def test_up_then_down_returns_to_start():
    # d = 1/u, so the middle node after two steps is back at S.
    stock, option, exercise = build_tree(S, K, T, r, sigma, 2)
    assert stock[2][1] == pytest.approx(S)


def test_european_tree_never_flags_early_exercise():
    stock, option, exercise = build_tree(S, K, T, r, sigma, 4, "put", american=False)
    assert not any(flag for layer in exercise for flag in layer)


def test_american_call_tree_never_flags_early_exercise():
    # No dividends: exercising a call early is never better than holding.
    stock, option, exercise = build_tree(S, K, T, r, sigma, 50, "call", american=True)
    assert not any(flag for layer in exercise for flag in layer)


def test_american_put_tree_flags_some_early_exercise():
    stock, option, exercise = build_tree(S, K, T, r, sigma, 4, "put", american=True)
    assert any(flag for layer in exercise for flag in layer)
