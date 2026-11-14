"""
Convergence: how the binomial price approaches Black-Scholes as the number
of steps grows.
"""

from optionlab.binomial import binomial_price
from optionlab.black_scholes import black_scholes_price


def convergence_table(S, K, T, r, sigma, option_type="call", max_steps=500):
    """
    Price the same European option with 1, 2, ..., max_steps tree steps.

    Returns
    -------
    steps  : list of step counts [1, 2, ..., max_steps]
    prices : binomial price for each step count
    errors : binomial price minus Black-Scholes price (can be + or -)
    exact  : the Black-Scholes price (one number)
    """
    exact = black_scholes_price(S, K, T, r, sigma, option_type)

    steps, prices, errors = [], [], []
    for n in range(1, max_steps + 1):
        price = binomial_price(S, K, T, r, sigma, n, option_type)
        steps.append(n)
        prices.append(price)
        errors.append(price - exact)

    return steps, prices, errors, exact
