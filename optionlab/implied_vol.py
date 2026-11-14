"""
Implied volatility: the sigma that makes the Black-Scholes price equal a
given (market) price.

Every Black-Scholes input can be looked up except volatility: S, K, T and r
are known. So the market price of an option tells us what volatility the
market is "implying". Traders quote options this way because it lets them
compare options with different strikes and expiries on one scale.
"""

import math

from optionlab.black_scholes import black_scholes_price


def implied_volatility(price, S, K, T, r, option_type="call",
                       low=1e-6, high=5.0, tolerance=1e-8, max_iterations=200):
    """
    Find sigma such that black_scholes_price(S, K, T, r, sigma) = price.

    Parameters
    ----------
    price : the option price to match (e.g. a market mid price)
    S, K, T, r, option_type : as in black_scholes_price()
    low, high : the search range for sigma (0.0001% to 500% by default)
    tolerance : stop when the range is narrower than this
    max_iterations : safety limit so the loop can never run forever

    Returns
    -------
    float : the implied volatility, or
    None  : if no volatility in [low, high] can produce that price.

    How bisection works
    -------------------
    The Black-Scholes price always goes UP when sigma goes up (vega > 0).
    So for any guess we know which way to move:

        price at guess too low   -> the answer is a HIGHER sigma
        price at guess too high  -> the answer is a LOWER sigma

    Start with a range [low, high] that contains the answer. Try the
    midpoint, throw away the half that cannot contain the answer, and
    repeat. Each round halves the range, so about 30 rounds shrink a range
    of width 5 down to 0.00000001.

    When there is no answer
    -----------------------
    As sigma goes to 0 the price falls to a floor (the no-arbitrage lower
    bound) and as sigma grows it rises towards a ceiling. A price below the
    floor or above the ceiling cannot come from ANY volatility. That
    happens with real quotes (stale prices, wide spreads), so we return
    None rather than a made-up number.

    A limitation to know about
    --------------------------
    For options that are deep in the money with little time left, almost
    all of the price is intrinsic value and vega is nearly zero. The price
    then barely changes with sigma, so tiny rounding in the price becomes a
    large error in the implied volatility. The solver still returns a
    number there, but it should not be trusted.
    """
    price_at_low = black_scholes_price(S, K, T, r, low, option_type)
    price_at_high = black_scholes_price(S, K, T, r, high, option_type)

    # The answer must lie strictly between the two ends, otherwise there is
    # none. "<=" matters: a price sitting exactly ON the floor has no time
    # value at all, so it carries no information about volatility.
    if price <= price_at_low or price >= price_at_high:
        return None

    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        price_at_middle = black_scholes_price(S, K, T, r, middle, option_type)

        if price_at_middle < price:
            low = middle        # need more volatility: keep the upper half
        else:
            high = middle       # need less volatility: keep the lower half

        if high - low < tolerance:
            break

    return (low + high) / 2.0
