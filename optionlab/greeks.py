"""
The Greeks: how sensitive the Black-Scholes price is to each input.

Each Greek is a slope (a derivative) of the price surface:

    delta = change in price per 1.00 rise in the stock price      dV/dS
    gamma = change in DELTA per 1.00 rise in the stock price      d2V/dS2
    vega  = change in price per 1.00 rise in volatility           dV/dsigma
            (1.00 = 100 percentage points, so divide by 100 for "per 1%")
    theta = change in price per year of time PASSING              -dV/dT
            (divide by 365 for "per day"; usually negative: options
            lose value as expiry approaches)

There are two ways to get each one, and this file has both:

1. A formula, found by differentiating the Black-Scholes formula by hand.
2. A finite difference: nudge the input a little, re-price, and measure
   the slope. This needs no calculus, only black_scholes_price(), so it
   is an independent check on the formulas.
"""

import math

from scipy.stats import norm

from optionlab.black_scholes import black_scholes_price


def d1_d2(S, K, T, r, sigma):
    """The two helper numbers from the Black-Scholes formula."""
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


# ---------------------------------------------------------------------------
# 1. Formulas
# ---------------------------------------------------------------------------
# norm.cdf(x) is N(x), the area under the bell curve to the left of x.
# norm.pdf(x) is the HEIGHT of the bell curve at x. It appears whenever we
# differentiate N(x), because the slope of an area is the curve's height.

def delta(S, K, T, r, sigma, option_type="call"):
    """
    call delta = N(d1)        between 0 and 1
    put delta  = N(d1) - 1    between -1 and 0

    Also the number of shares to hold to hedge one option.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma, option_type="call"):
    """
    gamma = pdf(d1) / (S * sigma * sqrt(T))

    The same for calls and puts: put-call parity says call - put = S - K*e^(-rT),
    and the right-hand side is a straight line in S, which has no curvature.
    option_type is accepted only so every Greek has the same inputs.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S, K, T, r, sigma, option_type="call"):
    """
    vega = S * pdf(d1) * sqrt(T)

    The same for calls and puts, for the same parity reason as gamma:
    S - K*e^(-rT) does not depend on sigma at all.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * math.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    """
    call theta = -S * pdf(d1) * sigma / (2 * sqrt(T))  -  r * K * e^(-rT) * N(d2)
    put theta  = -S * pdf(d1) * sigma / (2 * sqrt(T))  +  r * K * e^(-rT) * N(-d2)

    First term : value lost because there is less time for the stock to move.
    Second term: the effect of interest on the strike. A call holder pays K
                 later, so time passing brings that payment closer (bad).
                 A put holder RECEIVES K later, so time passing is good.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma)
    time_decay = -S * norm.pdf(d1) * sigma / (2.0 * math.sqrt(T))
    discounted_strike = K * math.exp(-r * T)
    if option_type == "call":
        return time_decay - r * discounted_strike * norm.cdf(d2)
    return time_decay + r * discounted_strike * norm.cdf(-d2)


# ---------------------------------------------------------------------------
# 2. Finite differences
# ---------------------------------------------------------------------------
# "Central" difference: step the same distance h both ways and take the
# slope between the two prices:   (V(x + h) - V(x - h)) / (2h).
# It is more accurate than a one-sided step because the errors on each
# side largely cancel.

def fd_delta(S, K, T, r, sigma, option_type="call", h=0.01):
    up = black_scholes_price(S + h, K, T, r, sigma, option_type)
    down = black_scholes_price(S - h, K, T, r, sigma, option_type)
    return (up - down) / (2.0 * h)


def fd_gamma(S, K, T, r, sigma, option_type="call", h=0.01):
    # Curvature: how much the middle sits below the average of its neighbours.
    up = black_scholes_price(S + h, K, T, r, sigma, option_type)
    middle = black_scholes_price(S, K, T, r, sigma, option_type)
    down = black_scholes_price(S - h, K, T, r, sigma, option_type)
    return (up - 2.0 * middle + down) / h**2


def fd_vega(S, K, T, r, sigma, option_type="call", h=0.0001):
    up = black_scholes_price(S, K, T, r, sigma + h, option_type)
    down = black_scholes_price(S, K, T, r, sigma - h, option_type)
    return (up - down) / (2.0 * h)


def fd_theta(S, K, T, r, sigma, option_type="call", h=0.0001):
    # Time PASSING means time to expiry T gets SMALLER, hence the minus sign.
    longer = black_scholes_price(S, K, T + h, r, sigma, option_type)
    shorter = black_scholes_price(S, K, T - h, r, sigma, option_type)
    return -(longer - shorter) / (2.0 * h)
