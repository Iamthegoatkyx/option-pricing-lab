"""
Black-Scholes pricing for European options.

The idea in one paragraph
-------------------------
The binomial tree lets the stock make a few big jumps. Black-Scholes is what
you get when the number of steps goes to infinity, so the stock moves
continuously. In that limit the stock price at expiry follows a lognormal
distribution (its logarithm is normally distributed), and the discounted
risk-neutral average payoff can be worked out exactly with a formula. No
tree, no loops: just a few lines of arithmetic and the normal distribution.
"""

import math

from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Price a European call or put with the Black-Scholes formula
    (no dividends).

    Parameters
    ----------
    S : float
        Stock price today, e.g. 100.
    K : float
        Strike price.
    T : float
        Time to expiry in YEARS (must be greater than 0).
    r : float
        Risk-free rate per year, continuously compounded (0.05 = 5%).
    sigma : float
        Volatility per year (0.20 = 20%, must be greater than 0).
    option_type : str
        "call" or "put". Anything else should raise ValueError.

    Returns
    -------
    float
        The option price today.

    The formulas, in plain English
    ------------------------------
    N(x) below means the standard normal cumulative distribution function:
    the probability that a standard normal random number is less than x.
    In code it is norm.cdf(x)  (already imported at the top of this file).

    1. Two helper numbers:

           d1 = ( ln(S / K) + (r + sigma^2 / 2) * T ) / ( sigma * sqrt(T) )
           d2 = d1 - sigma * sqrt(T)

       Read d2 as "how many standard deviations the stock is above the
       strike, after allowing for drift". sigma * sqrt(T) is the standard
       deviation of the log stock price over the whole life of the option.

    2. Call price:

           call = S * N(d1)  -  K * e^(-r*T) * N(d2)

       - N(d2) is the risk-neutral probability the call finishes in the
         money (stock above K at expiry), so K * e^(-rT) * N(d2) is the
         value today of the strike you expect to pay.
       - S * N(d1) is the value today of the stock you expect to receive.
         N(d1) is also the call's delta (stage 5).

    3. Put price:

           put = K * e^(-r*T) * N(-d2)  -  S * N(-d1)

       Same story mirrored: N(-d2) is the probability the stock finishes
       BELOW K. You can check this formula is consistent with the call
       formula using put-call parity and the fact that N(x) + N(-x) = 1.

    Hints for writing it
    --------------------
    a. First check option_type is "call" or "put"; raise ValueError if not.
    b. math.log is the natural logarithm ln. math.sqrt and math.exp you
       already know.
    c. Compute sigma * sqrt(T) once and reuse it for d1 and d2.
    d. Watch the brackets in d1: the whole top is divided by the whole
       bottom.
    e. Work through the hand example in tests/test_black_scholes.py with a
       calculator first. It has unusually clean numbers: d1 = 0.35 and
       d2 = 0.15.
    f. The function is about 10 lines. If yours is much longer, stop and
       re-read the formulas.
    """
    raise NotImplementedError("Your turn: write black_scholes_price() using the hints above.")
