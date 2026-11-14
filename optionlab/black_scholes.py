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


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Price a European call or put with the Black-Scholes formula,
    with an optional continuous dividend yield q.

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
    q : float
        Dividend yield per year, continuously compounded (0.01 = 1%).
        The default 0.0 means no dividends, so old code keeps working.

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

    Adding a dividend yield q
    -------------------------
    Owning the option is not owning the stock: you miss the dividends paid
    before expiry. A share bought today WITHOUT its dividends up to time T
    is worth S * e^(-q*T), not S. So wherever the formula has S, use
    S * e^(-q*T) instead. That gives three changes:

           d1   = ( ln(S / K) + (r - q + sigma^2 / 2) * T ) / ( sigma * sqrt(T) )
           call = S * e^(-q*T) * N(d1)  -  K * e^(-r*T) * N(d2)
           put  = K * e^(-r*T) * N(-d2) -  S * e^(-q*T) * N(-d1)

    d2 = d1 - sigma * sqrt(T) is unchanged. (The "- q" inside d1 comes from
    ln(S * e^(-qT) / K) = ln(S / K) - q*T.)

    Put-call parity becomes:  call - put = S * e^(-q*T) - K * e^(-r*T)

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
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    d1 = (math.log(S / K) + ((r - q) + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
