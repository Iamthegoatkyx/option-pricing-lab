"""
Binomial option pricing: the Cox-Ross-Rubinstein (CRR) tree.

The idea in one paragraph
-------------------------
Chop the option's life into `steps` short periods. In each period the stock
can only do one of two things: go up by a factor u, or down by a factor d.
That gives a tree of possible stock prices. At expiry (the right-hand edge of
the tree) we know exactly what the option is worth: its payoff. We then walk
backwards through the tree, one period at a time, and at each node the option
is worth the discounted "risk-neutral" average of its two possible values one
period later. The value at the very first node is today's price.
"""

import math


def binomial_price(S, K, T, r, sigma, steps, option_type="call", american=False):
    """
    Price a call or put with a Cox-Ross-Rubinstein binomial tree.

    Parameters
    ----------
    S : float
        Stock price today (the "spot" price), e.g. 100.
    K : float
        Strike price: the price you may buy (call) or sell (put) at.
    T : float
        Time to expiry in YEARS, e.g. 0.5 for six months.
    r : float
        Risk-free interest rate per year, continuously compounded,
        as a decimal: 0.05 means 5%.
    sigma : float
        Volatility of the stock per year, as a decimal: 0.20 means 20%.
    steps : int
        Number of time periods in the tree (1 or more).
    option_type : str
        "call" or "put". Anything else should raise ValueError.
    american : bool
        False -> European (can only be exercised at expiry).
        True  -> American (can be exercised at any node).

    Returns
    -------
    float
        The option price today.

    The formulas, in plain English
    ------------------------------
    1. Length of one period:
           dt = T / steps

    2. Up and down factors (this choice is what makes it "CRR"):
           u = e^(sigma * sqrt(dt))      stock is multiplied by u on an up move
           d = 1 / u                     ... and by d on a down move
       Because d = 1/u, an up followed by a down brings you back to where
       you started, so the tree "recombines": after i periods there are only
       i + 1 different stock prices, not 2^i.

    3. Risk-neutral probability of an up move:
           p = (e^(r * dt) - d) / (u - d)
       This is NOT the real-world chance of the stock rising. It is the one
       probability that makes the stock's expected growth equal the risk-free
       rate: p*u + (1-p)*d = e^(r*dt). Pricing with it gives the same answer
       as building a riskless hedge of stock + option, which is why it works.

    4. Stock price at a node reached by j up moves out of i periods:
           S * u^j * d^(i - j)          for j = 0, 1, ..., i

    5. Payoff at expiry:
           call: max(stock - K, 0)
           put:  max(K - stock, 0)

    6. One step backwards (discounted risk-neutral average):
           value = e^(-r * dt) * (p * value_if_up + (1 - p) * value_if_down)

    7. American only: at every node the holder can exercise right now, so
           value = max(value from step 6, payoff from exercising now)

    Hints for writing it
    --------------------
    a. First check option_type is "call" or "put"; raise ValueError if not.
    b. Compute dt, u, d, p and the one-period discount factor e^(-r*dt) ONCE,
       before any loop.
    c. Build a list of the steps + 1 option payoffs at expiry (i = steps),
       ordered by number of up moves j = 0 ... steps.
    d. Loop i backwards from steps - 1 down to 0. At each i, build a new,
       shorter list of i + 1 values. Node j's "down" child is entry j of the
       old list, and its "up" child is entry j + 1. (Draw a 2-step tree on
       paper and label the entries to convince yourself.)
    e. If american is True, compare each value with the immediate exercise
       payoff at that node (you need the stock price there: formula 4).
    f. When the loop ends, the list has exactly one entry: the price today.

    A plain Python list and for-loops are fine. Do not worry about speed yet.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    values = []
    for j in range(steps + 1):
        ST = S * u**j * d**(steps - j)
        if option_type == "call":
            values.append(max(ST - K, 0))
        else:
            values.append(max(K - ST, 0))

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold = disc * (p * values[j + 1] + (1 - p) * values[j])
            if american:
                St = S * u**j * d**(i - j)
                if option_type == "call":
                    exercise = max(St - K, 0)
                else:
                    exercise = max(K - St, 0)
                values[j] = max(hold, exercise)
            else:
                values[j] = hold

    return values[0]   
    
