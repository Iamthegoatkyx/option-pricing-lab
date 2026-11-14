"""
Two ways to model dividends, and what each one says the stock is worth
"without its dividends" up to an option's expiry.

Put-call parity for a dividend-paying stock is

    call - put = (S - value of dividends paid before expiry) - K * e^(-rT)

because the option holder never receives those dividends. The two models
differ only in how they work out "value of dividends paid before expiry".

1. Continuous yield q
   Pretend dividends trickle out smoothly at a rate q per year.
       value = S * (1 - e^(-q*T))
   Simple, and it is what the q in black_scholes_price() means. It grows
   smoothly with T.

2. Discrete cash dividends
   Use the actual lumps of cash on their actual dates.
       value = sum of  amount * e^(-r * time until that payment)
   for every payment whose ex-dividend date is on or before expiry.
   It jumps each time an expiry passes another ex-dividend date.

The future payments are not known yet, so project_dividends() GUESSES them
by assuming next year repeats last year. That is an assumption, not data.
"""

import math
from datetime import date, timedelta


def trailing_dividends(history, as_of):
    """
    The dividends paid in the 365 days up to as_of.

    history : list of (date as "YYYY-MM-DD", amount) pairs, oldest first
    Returns the pairs that fall inside the window.
    """
    end = date.fromisoformat(as_of)
    start = end - timedelta(days=365)
    return [(day, amount) for day, amount in history
            if start < date.fromisoformat(day) <= end]


def trailing_yield(history, spot, as_of):
    """Dividend yield: last 12 months of dividends divided by today's price."""
    total = sum(amount for day, amount in trailing_dividends(history, as_of))
    return total / spot


def project_dividends(history, as_of, years=2):
    """
    Guess future dividends: each payment of the last 12 months is assumed
    to repeat, with the same amount, 52 weeks later (and 104 weeks later,
    and so on for `years` years).

    52 weeks (364 days) rather than 365 days keeps the day of the week the
    same, and ex-dividend dates are set by weekday (SPY: third Friday of
    the quarter-end month).

    Returns a list of (date as "YYYY-MM-DD", amount), soonest first.
    """
    recent = trailing_dividends(history, as_of)

    projected = []
    for year in range(1, years + 1):
        for day, amount in recent:
            future = date.fromisoformat(day) + timedelta(days=364 * year)
            if future > date.fromisoformat(as_of):
                projected.append((future.isoformat(), amount))

    return sorted(projected)


def value_of_yield(spot, q, T):
    """Value today of the dividends before expiry under a continuous yield."""
    return spot * (1.0 - math.exp(-q * T))


def value_of_cash_dividends(projected, valuation_date, expiry, r):
    """
    Value today of the cash dividends with an ex-date after valuation_date
    and on or before expiry, each discounted back at the risk-free rate.

    An option that expires ON an ex-dividend date is counted as affected:
    the stock has already dropped by the dividend when it expires.
    """
    start = date.fromisoformat(valuation_date)
    end = date.fromisoformat(expiry)

    total = 0.0
    for day, amount in projected:
        ex_date = date.fromisoformat(day)
        if start < ex_date <= end:
            years_away = (ex_date - start).days / 365.0
            total += amount * math.exp(-r * years_away)
    return total
