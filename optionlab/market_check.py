"""
Compare the models with real option quotes.

Steps, each one a small function:

    clean_quotes()         throw away quotes that cannot be trusted
    add_time_to_expiry()   turn expiry dates into T in years
    pick_expiry()          choose the listed expiry nearest a target
    add_implied_vols()     run the stage 7 solver on every quote
    at_the_money_vol()     one volatility to represent an expiry
    add_model_prices()     Black-Scholes price of every quote at that one vol

Limitations (also listed in the README)
---------------------------------------
- No dividends. SPY and EWJ both pay them, so the model is biased.
- SPY and EWJ options are American, but implied volatilities here come
  from the European Black-Scholes formula.
"""

import math
from datetime import date

import pandas as pd

from optionlab.black_scholes import black_scholes_price
from optionlab.implied_vol import implied_volatility


def clean_quotes(chain, max_relative_spread=0.25):
    """
    Remove quotes that are not real, tradeable prices.

    A quote is dropped if:
      1. the bid is zero or missing  -> nobody is willing to buy it, so the
                                        "price" is only a seller's wish;
      2. the ask is below the bid    -> a broken or stale quote;
      3. the spread is very wide     -> (ask - bid) / mid above
                                        max_relative_spread. The true price
                                        could be anywhere in that gap, so
                                        the mid is a guess.

    Adds two columns to what is kept:
      mid             = (bid + ask) / 2, our estimate of the market price
      relative_spread = (ask - bid) / mid

    Returns
    -------
    cleaned : the DataFrame of surviving quotes
    removed : dictionary counting how many rows each rule removed
    """
    removed = {}
    quotes = chain.copy()

    no_bid = quotes["bid"].isna() | (quotes["bid"] <= 0)
    removed["zero or missing bid"] = int(no_bid.sum())
    quotes = quotes[~no_bid]

    crossed = quotes["ask"].isna() | (quotes["ask"] < quotes["bid"])
    removed["ask below bid"] = int(crossed.sum())
    quotes = quotes[~crossed]

    quotes = quotes.assign(mid=(quotes["bid"] + quotes["ask"]) / 2.0)
    quotes = quotes.assign(relative_spread=(quotes["ask"] - quotes["bid"]) / quotes["mid"])

    wide = quotes["relative_spread"] > max_relative_spread
    removed["spread too wide"] = int(wide.sum())
    quotes = quotes[~wide]

    return quotes.reset_index(drop=True), removed


def add_time_to_expiry(quotes, valuation_date):
    """
    Add 'days' (calendar days from valuation_date to expiry) and
    'T' = days / 365, the time in years our formulas need.
    Options that have already expired (days <= 0) are dropped.
    """
    start = date.fromisoformat(valuation_date)
    days = [(date.fromisoformat(expiry) - start).days for expiry in quotes["expiry"]]

    quotes = quotes.assign(days=days)
    quotes = quotes.assign(T=quotes["days"] / 365.0)
    return quotes[quotes["days"] > 0].reset_index(drop=True)


def pick_expiry(quotes, target_days):
    """The listed expiry whose days-to-expiry is closest to target_days."""
    choices = quotes[["expiry", "days"]].drop_duplicates()
    gap = (choices["days"] - target_days).abs()
    return choices.loc[gap.idxmin(), "expiry"]


def add_implied_vols(quotes, spot, r):
    """
    Add an 'implied_vol' column: the Black-Scholes volatility that
    reproduces each quote's mid price. Rows where no volatility works
    (the solver returns None) get a missing value.
    """
    vols = []
    for row in quotes.itertuples():
        vols.append(implied_volatility(row.mid, spot, row.strike, row.T, r, row.option_type))

    return quotes.assign(implied_vol=pd.Series(vols, index=quotes.index, dtype="float64"))


def at_the_money_vol(quotes, spot):
    """
    One volatility to stand for a whole expiry: the average implied vol of
    the call and put whose strike is closest to the spot price.

    At-the-money options are the most heavily traded and have the largest
    vega, so their implied vol is the most reliable single number.
    Returns (None, None) if no quote has an implied vol.
    """
    usable = quotes.dropna(subset=["implied_vol"])
    if usable.empty:
        return None, None       # a thin market can leave nothing to use
    nearest_strike = usable.loc[(usable["strike"] - spot).abs().idxmin(), "strike"]
    at_the_money = usable[usable["strike"] == nearest_strike]
    return float(at_the_money["implied_vol"].mean()), float(nearest_strike)


def add_model_prices(quotes, spot, r, sigma):
    """
    Add 'model_price': the Black-Scholes price of every quote using the
    SAME volatility sigma for all strikes, and 'model_error' = model - mid.

    Black-Scholes assumes one constant volatility. If that were true of the
    market, one sigma would price every strike correctly and all the errors
    would be about zero.
    """
    prices = []
    for row in quotes.itertuples():
        prices.append(float(black_scholes_price(spot, row.strike, row.T, r, sigma, row.option_type)))

    quotes = quotes.assign(model_price=prices)
    return quotes.assign(model_error=quotes["model_price"] - quotes["mid"])


def keep_near_the_money(quotes, spot, low=0.8, high=1.2):
    """
    Keep strikes between low * spot and high * spot (80% to 120% by default).

    Far from the money, options are either almost worthless or almost all
    intrinsic value. Either way vega is tiny, so implied vol is unreliable
    (the stage 7 limitation), and quotes there are often stale.
    """
    near = (quotes["strike"] >= low * spot) & (quotes["strike"] <= high * spot)
    return quotes[near].reset_index(drop=True)


def parity_gap(quotes, spot, r, dividend_value=0.0):
    """
    How far the nearest-to-the-money call and put of ONE expiry are from
    put-call parity:

        gap = (S - dividend_value - K * e^(-rT))  -  (call mid - put mid)

    dividend_value is the value today of the dividends paid before expiry
    (see optionlab/dividends.py). The default 0.0 means "no dividends".

    With no dividends, European options and perfectly timed prices the gap
    would be zero. Returns (strike, gap), or (None, None) if no strike has
    both a call and a put quote.
    """
    calls = quotes[quotes["option_type"] == "call"].set_index("strike")
    puts = quotes[quotes["option_type"] == "put"].set_index("strike")
    shared = [strike for strike in calls.index if strike in puts.index]
    if not shared:
        return None, None

    strike = min(shared, key=lambda k: abs(k - spot))
    T = float(calls.loc[strike, "T"])
    market = float(calls.loc[strike, "mid"] - puts.loc[strike, "mid"])
    theory = spot - dividend_value - strike * math.exp(-r * T)
    return float(strike), theory - market


def parity_gap_table(quotes, spot, r, valuation_date, q, projected_dividends):
    """
    The put-call parity gap of every expiry under three dividend models:

        gap_none  : no dividends
        gap_yield : continuous yield q
        gap_cash  : projected discrete cash dividends

    quotes must already be cleaned and have 'days' and 'T' columns.
    Returns a DataFrame with one row per expiry that has a call and a put
    at the same strike.
    """
    from optionlab.dividends import value_of_cash_dividends, value_of_yield

    rows = []
    for expiry in sorted(quotes["expiry"].unique()):
        one_expiry = quotes[quotes["expiry"] == expiry]
        strike, gap_none = parity_gap(one_expiry, spot, r)
        if strike is None:
            continue

        T = float(one_expiry["T"].iloc[0])
        yield_value = value_of_yield(spot, q, T)
        cash_value = value_of_cash_dividends(projected_dividends, valuation_date, expiry, r)

        rows.append({
            "expiry": expiry,
            "days": int(one_expiry["days"].iloc[0]),
            "strike": strike,
            "yield_value": yield_value,
            "cash_value": cash_value,
            "gap_none": gap_none,
            "gap_yield": parity_gap(one_expiry, spot, r, yield_value)[1],
            "gap_cash": parity_gap(one_expiry, spot, r, cash_value)[1],
        })

    return pd.DataFrame(rows)
