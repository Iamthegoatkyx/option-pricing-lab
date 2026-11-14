"""
Download real market data and save it as a dated snapshot.

Why save a snapshot?
--------------------
Option quotes change every second. If the analysis downloaded fresh data
each time it ran, no result could ever be reproduced or checked. So we
download ONCE, save the raw files under data/<date>/, and every later step
reads those files.

Sources
-------
- Option chains and the stock price: Yahoo Finance, through yfinance.
- Risk-free rate: FRED series DGS3MO (3-month US Treasury yield).
"""

import json
import math
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"


def parse_fred_csv(text):
    """
    Read FRED's CSV text and return (date, yield in percent) for the most
    recent day that has a value.

    The file looks like:
        observation_date,DGS3MO
        2026-11-10,4.30
        2026-11-11,            <- holidays are blank, so we skip them
        2026-11-12,4.32
    """
    lines = text.strip().splitlines()
    for line in reversed(lines[1:]):            # newest first, skip the header
        date, value = line.split(",")
        if value.strip() not in ("", "."):
            return date, float(value)
    raise ValueError("no usable value in the FRED file")


def to_continuous_rate(yield_percent):
    """
    Turn a quoted annual yield in percent into the continuously compounded
    rate our formulas use.

    A yield of y per year means 1 grows to (1 + y) in a year. Continuous
    compounding grows it to e^r. Setting those equal gives r = ln(1 + y).
    (This treats the quoted yield as a simple annual rate, which is a small
    approximation: Treasury yields use their own quoting conventions.)
    """
    return math.log(1.0 + yield_percent / 100.0)


def fetch_risk_free_rate():
    """Download DGS3MO and return (date, yield in percent)."""
    # requests is used rather than Python's built-in downloader because it
    # ships with its own up-to-date security certificates.
    response = requests.get(FRED_URL, timeout=30)
    response.raise_for_status()
    return parse_fred_csv(response.text)


def download_chain(ticker):
    """
    Download every listed option for one ticker.

    Returns
    -------
    spot       : last closing price of the underlying
    spot_date  : the date of that close, as "YYYY-MM-DD"
    chain      : DataFrame, one row per option contract
    """
    stock = yf.Ticker(ticker)

    history = stock.history(period="5d")
    spot = float(history["Close"].iloc[-1])
    spot_date = history.index[-1].strftime("%Y-%m-%d")

    pieces = []
    for expiry in stock.options:                 # each expiry date is one download
        chain = stock.option_chain(expiry)
        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            piece = pd.DataFrame({
                "ticker": ticker,
                "expiry": expiry,
                "option_type": option_type,
                "strike": table["strike"],
                "bid": table["bid"],
                "ask": table["ask"],
                "last_price": table["lastPrice"],
                "volume": table["volume"],
                "open_interest": table["openInterest"],
                "last_trade": table["lastTradeDate"].astype(str),
            })
            pieces.append(piece)

    return spot, spot_date, pd.concat(pieces, ignore_index=True)


def download_dividends(ticker):
    """Every past dividend of one ticker: DataFrame with ex_date and amount."""
    history = yf.Ticker(ticker).dividends
    return pd.DataFrame({
        "ex_date": [day.strftime("%Y-%m-%d") for day in history.index],
        "amount": history.values,
    })


def load_dividends(folder, ticker):
    """Read a saved dividend file as a list of (ex_date, amount) pairs."""
    table = pd.read_csv(Path(folder) / f"{ticker}_dividends.csv")
    return list(zip(table["ex_date"], table["amount"]))


def save_snapshot(tickers, data_folder, snapshot_date):
    """
    Download the rate and each ticker's chain, and save them under
    data_folder/<snapshot_date>/ :

        <ticker>_options.csv   the raw chain, untouched
        <ticker>_dividends.csv every past dividend (ex-date and amount)
        snapshot.json          spot prices, the rate, and where they came from
    """
    folder = Path(data_folder) / snapshot_date
    folder.mkdir(parents=True, exist_ok=True)

    rate_date, yield_percent = fetch_risk_free_rate()
    info = {
        "snapshot_date": snapshot_date,
        "risk_free": {
            "series": "DGS3MO (FRED, 3-month US Treasury)",
            "date": rate_date,
            "yield_percent": yield_percent,
            "continuous_rate": to_continuous_rate(yield_percent),
        },
        "underlyings": {},
    }

    for ticker in tickers:
        spot, spot_date, chain = download_chain(ticker)
        chain.to_csv(folder / f"{ticker}_options.csv", index=False)
        download_dividends(ticker).to_csv(folder / f"{ticker}_dividends.csv", index=False)
        info["underlyings"][ticker] = {
            "spot": spot,
            "spot_date": spot_date,
            "contracts": len(chain),
            "expiries": int(chain["expiry"].nunique()),
        }

    with open(folder / "snapshot.json", "w") as file:
        json.dump(info, file, indent=2)

    return folder, info


def load_snapshot(folder):
    """Read a saved snapshot back: (info dictionary, {ticker: chain DataFrame})."""
    folder = Path(folder)
    with open(folder / "snapshot.json") as file:
        info = json.load(file)

    chains = {}
    for ticker in info["underlyings"]:
        chains[ticker] = pd.read_csv(folder / f"{ticker}_options.csv")
    return info, chains


def latest_snapshot_folder(data_folder):
    """The most recent dated folder inside data/, or None if there is none."""
    folders = sorted(p for p in Path(data_folder).iterdir()
                     if p.is_dir() and (p / "snapshot.json").exists())
    return folders[-1] if folders else None
