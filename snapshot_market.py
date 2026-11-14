"""
Download today's option chains and risk-free rate and save a dated snapshot.

Run with:   python snapshot_market.py

Run it once per day at most. Everything else (make_charts.py) reads the
saved files, so results stay reproducible.
"""

from datetime import date
from pathlib import Path

from optionlab.market_data import save_snapshot

TICKERS = ["SPY", "EWJ"]
DATA = Path(__file__).parent / "data"

folder, info = save_snapshot(TICKERS, DATA, date.today().isoformat())

print(f"saved snapshot to {folder.relative_to(DATA.parent)}")
rate = info["risk_free"]
print(f"risk-free: {rate['yield_percent']}% on {rate['date']} "
      f"(continuous rate {rate['continuous_rate']:.4%})")
for ticker, details in info["underlyings"].items():
    print(f"{ticker}: spot {details['spot']:.2f} on {details['spot_date']}, "
          f"{details['contracts']} contracts across {details['expiries']} expiries")
