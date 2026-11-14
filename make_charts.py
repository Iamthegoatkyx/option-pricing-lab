"""
Build every chart and save it to outputs/ as an interactive HTML file.

Run with:   python make_charts.py
"""

from pathlib import Path

from optionlab.charts import (convergence_figure, exercise_boundary_figure,
                              greeks_surface_figure, price_surface_figure,
                              tree_figure)

OUTPUTS = Path(__file__).parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# Inputs used for the example charts.
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def save(fig, name):
    path = OUTPUTS / name
    # "cdn" loads the Plotly library from the internet instead of copying
    # 4 MB of it into every file, so the HTML files stay small.
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"saved {path.relative_to(OUTPUTS.parent)}")


# Stage 1: small trees you can check by hand.
save(tree_figure(S, K, T, r, sigma, 4, "call", american=False),
     "binomial_tree_european_call.html")
save(tree_figure(S, K, T, r, sigma, 4, "put", american=True),
     "binomial_tree_american_put.html")

# Stage 3: binomial price vs number of steps, against Black-Scholes.
save(convergence_figure(S, K, T, r, sigma, "call", max_steps=500),
     "convergence.html")

# Stage 4: 3D Black-Scholes price surface across spot x time to expiry.
save(price_surface_figure(K, r, sigma), "price_surface.html")

# Stage 5: 3D Greeks surfaces with a dropdown.
save(greeks_surface_figure(K, r, sigma), "greeks_surfaces.html")

# Stage 6: American put early-exercise boundary.
save(exercise_boundary_figure(S, K, T, r, sigma), "exercise_boundary.html")


# ---------------------------------------------------------------------------
# Stage 8: real market check, from the latest saved snapshot in data/.
# ---------------------------------------------------------------------------
import pandas as pd

from optionlab.charts import market_vs_model_figure, smile_figure
from optionlab.market_check import (add_implied_vols, add_model_prices,
                                    add_time_to_expiry, at_the_money_vol,
                                    clean_quotes, keep_near_the_money,
                                    parity_gap, pick_expiry)
from optionlab.dividends import trailing_yield
from optionlab.market_data import (latest_snapshot_folder, load_dividends,
                                   load_snapshot)

DATA = Path(__file__).parent / "data"
snapshot_folder = latest_snapshot_folder(DATA)

if snapshot_folder is None:
    print("no market snapshot found: run  python snapshot_market.py  first")
else:
    info, chains = load_snapshot(snapshot_folder)
    rate = info["risk_free"]["continuous_rate"]
    print(f"\nmarket snapshot {info['snapshot_date']}, "
          f"risk-free {info['risk_free']['yield_percent']}% ({info['risk_free']['date']})")

    for ticker, chain in chains.items():
        spot = info["underlyings"][ticker]["spot"]
        spot_date = info["underlyings"][ticker]["spot_date"]
        # Dividend yield: the last 12 months of dividends over the spot price.
        dividend_yield = trailing_yield(load_dividends(snapshot_folder, ticker),
                                        spot, spot_date)
        note = (f"quotes saved {info['snapshot_date']}, spot {spot:.2f} is the "
                f"{spot_date} close, r = {rate:.2%}, dividend yield q = "
                f"{dividend_yield:.2%}, European Black-Scholes, strikes within 20% of spot")

        # 1. clean
        cleaned, removed = clean_quotes(chain)
        cleaned = add_time_to_expiry(cleaned, spot_date)
        print(f"\n{ticker}: {len(chain)} raw quotes, {len(cleaned)} kept. Removed: {removed}")

        # 2. keep sensible strikes and expiries between 1 week and 250 days
        near = keep_near_the_money(cleaned, spot)
        near = near[(near["days"] >= 7) & (near["days"] <= 250)]

        # 3. choose expiries: the ones nearest 30, 90 and 180 days if the
        #    market is busy; every expiry if it is thin.
        if near["expiry"].nunique() > 6:
            chosen = sorted({pick_expiry(near, target) for target in (30, 90, 180)})
        else:
            chosen = sorted(near["expiry"].unique())
        selected = near[near["expiry"].isin(chosen)].reset_index(drop=True)

        # 4. implied vols, first ignoring dividends (kept for comparison),
        #    then with the dividend yield. The charts use the second.
        without_dividends = add_implied_vols(selected, spot, rate)
        selected = add_implied_vols(selected, spot, rate, dividend_yield)
        print(f"  {len(selected)} quotes in the chosen expiries, "
              f"{int(selected['implied_vol'].isna().sum())} with no implied vol")
        save(smile_figure(selected, ticker, spot, note), f"smile_{ticker}.html")

        # 5. market vs model for the expiry with the most usable quotes
        usable = selected.dropna(subset=["implied_vol"])
        busiest = usable.groupby("expiry").size().idxmax()
        one_expiry = selected[selected["expiry"] == busiest]
        sigma, atm_strike = at_the_money_vol(one_expiry, spot)
        one_expiry = add_model_prices(one_expiry, spot, rate, sigma, dividend_yield)
        print(f"  market vs model: expiry {busiest}, {len(one_expiry)} quotes, "
              f"at-the-money strike {atm_strike:g}, sigma {sigma:.2%}")
        save(market_vs_model_figure(one_expiry, ticker, spot, sigma, note),
             f"market_vs_model_{ticker}.html")

        # 6. summary numbers for the README and for discussion
        for expiry in chosen:
            rows = selected[selected["expiry"] == expiry]
            vol, strike = at_the_money_vol(rows, spot)
            gap_strike, gap = parity_gap(rows, spot, rate)
            vol_text = "n/a" if vol is None else f"{vol:.2%}"
            gap_text = "n/a" if gap is None else f"{gap:+.2f} at K={gap_strike:g}"
            print(f"  {expiry} ({int(rows['days'].iloc[0])}d): {len(rows)} quotes, "
                  f"ATM vol {vol_text}, parity gap {gap_text}")

            # Call vol and put vol at the strike nearest spot, with q = 0
            # and with the dividend yield.
            if strike is not None:
                for label, table in (("q = 0", without_dividends), (f"q = {dividend_yield:.2%}", selected)):
                    at_strike = table[(table["expiry"] == expiry) & (table["strike"] == strike)]
                    vols = {row.option_type: row.implied_vol for row in at_strike.itertuples()}
                    if "call" in vols and "put" in vols:
                        print(f"      {label}: call vol {vols['call']:.2%}, put vol {vols['put']:.2%}, "
                              f"put minus call {vols['put'] - vols['call']:+.2%}")


# ---------------------------------------------------------------------------
# Dividends: does a continuous yield or discrete cash dividends close the
# SPY put-call parity gap better?  (EWJ has no strike with both a clean
# call and a clean put, so there is nothing to compare there.)
# ---------------------------------------------------------------------------
from optionlab.charts import parity_gap_figure
from optionlab.dividends import project_dividends, trailing_dividends, trailing_yield
from optionlab.market_check import parity_gap_table
from optionlab.market_data import load_dividends

if snapshot_folder is not None and (snapshot_folder / "SPY_dividends.csv").exists():
    ticker = "SPY"
    spot = info["underlyings"][ticker]["spot"]
    spot_date = info["underlyings"][ticker]["spot_date"]

    history = load_dividends(snapshot_folder, ticker)
    q = trailing_yield(history, spot, spot_date)
    projected = project_dividends(history, spot_date)
    print(f"\n{ticker} dividends in the last 12 months: {trailing_dividends(history, spot_date)}")
    print(f"trailing yield q = {q:.3%}")
    print(f"projected (assumes next year repeats last year): {projected[:5]}")

    cleaned, removed = clean_quotes(chains[ticker])
    cleaned = add_time_to_expiry(cleaned, spot_date)
    cleaned = cleaned[(cleaned["days"] >= 7) & (cleaned["days"] <= 400)]

    table = parity_gap_table(cleaned, spot, rate, spot_date, q, projected)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))

    for label, part in (("all expiries", table),
                        ("up to 105 days", table[table["days"] <= 105]),
                        ("beyond 105 days", table[table["days"] > 105])):
        print(f"  {label} ({len(part)} expiries):")
        for column in ("gap_none", "gap_yield", "gap_cash"):
            print(f"    {column}: average {part[column].mean():+.2f}, "
                  f"lowest {part[column].min():+.2f}, highest {part[column].max():+.2f}, "
                  f"standard deviation {part[column].std(ddof=0):.2f}")

    note = (f"quotes saved {info['snapshot_date']}, spot {spot:.2f} is the "
            f"{spot_date} close, r = {rate:.2%}, q = {q:.2%}")
    save(parity_gap_figure(table, projected, spot_date, ticker, note),
         f"parity_gap_{ticker}.html")
