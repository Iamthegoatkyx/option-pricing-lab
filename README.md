# Option Pricing Lab

From binomial trees to Black-Scholes in 3D — a learning project that builds
option pricers from scratch, tests them, visualises them, and then checks
them against real market quotes.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Binomial tree pricer (Cox-Ross-Rubinstein), European + American | done |
| 2 | Black-Scholes pricer | done |
| 3 | Convergence: binomial vs Black-Scholes | done |
| 4 | 3D price surface (spot x time to expiry) | done |
| 5 | 3D Greeks surfaces, checked by finite differences | done |
| 6 | American put early-exercise boundary | done |
| 7 | Implied volatility solver | done |
| 8 | Real-market check (SPY, EWJ option chains) | done without dividends |
| 9 | Dividends: yield `q` in both pricers; yield vs discrete cash dividends on the SPY parity gap | parity comparison done; `q` in the pricers in progress (18 tests failing until then) |

## Setup (Mac)

```bash
# 1. Get the code
git clone https://github.com/Iamthegoatkyx/option-pricing-lab.git
cd option-pricing-lab

# 2. Create a virtual environment (a private copy of Python for this project,
#    so its packages do not clash with other projects)
python3 -m venv .venv

# 3. Activate it (do this every time you open a new terminal)
source .venv/bin/activate

# 4. Install the packages
pip install -r requirements.txt

# 5. Run the tests
pytest
```

## Layout

```
optionlab/     the pricing code (one file per model)
tests/         pytest tests
outputs/       interactive Plotly HTML charts
data/          dated snapshots of market data
```

## Known limitations

- No dividends yet. A dividend yield will be added in a later step; until
  then, model prices for dividend-paying underlyings (SPY, EWJ) are biased.

- Implied volatility is unreliable for deep in-the-money options with little
  time left: vega is nearly zero there, so the price says almost nothing
  about volatility.

- The market check prices American options (SPY and EWJ options can be
  exercised early) with the European Black-Scholes formula.
- The spot price is the stock's closing price, while option quotes may be
  from a slightly different time. Any mismatch shifts every implied vol.
- The relative-spread filter lets through some stale deep in-the-money
  quotes, because a wide spread is still small relative to a large price.
- EWJ options are thinly traded: in the 2026-11-14 snapshot only 68 of 663
  quotes survive cleaning.

- Future dividends are a guess: the discrete model assumes each of the last
  four payments repeats 52 weeks later with the same amount.

## Market data

```bash
python snapshot_market.py   # download today's chains + rate into data/<date>/
python make_charts.py       # rebuild every chart from the latest snapshot
```

The snapshot in `data/2026-11-14/` was taken on a Saturday, so it holds
Friday 13 November 2026 closing quotes. The risk-free rate is the 3-month
Treasury yield (FRED series DGS3MO).

## Charts

Build all charts with `python make_charts.py`, then open the HTML files in
`outputs/` in a browser.

- `binomial_tree_european_call.html`, `binomial_tree_american_put.html`:
  4-step trees with the stock price and option value at every node.
- `convergence.html`: binomial price for 1 to 500 steps against the
  Black-Scholes price, and the size of the error on log-log axes.
- `price_surface.html`: rotatable 3D Black-Scholes price surface across
  spot price and time to expiry, with Call / Put buttons.
- `greeks_surfaces.html`: rotatable 3D surfaces of delta, gamma, vega and
  theta, with a dropdown to switch between them.
- `exercise_boundary.html`: the stock price below which the American put
  should be exercised, through time, from a 1000-step and a 50-step tree.
- `smile_SPY.html`, `smile_EWJ.html`: implied volatility by strike for
  several expiries, calls and puts shown separately.
- `market_vs_model_SPY.html`, `market_vs_model_EWJ.html`: market mid prices
  (with bid-ask bars) against Black-Scholes prices at one volatility.
- `parity_gap_SPY.html`: the put-call parity gap by expiry with no
  dividends, a continuous yield, and projected discrete cash dividends.
