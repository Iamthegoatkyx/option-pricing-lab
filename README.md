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
| 6 | American put early-exercise boundary | not started |
| 7 | Implied volatility solver | not started |
| 8 | Real-market check (SPY, EWJ option chains) | not started |

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
