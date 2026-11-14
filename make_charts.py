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
