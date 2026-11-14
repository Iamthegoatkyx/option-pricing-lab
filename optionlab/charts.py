"""
Plotly charts. Each function returns a figure; make_charts.py saves them
as interactive HTML files in outputs/.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optionlab import greeks
from optionlab.boundary import exercise_boundary
from optionlab.convergence import convergence_table
from optionlab.surfaces import greek_surface, price_surface
from optionlab.tree import build_tree

# Colours used across all charts (dark theme).
LINE_COLOUR = "#5b6472"      # branches of the tree
NODE_COLOUR = "#4cc9f0"      # ordinary node
EXERCISE_COLOUR = "#f72585"  # node where early exercise is best
ERROR_COLOUR = "#ffd166"     # error curves


def tree_figure(S, K, T, r, sigma, steps, option_type="call", american=False):
    """
    Draw a small binomial tree with the stock price and option value
    written at every node.

    Where each node is drawn
    ------------------------
    x = time step i
    y = (up moves) - (down moves) = j - (i - j) = 2j - i
    So an up move goes one step right and one step up, a down move goes one
    step right and one step down, and "up then down" lands on the same dot
    as "down then up". That is the recombining tree.
    """
    stock, option, exercise = build_tree(S, K, T, r, sigma, steps, option_type, american)

    # --- branches: one line from each node to its up child and down child ---
    line_x, line_y = [], []
    for i in range(steps):
        for j in range(i + 1):
            y = 2 * j - i
            # None breaks the line, so all branches can share one trace.
            line_x += [i, i + 1, None, i, i + 1, None]
            line_y += [y, y + 1, None, y, y - 1, None]

    # --- nodes: position, label and colour for each one ---
    node_x, node_y, labels, colours = [], [], [], []
    for i in range(steps + 1):
        for j in range(i + 1):
            node_x.append(i)
            node_y.append(2 * j - i)
            label = f"S = {stock[i][j]:.2f}<br>V = {option[i][j]:.2f}"
            if exercise[i][j]:
                label += "<br>exercise"
            labels.append(label)
            colours.append(EXERCISE_COLOUR if exercise[i][j] else NODE_COLOUR)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines",
        line=dict(color=LINE_COLOUR, width=1.5), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=14, color=colours),
        text=labels, textposition="top center", textfont=dict(size=11),
        hoverinfo="skip",
    ))

    style = "American" if american else "European"
    dt = T / steps
    fig.update_layout(
        template="plotly_dark",
        title=(f"{steps}-step binomial tree: {style} {option_type}"
               f"<br><sup>S = {S:g}, K = {K:g}, T = {T:g} yr, r = {r:.0%}, "
               f"sigma = {sigma:.0%}.  S = stock price, V = option value"
               + (".  Pink = exercise now" if american else "") + "</sup>"),
        showlegend=False,
        xaxis=dict(
            title="time (years)",
            tickvals=list(range(steps + 1)),
            ticktext=[f"{i * dt:g}" for i in range(steps + 1)],
            range=[-0.5, steps + 0.5],
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(visible=False, range=[-steps - 0.6, steps + 1.4]),
        height=750,
    )
    return fig


def convergence_figure(S, K, T, r, sigma, option_type="call", max_steps=500):
    """
    Two panels sharing the idea "more steps -> closer to Black-Scholes":

    top    : binomial price for each step count, with the Black-Scholes
             price as a flat dashed line.
    bottom : size of the error |binomial - Black-Scholes| on log-log axes.
             On log-log axes a relationship like error = c / n shows up as a
             straight line sloping down, so the grey guide line 1/n lets you
             read off how fast the error shrinks.
    """
    steps, prices, errors, exact = convergence_table(S, K, T, r, sigma, option_type, max_steps)
    abs_errors = [abs(e) for e in errors]

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.14,
        subplot_titles=("Price vs number of steps", "Absolute error vs number of steps (log-log)"),
    )

    # --- top panel ---
    fig.add_trace(go.Scatter(
        x=steps, y=prices, mode="lines", name="binomial",
        line=dict(color=NODE_COLOUR, width=1),
        hovertemplate="steps = %{x}<br>price = %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[steps[0], steps[-1]], y=[exact, exact], mode="lines",
        name=f"Black-Scholes = {exact:.4f}",
        line=dict(color=EXERCISE_COLOUR, width=2, dash="dash"), hoverinfo="skip",
    ), row=1, col=1)

    # --- bottom panel ---
    fig.add_trace(go.Scatter(
        x=steps, y=abs_errors, mode="lines", name="|error|",
        line=dict(color=ERROR_COLOUR, width=1),
        hovertemplate="steps = %{x}<br>|error| = %{y:.5f}<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=[1 / n for n in steps], mode="lines", name="guide: 1 / steps",
        line=dict(color=LINE_COLOUR, width=1.5, dash="dot"), hoverinfo="skip",
    ), row=2, col=1)

    fig.update_xaxes(title_text="number of steps", row=1, col=1)
    fig.update_yaxes(title_text="option price", row=1, col=1)
    fig.update_xaxes(title_text="number of steps (log scale)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="|binomial - Black-Scholes| (log scale)", type="log", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=(f"Binomial tree converging to Black-Scholes: European {option_type}"
               f"<br><sup>S = {S:g}, K = {K:g}, T = {T:g} yr, r = {r:.0%}, "
               f"sigma = {sigma:.0%}, steps 1 to {max_steps}</sup>"),
        height=850,
    )
    return fig


def price_surface_figure(K, r, sigma):
    """
    Rotatable 3D surface of the Black-Scholes price across
    spot price (x) and time to expiry (y), with buttons to switch
    between the call and the put.

    Both surfaces are added to the figure; the buttons just choose which
    one is visible.
    """
    def surface_title(option_type):
        return (f"Black-Scholes price surface: European {option_type}"
                f"<br><sup>K = {K:g}, r = {r:.0%}, sigma = {sigma:.0%}. "
                f"Drag to rotate, scroll to zoom</sup>")

    fig = go.Figure()

    for option_type in ("call", "put"):
        spots, times, grid = price_surface(K, r, sigma, option_type)
        fig.add_trace(go.Surface(
            x=spots, y=times, z=grid,
            colorscale="Viridis",
            visible=(option_type == "call"),      # start by showing the call
            colorbar=dict(title="price"),
            hovertemplate=("spot = %{x:.1f}<br>time = %{y:.2f} yr"
                           "<br>price = %{z:.2f}<extra></extra>"),
        ))

    fig.update_layout(
        template="plotly_dark",
        title=surface_title("call"),
        scene=dict(
            xaxis_title="spot price",
            yaxis_title="time to expiry (years)",
            zaxis_title="option price",
        ),
        updatemenus=[dict(
            type="buttons", direction="right", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
            showactive=False, bgcolor="#2a2f3a", font=dict(color="#e6e6e6"),
            # Each button changes which surface is visible AND the title,
            # so the title always says which option you are looking at.
            buttons=[
                dict(label="Call", method="update",
                     args=[{"visible": [True, False]}, {"title.text": surface_title("call")}]),
                dict(label="Put", method="update",
                     args=[{"visible": [False, True]}, {"title.text": surface_title("put")}]),
            ],
        )],
        height=750,
    )
    return fig


def greeks_surface_figure(K, r, sigma):
    """
    Rotatable 3D surfaces of delta, gamma, vega and theta across
    spot price (x) and time to expiry (y), with a dropdown to switch.

    Gamma and vega are the same for calls and puts, so they appear once.
    Delta and theta differ, so each has a call and a put entry.

    Units on the chart (chosen to be the ones traders quote):
      vega  is shown per 1 percentage point of volatility (formula / 100)
      theta is shown per calendar day                     (formula / 365)
    """
    # (dropdown label, function, option type, scale factor, z-axis title)
    choices = [
        ("Delta (call)", greeks.delta, "call", 1.0, "delta"),
        ("Delta (put)", greeks.delta, "put", 1.0, "delta"),
        ("Gamma", greeks.gamma, "call", 1.0, "gamma"),
        ("Vega", greeks.vega, "call", 1.0 / 100.0, "vega per 1% vol"),
        ("Theta (call)", greeks.theta, "call", 1.0 / 365.0, "theta per day"),
        ("Theta (put)", greeks.theta, "put", 1.0 / 365.0, "theta per day"),
    ]

    def greeks_title(label):
        return (f"Black-Scholes Greeks surface: {label}"
                f"<br><sup>K = {K:g}, r = {r:.0%}, sigma = {sigma:.0%}. "
                f"Drag to rotate, scroll to zoom</sup>")

    fig = go.Figure()
    buttons = []

    for position, (label, function, option_type, scale, z_title) in enumerate(choices):
        spots, times, grid = greek_surface(function, K, r, sigma, option_type)
        scaled = [[value * scale for value in row] for row in grid]

        fig.add_trace(go.Surface(
            x=spots, y=times, z=scaled,
            colorscale="Viridis",
            visible=(position == 0),              # start with the first one
            colorbar=dict(title=z_title),
            hovertemplate=("spot = %{x:.1f}<br>time = %{y:.2f} yr"
                           f"<br>{z_title} = " + "%{z:.4f}<extra></extra>"),
        ))

        # This button shows surface number `position` and hides the rest.
        visible = [k == position for k in range(len(choices))]
        buttons.append(dict(
            label=label, method="update",
            args=[{"visible": visible},
                  {"title.text": greeks_title(label),
                   "scene.zaxis.title.text": z_title}],
        ))

    fig.update_layout(
        template="plotly_dark",
        title=greeks_title(choices[0][0]),
        scene=dict(
            xaxis_title="spot price",
            yaxis_title="time to expiry (years)",
            zaxis_title=choices[0][4],
        ),
        updatemenus=[dict(
            type="dropdown", x=0.0, y=1.0, xanchor="left", yanchor="bottom",
            showactive=False, bgcolor="#2a2f3a", font=dict(color="#e6e6e6"),
            buttons=buttons,
        )],
        height=750,
    )
    return fig


def exercise_boundary_figure(S, K, T, r, sigma, fine_steps=1000, coarse_steps=50):
    """
    The American put's early-exercise boundary through time.

    Below the line: exercise now. Above it: keep holding.
    Two trees are drawn so you can see what the number of steps does:
    a fine one (line) and a coarse one (dots).
    """
    fine_times, fine = exercise_boundary(S, K, T, r, sigma, fine_steps)
    coarse_times, coarse = exercise_boundary(S, K, T, r, sigma, coarse_steps)

    lowest = min(price for price in fine if price is not None)
    floor = lowest - 5.0                      # bottom edge of the chart

    fig = go.Figure()

    # Invisible flat line along the bottom, so the next trace can shade
    # the area between itself and this one.
    fig.add_trace(go.Scatter(
        x=[fine_times[0], fine_times[-1]], y=[floor, floor], mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    # Plotly leaves a gap wherever a value is None, which is what we want
    # for the early steps where the tree has no exercise node.
    fig.add_trace(go.Scatter(
        x=fine_times, y=fine, mode="lines", name=f"boundary, {fine_steps} steps",
        line=dict(color=EXERCISE_COLOUR, width=1.5),
        fill="tonexty", fillcolor="rgba(247, 37, 133, 0.15)",
        hovertemplate="time = %{x:.3f} yr<br>exercise below %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=coarse_times, y=coarse, mode="markers", name=f"boundary, {coarse_steps} steps",
        marker=dict(color=NODE_COLOUR, size=6),
        hovertemplate="time = %{x:.3f} yr<br>exercise below %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0.0, T], y=[K, K], mode="lines", name=f"strike K = {K:g}",
        line=dict(color=LINE_COLOUR, width=1.5, dash="dash"), hoverinfo="skip",
    ))

    fig.add_annotation(x=0.6 * T, y=floor + 2.5, text="exercise now", showarrow=False,
                       font=dict(color=EXERCISE_COLOUR))
    fig.add_annotation(x=0.3 * T, y=K - 6.0, text="hold", showarrow=False)

    fig.update_layout(
        template="plotly_dark",
        title=(f"American put: early-exercise boundary from the binomial tree"
               f"<br><sup>S = {S:g}, K = {K:g}, T = {T:g} yr, r = {r:.0%}, "
               f"sigma = {sigma:.0%}</sup>"),
        xaxis=dict(title="time from today (years); expiry is at the right edge",
                   range=[0.0, T]),
        yaxis=dict(title="stock price", range=[floor, K + 3.0]),
        height=650,
    )
    return fig


# One colour per expiry on the market charts.
EXPIRY_COLOURS = [NODE_COLOUR, ERROR_COLOUR, EXERCISE_COLOUR,
                  "#80ed99", "#c77dff", "#ff9f1c", "#e6e6e6"]


def smile_figure(quotes, ticker, spot, snapshot_note):
    """
    Implied volatility against strike, one colour per expiry.
    Calls are filled circles, puts are hollow diamonds.

    If Black-Scholes described the market exactly, every point of one
    expiry would sit on a flat horizontal line.
    """
    fig = go.Figure()
    usable = quotes.dropna(subset=["implied_vol"])

    expiries = sorted(usable["expiry"].unique())
    for position, expiry in enumerate(expiries):
        colour = EXPIRY_COLOURS[position % len(EXPIRY_COLOURS)]
        one_expiry = usable[usable["expiry"] == expiry]
        days = int(one_expiry["days"].iloc[0])

        for option_type, symbol in (("call", "circle"), ("put", "diamond-open")):
            rows = one_expiry[one_expiry["option_type"] == option_type].sort_values("strike")
            if rows.empty:
                continue
            fig.add_trace(go.Scatter(
                x=rows["strike"], y=rows["implied_vol"] * 100.0, mode="markers",
                name=f"{expiry} ({days}d) {option_type}s, {len(rows)} quotes",
                marker=dict(color=colour, symbol=symbol, size=7),
                hovertemplate=("strike = %{x}<br>implied vol = %{y:.1f}%"
                               f"<extra>{expiry} {option_type}</extra>"),
            ))

    fig.add_vline(x=spot, line=dict(color=LINE_COLOUR, dash="dash"),
                  annotation_text=f"spot {spot:.2f}", annotation_position="top")
    fig.update_layout(
        template="plotly_dark",
        title=(f"{ticker}: implied volatility by strike"
               f"<br><sup>{snapshot_note}</sup>"),
        xaxis_title="strike",
        yaxis_title="implied volatility (%)",
        height=650,
    )
    return fig


def market_vs_model_figure(quotes, ticker, spot, sigma, snapshot_note):
    """
    Market prices against Black-Scholes prices for ONE expiry.

    Dots are market mid prices, with a bar from bid to ask.
    Lines are Black-Scholes prices using one volatility (the at-the-money
    implied vol) for every strike.
    """
    fig = go.Figure()
    expiry = quotes["expiry"].iloc[0]
    days = int(quotes["days"].iloc[0])

    for option_type, colour in (("call", NODE_COLOUR), ("put", EXERCISE_COLOUR)):
        rows = quotes[quotes["option_type"] == option_type].sort_values("strike")
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["strike"], y=rows["mid"], mode="markers",
            name=f"market {option_type}s (mid, bar = bid to ask)",
            marker=dict(color=colour, size=6),
            error_y=dict(type="data", symmetric=False,
                         array=rows["ask"] - rows["mid"],
                         arrayminus=rows["mid"] - rows["bid"],
                         color=colour, thickness=1, width=0),
            hovertemplate="strike = %{x}<br>market mid = %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=rows["strike"], y=rows["model_price"], mode="lines",
            name=f"Black-Scholes {option_type}s, sigma = {sigma:.1%}",
            line=dict(color=colour, width=1.5, dash="dot"),
            hovertemplate="strike = %{x}<br>model = %{y:.2f}<extra></extra>",
        ))

    fig.add_vline(x=spot, line=dict(color=LINE_COLOUR, dash="dash"),
                  annotation_text=f"spot {spot:.2f}", annotation_position="top")
    fig.update_layout(
        template="plotly_dark",
        title=(f"{ticker}: market vs Black-Scholes prices, expiry {expiry} ({days} days)"
               f"<br><sup>{snapshot_note}</sup>"),
        xaxis_title="strike",
        yaxis_title="option price",
        height=650,
    )
    return fig
