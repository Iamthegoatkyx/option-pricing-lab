"""
Plotly charts. Each function returns a figure; make_charts.py saves them
as interactive HTML files in outputs/.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optionlab.convergence import convergence_table
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
