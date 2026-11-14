"""
Plotly charts. Each function returns a figure; make_charts.py saves them
as interactive HTML files in outputs/.
"""

import plotly.graph_objects as go

from optionlab.tree import build_tree

# Colours used across all charts (dark theme).
LINE_COLOUR = "#5b6472"      # branches of the tree
NODE_COLOUR = "#4cc9f0"      # ordinary node
EXERCISE_COLOUR = "#f72585"  # node where early exercise is best


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
