"""
Early-exercise boundary of an American put, read off the binomial tree.

At any moment there is a critical stock price. Below it, the put is so deep
in the money that taking K - S now is better than waiting. Above it, holding
is better. The line joining those critical prices through time is the
early-exercise boundary.
"""

from optionlab.tree import build_tree


def exercise_boundary(S, K, T, r, sigma, steps):
    """
    For each time step of the tree, find the HIGHEST stock price at which
    the American put should be exercised immediately.

    Returns
    -------
    times    : time from today in years, one per step (0, dt, 2*dt, ...)
    boundary : the critical stock price at each of those times, or None
               where the tree has no exercise node at that step.

    Two things to know about the output
    -----------------------------------
    - Early steps are often None. The tree starts at S and needs several
      down moves before it reaches prices low enough to exercise, so it
      simply has no nodes down there yet. That is a limit of the tree's
      shape, not a statement that exercise is never optimal.
    - The line is jagged. The tree only has stock prices at its nodes, so
      the true boundary lies somewhere between the highest "exercise" node
      and the node just above it. More steps means finer nodes.

    The final step (expiry itself) is left out: at expiry there is no
    "exercise early or hold" decision left to make.
    """
    stock, option, exercise = build_tree(S, K, T, r, sigma, steps, "put", american=True)
    dt = T / steps

    times, boundary = [], []
    for i in range(steps):                      # every step except expiry
        # Stock prices at this step where exercise is flagged.
        exercise_prices = [stock[i][j] for j in range(i + 1) if exercise[i][j]]

        times.append(i * dt)
        if exercise_prices:
            boundary.append(max(exercise_prices))
        else:
            boundary.append(None)

    return times, boundary
