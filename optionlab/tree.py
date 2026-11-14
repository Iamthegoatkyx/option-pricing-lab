"""
Keep the WHOLE binomial tree, for drawing and for the early-exercise boundary.

binomial_price() only returns today's price: it throws away each layer of the
tree once it has stepped back past it. To draw the tree we need every node,
so build_tree() does the same calculation but stores each layer.

How the tree is stored
----------------------
A list of lists. tree[i][j] is the node at time step i that was reached by
j up moves (and so i - j down moves). Step i has i + 1 nodes.

    stock[0] = [S]
    stock[1] = [S*d, S*u]
    stock[2] = [S*d*d, S, S*u*u]
    ...
"""

import math

# Differences smaller than this are treated as "equal" (see build_tree).
ROUNDING_TOLERANCE = 1e-9


def build_tree(S, K, T, r, sigma, steps, option_type="call", american=False):
    """
    Build the full CRR tree. Same inputs as binomial_price().

    Returns three lists of lists, all indexed [i][j]
    (i = time step, j = number of up moves):

    stock    : stock price at each node
    option   : option value at each node
    exercise : True where an American option should be exercised at once
               (always False for European options and at nodes where the
               payoff is zero)
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    def payoff(stock_price):
        # What the option pays if exercised when the stock is at stock_price.
        if option_type == "call":
            return max(stock_price - K, 0.0)
        return max(K - stock_price, 0.0)

    # Stock prices: fill every layer going forwards in time.
    stock = []
    for i in range(steps + 1):
        stock.append([S * u**j * d**(i - j) for j in range(i + 1)])

    # Option values: start with empty layers, fill the last one with payoffs,
    # then fill the rest going backwards in time.
    option = [[0.0] * (i + 1) for i in range(steps + 1)]
    exercise = [[False] * (i + 1) for i in range(steps + 1)]

    for j in range(steps + 1):
        option[steps][j] = payoff(stock[steps][j])

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # Value of keeping the option alive for one more step.
            hold = disc * (p * option[i + 1][j + 1] + (1 - p) * option[i + 1][j])
            exercise_now = payoff(stock[i][j])

            # Exercise only if it is better by more than rounding error.
            # In theory the two can be exactly equal (e.g. when r = 0), and
            # floating-point arithmetic would then pick a winner at random.
            if american and exercise_now > hold + ROUNDING_TOLERANCE:
                option[i][j] = exercise_now
                exercise[i][j] = True
            else:
                option[i][j] = hold

    return stock, option, exercise
