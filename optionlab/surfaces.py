"""
Grids of values across spot price x time to expiry, for 3D surface charts.
"""

from optionlab.black_scholes import black_scholes_price


def make_axis(low, high, points):
    """Evenly spaced numbers from low to high inclusive, e.g. (0, 1, 5) ->
    [0, 0.25, 0.5, 0.75, 1]."""
    step = (high - low) / (points - 1)
    return [low + step * k for k in range(points)]


def price_surface(K, r, sigma, option_type="call",
                  spot_low=50.0, spot_high=150.0, spot_points=51,
                  time_low=0.01, time_high=2.0, time_points=40):
    """
    Black-Scholes price for every combination of spot price and time to
    expiry on a grid.

    Time starts at 0.01 years (about 4 days), not 0, because the formula
    divides by sigma * sqrt(T) and so cannot be evaluated at T = 0 exactly.

    Returns
    -------
    spots : list of spot prices (the x axis)
    times : list of times to expiry in years (the y axis)
    grid  : list of lists; grid[a][b] is the price when
            time = times[a] and spot = spots[b].
            (Rows are times and columns are spots because that is the
            layout Plotly's Surface expects: z[y index][x index].)
    """
    spots = make_axis(spot_low, spot_high, spot_points)
    times = make_axis(time_low, time_high, time_points)

    grid = []
    for T in times:
        row = [black_scholes_price(S, K, T, r, sigma, option_type) for S in spots]
        grid.append(row)

    return spots, times, grid


def greek_surface(greek_function, K, r, sigma, option_type="call",
                  spot_low=50.0, spot_high=150.0, spot_points=51,
                  time_low=0.01, time_high=2.0, time_points=40):
    """
    Same grid as price_surface(), but filled with one of the Greeks.

    greek_function is the function itself, passed in like any other value,
    e.g. greek_surface(greeks.delta, 100, 0.05, 0.2). That way one loop
    serves all four Greeks instead of four copies of the same code.
    """
    spots = make_axis(spot_low, spot_high, spot_points)
    times = make_axis(time_low, time_high, time_points)

    grid = []
    for T in times:
        row = [greek_function(S, K, T, r, sigma, option_type) for S in spots]
        grid.append(row)

    return spots, times, grid
