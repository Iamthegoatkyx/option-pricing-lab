"""Tests for the American put early-exercise boundary."""

from optionlab.boundary import exercise_boundary

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def known(values):
    """Drop the None entries (steps where the tree has no exercise node)."""
    return [v for v in values if v is not None]


def test_one_entry_per_step_before_expiry():
    times, boundary = exercise_boundary(S, K, T, r, sigma, 200)
    assert len(times) == 200
    assert len(boundary) == 200
    assert times[0] == 0.0


def test_boundary_is_always_below_the_strike():
    # You would never exercise a put when the stock is above the strike.
    times, boundary = exercise_boundary(S, K, T, r, sigma, 200)
    assert all(price < K for price in known(boundary))


def test_boundary_ends_higher_than_it_starts():
    times, boundary = exercise_boundary(S, K, T, r, sigma, 200)
    values = known(boundary)
    assert values[-1] > values[0]


def test_no_early_exercise_when_interest_rate_is_zero():
    # The only reason to exercise a put early is to earn interest on K.
    # With r = 0 there is no such reason, so no node should be flagged.
    times, boundary = exercise_boundary(S, K, T, 0.0, sigma, 400)
    assert known(boundary) == []


def test_higher_interest_rate_raises_the_boundary():
    low_times, low = exercise_boundary(S, K, T, 0.02, sigma, 400)
    high_times, high = exercise_boundary(S, K, T, 0.10, sigma, 400)
    assert high[200] > low[200]          # compare half-way through


def test_higher_volatility_lowers_the_boundary():
    calm_times, calm = exercise_boundary(S, K, T, r, 0.10, 400)
    wild_times, wild = exercise_boundary(S, K, T, r, 0.40, 400)
    assert wild[200] < calm[200]
