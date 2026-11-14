"""Tests for the price grid behind the 3D surface."""

import pytest

from optionlab.black_scholes import black_scholes_price
from optionlab.surfaces import make_axis, price_surface

K, r, sigma = 100.0, 0.05, 0.20


def test_make_axis_is_evenly_spaced_and_includes_both_ends():
    assert make_axis(0.0, 1.0, 5) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_grid_has_one_row_per_time_and_one_column_per_spot():
    spots, times, grid = price_surface(K, r, sigma, "call", spot_points=11, time_points=7)
    assert len(spots) == 11
    assert len(times) == 7
    assert len(grid) == 7
    assert all(len(row) == 11 for row in grid)


def test_grid_values_come_from_black_scholes_price():
    spots, times, grid = price_surface(K, r, sigma, "put", spot_points=11, time_points=7)
    assert grid[3][8] == pytest.approx(
        black_scholes_price(spots[8], K, times[3], r, sigma, "put"))


def test_call_price_rises_with_spot_along_every_row():
    spots, times, grid = price_surface(K, r, sigma, "call")
    for row in grid:
        assert all(later >= earlier for earlier, later in zip(row, row[1:]))


def test_put_price_falls_with_spot_along_every_row():
    spots, times, grid = price_surface(K, r, sigma, "put")
    for row in grid:
        assert all(later <= earlier for earlier, later in zip(row, row[1:]))
