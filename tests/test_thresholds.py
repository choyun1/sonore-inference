"""The threshold of Eqn 2, read as integrals, against cases with known answers."""

import numpy as np
import pytest

from sonore_inference.thresholds import probability_from_log_odds, threshold

VALUES = [0, 5, 10, 20, 30, 40, 50]


def test_step_gives_the_crossing():
    # p(H1) jumps between 10 and 20: linear interpolation crosses 0.5 at 15
    assert threshold(VALUES, [0, 0, 0, 1, 1, 1, 1]) == pytest.approx(15.0)


def test_never_and_always():
    assert threshold(VALUES, np.zeros(7)) == 50.0
    assert threshold(VALUES, np.ones(7)) == 0.0


def test_both_sides_of_eqn_2_balance():
    probability_high = probability_from_log_odds([-9, -9, -3, 2, 40, 100, 150])
    tau = threshold(VALUES, probability_high)
    grid = np.linspace(0, 50, 200_001)
    p_high = np.interp(grid, VALUES, probability_high)
    below, above = grid < tau, grid > tau
    step = grid[1] - grid[0]
    assert np.sum(p_high[below]) * step == pytest.approx(np.sum(1 - p_high[above]) * step, abs=1e-3)


def test_probability_from_log_odds():
    assert probability_from_log_odds(0.0) == 0.5
    assert probability_from_log_odds(np.log(3)) == pytest.approx(0.75)


def test_rejects_unsorted_values():
    with pytest.raises(ValueError):
        threshold([0, 10, 5], [0, 1, 1])
