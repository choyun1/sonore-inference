"""Thresholds from posterior probabilities along a stimulus parameter [Eqn 2 of Cusimano et al., 2024].

When hypothesis H0 explains low values of a stimulus parameter and H1 high
values, the paper defines the threshold ``tau`` by

    sum over values below tau of p(H1 | X) = sum over values above tau of p(H0 | X)

and describes it as based on the integral of the posterior over the
parameter. Read as integrals over ``[low, high]``, with the posterior
interpolated linearly between the measured values, the two sides differ by
``tau - low - integral of p(H0 | X)``, so

    tau = low + integral from low to high of p(H0 | X).

The threshold is the area under the H0 curve: unique, and equal to the
crossing point when the posterior steps from 1 to 0. When H0 is preferred
everywhere it is ``high``, which the paper reports as not measurable.
"""

from __future__ import annotations

import numpy as np


def probability_from_log_odds(log_odds):
    """``p(H1 | X)`` from the log posterior odds of H1 over H0."""
    return 1 / (1 + np.exp(-np.asarray(log_odds, float)))


def threshold(values, probability_high) -> float:
    """Eqn 2's threshold, from ``p(H1 | X)`` measured at increasing stimulus ``values``."""
    values = np.asarray(values, float)
    probability_low = 1 - np.asarray(probability_high, float)
    if values.ndim != 1 or values.shape != probability_low.shape or np.any(np.diff(values) <= 0):
        raise ValueError("values must be increasing and match the probabilities")
    area = np.sum(np.diff(values) * (probability_low[1:] + probability_low[:-1]) / 2)
    return float(values[0] + area)
