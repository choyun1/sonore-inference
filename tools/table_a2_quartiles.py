"""Check readings of Table A.2's "Dist. params." against its printed quartiles (design C3).

Table A.2 gives, for each GP's sigma and lengthscale, two numbers ("Dist.
params."), bounds, and the quartiles of 5000 samples. The text says the
inverse softplus of each is normally distributed, truncated to the bounds
[paper App. A.2, A.3]. It does not say what the second number is. This
script samples each entry under four readings of it, as the normal's scale
directly, its absolute value, softplus of it, and exp of it, and prints the
quartiles of softplus(x) truncated to the bounds next to the paper's.

    python tools/table_a2_quartiles.py [--draws 200000]
"""

import argparse
import math

import numpy as np

# (sound type, GP, parameter): (first, second, low, high, Q1, Q2, Q3), from Table A.2
TABLE = {
    ("W,H", "f0", "sigma"): (5.6, 4.7, 0.1, 33, 3.0, 5.9, 9.0),
    ("W,H", "f0", "lengthscale"): (2.2, 6.0, 0.1, 10, 0.31, 2.5, 5.5),
    # the same with the lower bound BASS's config has, which fits Q2 and Q3
    ("W,H", "f0", "lengthscale 0.01"): (2.2, 6.0, 0.01, 10, 0.31, 2.5, 5.5),
    ("W", "level", "sigma"): (1.0, -0.96, 0.1, 50, 1.2, 1.3, 1.5),
    ("W", "level", "lengthscale"): (6.9, 0.23, 0.01, 10, 6.4, 6.9, 7.5),
    ("H", "level", "sigma"): (7.8, 2.2, 0.1, 50, 6.3, 7.8, 9.4),
    ("H", "level", "lengthscale"): (-1.7, 0.99, 0.01, 10, 0.078, 0.18, 0.39),
    ("H", "spectrum", "sigma"): (12, 4.6, 0.1, 50, 8.8, 11.8, 14.9),
    ("H", "spectrum", "lengthscale"): (-0.13, 9.2, 0.1, 33, 1.3, 4.7, 9.4),
    ("N", "level", "sigma"): (1.0, 0.81, 0.1, 50, 0.80, 1.3, 2.0),
    ("N", "level", "lengthscale"): (-5.1, 2.4, 0.01, 10, 0.022, 0.05, 0.16),
    ("N", "spectrum", "sigma"): (8.4, 1.3, 0.1, 50, 7.4, 8.4, 9.5),
    ("N", "spectrum", "lengthscale"): (16, 6e-3, 0.1, 33, 16.0, 16.5, 17.0),
}


def softplus(x):
    return np.logaddexp(0.0, x)


READINGS = {
    "scale": lambda s: s,
    "|scale|": abs,
    "softplus": lambda s: float(softplus(s)),
    "exp": math.exp,
}


def quartiles(loc, scale, low, high, draws, rng):
    if scale <= 0:
        return None
    kept = []
    while sum(len(k) for k in kept) < draws:
        x = softplus(rng.normal(loc, scale, draws))
        kept.append(x[(x >= low) & (x <= high)])
        if len(kept) > 200 and sum(len(k) for k in kept) < draws / 100:
            return None  # almost all mass outside the bounds
    return np.quantile(np.concatenate(kept)[:draws], [0.25, 0.5, 0.75])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=200_000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    print(f"{'entry':<26} {'paper Q1/Q2/Q3':<20} " + " ".join(f"{name:<20}" for name in READINGS))
    for (kind, gp, parameter), (loc, second, low, high, *paper) in TABLE.items():
        cells = []
        for reading in READINGS.values():
            q = quartiles(loc, reading(second), low, high, args.draws, rng)
            cells.append("-" if q is None else "/".join(f"{v:.3g}" for v in q))
        name = f"{kind} {gp} {parameter}"
        print(f"{name:<26} {'/'.join(f'{v:g}' for v in paper):<20} " + " ".join(f"{c:<20}" for c in cells))


if __name__ == "__main__":
    main()
