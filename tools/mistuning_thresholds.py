"""Mistuning thresholds from the output of tools/compare_mistuned_harmonic.py.

Rows for the same f0 and harmonic are merged across files (so a finer grid
can be run separately and added), sorted by mistuning; then it converts the log posterior odds
of two sources over one into probabilities and applies the threshold of
Eqn 2 (``sonore_inference.thresholds``), for the Laplace and the
importance-sampling estimates. A threshold of 50% means two sources were
never preferred, which the paper reports as not measurable.

A row is left out, and listed, when either evidence estimate is unreliable:
the Laplace approximation had to floor a Hessian eigenvalue (a flat
direction, where its Gaussian is far wider than the posterior), or the
importance-sampling estimate is not finite.

    python tools/mistuning_thresholds.py coarse/f0-*_h*.txt fine/f0-*_h*.txt
"""

import argparse
import math
import re

from sonore_inference.thresholds import probability_from_log_odds, threshold

HEADER = re.compile(r"f0 (\S+) Hz, harmonic (\d+) mistuned")
ROW = re.compile(r"\s*(\d+(?:\.\d+)?)" + r"\s+(-?(?:[\d.]+|inf|nan))" * 9 + r"\s*$")
FLOORED = re.compile(r"\s*note: floored Hessian eigenvalues H1 (\d+), H2 (\d+)")


def read(path):
    """(f0, harmonic, rows, left out) of one output file.

    ``rows`` maps each kept mistuning to its log odds (Laplace, importance
    sampling); ``left out`` lists the mistunings dropped as unreliable.
    """
    f0 = harmonic = None
    rows, left_out = {}, []
    with open(path) as lines:
        for line in lines:
            if match := HEADER.match(line):
                f0, harmonic = float(match[1]), int(match[2])
            elif match := ROW.match(line):
                numbers = [float(value) for value in match.groups()]
                percent = numbers[0]
                rows[percent] = (numbers[7], numbers[8])
                if not all(math.isfinite(value) for value in numbers):
                    left_out.append(percent)
            elif (match := FLOORED.match(line)) and (int(match[1]) or int(match[2])):
                left_out.append(percent)
    if f0 is None:
        raise ValueError(f"{path} is not an output of tools/compare_mistuned_harmonic.py")
    for percent in left_out:
        rows.pop(percent, None)
    return f0, harmonic, rows, sorted(set(left_out))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()
    conditions, left_out = {}, []
    for path in args.outputs:
        f0, harmonic, rows, dropped = read(path)
        conditions.setdefault((f0, harmonic), {}).update(rows)
        left_out += [(f0, harmonic, percent) for percent in dropped]
    for f0, harmonic, percent in left_out:
        print(f"left out: f0 {f0:g} Hz, harmonic {harmonic}, {percent:g}% (unreliable evidence estimate)")
    print("f0_hz  harmonic  n_mistunings  threshold_laplace  threshold_is  p_two_at_max")
    for (f0, harmonic), rows in sorted(conditions.items()):
        percents = sorted(rows)
        laplace = [rows[percent][0] for percent in percents]
        importance = [rows[percent][1] for percent in percents]
        tau_laplace = threshold(percents, probability_from_log_odds(laplace))
        tau_importance = threshold(percents, probability_from_log_odds(importance))
        p_two = probability_from_log_odds(laplace[-1])
        print(
            f"{f0:5g}  {harmonic:8d}  {len(percents):12d}"
            f"  {tau_laplace:17.1f}  {tau_importance:12.1f}  {p_two:12.3f}"
        )


if __name__ == "__main__":
    main()
