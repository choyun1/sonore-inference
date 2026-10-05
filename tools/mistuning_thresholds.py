"""Mistuning thresholds from the output of tools/compare_mistuned_harmonic.py.

For each output file (one f0 and harmonic), converts the log posterior odds
of two sources over one into probabilities and applies the threshold of
Eqn 2 (``sonore_inference.thresholds``), for the Laplace and the
importance-sampling estimates. A threshold of 50% means two sources were
never preferred, which the paper reports as not measurable.

    python tools/mistuning_thresholds.py results/f0-*_h*.txt
"""

import argparse
import re

from sonore_inference.thresholds import probability_from_log_odds, threshold

HEADER = re.compile(r"f0 (\S+) Hz, harmonic (\d+) mistuned")
ROW = re.compile(r"\s*(\d+(?:\.\d+)?)" + r"\s+(-?[\d.]+)" * 9 + r"\s*$")


def read(path):
    """(f0, harmonic, percents, log odds by Laplace, log odds by importance sampling) of one output file."""
    f0 = harmonic = None
    percents, laplace, importance = [], [], []
    with open(path) as lines:
        for line in lines:
            if match := HEADER.match(line):
                f0, harmonic = float(match[1]), int(match[2])
            elif match := ROW.match(line):
                numbers = [float(value) for value in match.groups()]
                percents.append(numbers[0])
                laplace.append(numbers[7])
                importance.append(numbers[8])
    return f0, harmonic, percents, laplace, importance


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()
    print("f0_hz  harmonic  threshold_laplace  threshold_is  p_two_at_max")
    for f0, harmonic, percents, laplace, importance in sorted(read(path) for path in args.outputs):
        tau_laplace = threshold(percents, probability_from_log_odds(laplace))
        tau_importance = threshold(percents, probability_from_log_odds(importance))
        p_two = probability_from_log_odds(laplace[-1])
        print(f"{f0:5g}  {harmonic:8d}  {tau_laplace:17.1f}  {tau_importance:12.1f}  {p_two:12.3f}")


if __name__ == "__main__":
    main()
