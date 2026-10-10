"""Tables of log odds from the output of tools/compare_two_notes.py.

Reads any number of its outputs (fits of the same stimulus may be split across
files, for example the harmonic-plus-whistles fits run separately) and prints,
for each evidence estimate, the log posterior odds of ``two`` and ``whistles``
over the best one-source fit, as compare_two_notes.py does for the fits in one
file. Values are recomputed from the printed one-decimal numbers, so they can
differ from that script's own lines by 0.1. A ``*`` marks a Laplace value
that rests on a fit whose Hessian had floored eigenvalues (no true peak).

For each interval it also prints the asynchrony threshold by Eqn 2
(``sonore_inference.thresholds``), with ``one`` as H0 and ``two`` as H1; the
largest asynchrony means two sources were never preferred (not measurable).

    python tools/two_notes_summary.py results/*.txt
"""

import argparse
import re

import numpy as np

from sonore_inference.thresholds import probability_from_log_odds, threshold

ROW = re.compile(r"(?P<stimulus>\S.*?\S)\s{2,}(?P<hypothesis>one@\S+|two|whistles)\s+(?P<numbers>-?\d.*)$")
STIMULUS = re.compile(r"(?P<interval>\S+) (?P<value>[\d.]+) (?P<unit>ms|Hz)$")
ESTIMATES = {"laplace": 2, "importance": 3, "elbo": 8, "variational": 9}


def read(paths):
    """{stimulus: {hypothesis: {estimate: log posterior}}}, and the floored fits."""
    fits, floored = {}, set()
    for path in paths:
        with open(path) as lines:
            for line in lines:
                match = ROW.match(line.rstrip())
                if not match:
                    continue
                numbers = [float(value) for value in match["numbers"].split()]
                structure = numbers[1]  # numbers[0] is n_params
                posteriors = {
                    estimate: structure + numbers[column]
                    for estimate, column in ESTIMATES.items()
                    if column < len(numbers)
                }
                fits.setdefault(match["stimulus"], {})[match["hypothesis"]] = posteriors
                if numbers[5]:
                    floored.add((match["stimulus"], match["hypothesis"]))
    return fits, floored


def odds(fits, floored, estimate):
    """{stimulus: {alternative: (log odds over the best one source, floored)}}."""
    table = {}
    for stimulus, hypotheses in fits.items():
        ones = {h: p[estimate] for h, p in hypotheses.items() if h.startswith("one@") and estimate in p}
        if not ones:
            continue
        best = max(ones, key=ones.get)
        for alternative in ("two", "whistles"):
            if alternative in hypotheses and estimate in hypotheses[alternative]:
                rests_on_floored = estimate == "laplace" and bool(
                    {(stimulus, best), (stimulus, alternative)} & floored
                )
                value = hypotheses[alternative][estimate] - ones[best]
                table.setdefault(stimulus, {})[alternative] = (value, rests_on_floored)
    return table


def sort_key(stimulus):
    match = STIMULUS.match(stimulus)
    return (match["interval"], float(match["value"])) if match else (stimulus, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", nargs="+")
    args = parser.parse_args()
    fits, floored = read(args.outputs)
    for estimate in ESTIMATES:
        table = odds(fits, floored, estimate)
        if not table:
            continue
        print(f"== {estimate} ==")
        print(f"{'stimulus':34s}  {'two':>8s}  {'whistles':>8s}")
        curves = {}
        for stimulus in sorted(table, key=sort_key):
            cells = []
            for alternative in ("two", "whistles"):
                value, mark = table[stimulus].get(alternative, (None, False))
                cells.append(" " * 8 if value is None else f"{value:7.1f}" + ("*" if mark else " "))
            print(f"{stimulus:34s}  {cells[0]}  {cells[1]}")
            match = STIMULUS.match(stimulus)
            if match and match["unit"] == "ms" and "two" in table[stimulus]:
                curves.setdefault(match["interval"], []).append(
                    (float(match["value"]), table[stimulus]["two"][0])
                )
        for interval, points in curves.items():
            points.sort()
            asynchronies = [a for a, _ in points]
            with np.errstate(over="ignore"):  # odds of thousands saturate to probability 0 or 1
                probability = probability_from_log_odds([value for _, value in points])
            tau = threshold(asynchronies, probability)
            print(
                f"  {interval}: Eqn 2 asynchrony threshold {tau:.1f} ms"
                f" over {asynchronies[0]:g}-{asynchronies[-1]:g} ms"
            )
        print()


if __name__ == "__main__":
    main()
