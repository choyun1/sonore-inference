# A3. Whistle onset and duration inferred

> **Under review.** This is a working record, not reviewed results. The work was
> done with AI assistance (Claude), and Adrian Cho has not yet checked it
> carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-05). **Paper target:** Fig. 7D (right).
**Code:** PR [#12](https://github.com/choyun1/sonore-inference/pull/12), first on main at `12305f6`.

## Question

Shared background: [how the experiments work](index.md#how-the-experiments-work).


In the paper every event's onset and duration are unknowns with a prior. In
A1 and A2 the whistle was given the stimulus's timing for free. How much
does paying for it change the thresholds?

## Setup

As A2 (calibrated FFT cochleagram), plus: the whistle's onset (uniform over
the scene) and log duration (Student-t from the normal-gamma prior of
Table A.1) are inferred. Cosine component phases. Mistuning on a 25-level
grid (0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50).

## Results

Thresholds, % of f0 (`data/A3/thresholds.txt`). 5 rows left out.

| f0 (Hz) | Harmonic | Paper's model (by eye) | A2 | A3, Laplace | A3, importance sampling |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | about 45 | 12.2 | 15.0 | 14.9 |
| 100 | 2 | n.m. | 39.5 | n.m. | n.m. |
| 100 | 3 | n.m. | 41.6 | n.m. | n.m. |
| 200 | 1 | about 10 | 5.6 | 7.4 | 7.2 |
| 200 | 2 | about 20 | 11.1 | 13.5 | 13.2 |
| 200 | 3 | about 21 | 17.8 | 22.8 | 22.4 |
| 400 | 1 | about 5 | 2.6 | 3.6 | 3.5 |
| 400 | 2 | about 10 | 6.5 | 7.9 | 7.7 |
| 400 | 3 | about 13 | 9.5 | 12.9 | 12.6 |

## What it says

Inferring the whistle's timing lowers the two-source evidence by about 10
log units (median over the 189 rows A2 and A3 share: −10.7 in H2's Laplace
estimate; a direct comparison of the two data folders). Every threshold
rises. 7 of 9 conditions are now within about 3 points of the paper's
model, counting 100 Hz harmonics 2 and 3, which become not measurable as in
the paper. Still off: 100 Hz harmonic 1 (15.0 against about 45) and
200 Hz harmonic 2 (13.5 against about 20).

## Reproduce

Not re-run on today's main.

```
python tools/compare_mistuned_harmonic.py --f0 100 --harmonic 1 --cochleagram fft-calibrated \
    --whistle-timing inferred --percents 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50
python tools/mistuning_thresholds.py data/A3/f0-*.txt
```

Raw outputs: [`data/A3/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A3).

## Open

- 100 Hz harmonic 1 and 200 Hz harmonic 2 (taken up in A4 to A6).
