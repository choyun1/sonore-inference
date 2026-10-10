# A5. Full model: trajectories and harmonic timing

**Status:** done (2026-10-06); milestone (a) closed 2026-10-07 with the
100 Hz harmonic 1 gap recorded. **Paper target:** Fig. 7D (right).
**Code:** PR [#13](https://github.com/choyun1/sonore-inference/pull/13) (first on main at `ceccaee`); re-run
through the scene module of PR [#15](https://github.com/choyun1/sonore-inference/pull/15) (`393ed82`).

## Question

What is left after adding the rest of BASS's harmonic and whistle sources?

## Setup

As A4, plus:
- f0, whistle frequency and both levels as trajectories on a 10 ms grid,
  with the paper's Gaussian-process priors (hyperparameters fixed at the
  medians of Table A.2);
- the harmonic source's own onset and duration inferred.

## Results

Thresholds, % of f0 (`data/A5/thresholds.txt`). 48 of 225 rows left out
(a fit stopped at a point that was not a peak, so the Laplace estimate is
not valid there).

| f0 (Hz) | Harmonic | Paper's model (by eye) | A4 | A5, Laplace | A5, importance sampling | Levels used |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1 | about 45 | 23.5 | 26.6 | 26.2 | 25 |
| 100 | 2 | n.m. | n.m. | n.m. | n.m. | 24 |
| 100 | 3 | n.m. | n.m. | n.m. | n.m. | 24 |
| 200 | 1 | about 10 | 7.6 | 8.2 | 8.4 | 22 |
| 200 | 2 | about 20 | 13.6 | 15.7 | 15.0 | 20 |
| 200 | 3 | about 21 | 22.1 | 24.8 | 25.0 | 17 |
| 400 | 1 | about 5 | 4.5 | 4.0 | 4.0 | 17 |
| 400 | 2 | about 10 | 7.7 | 6.8 | 8.5 | 9 |
| 400 | 3 | about 13 | 12.5 | 13.0 | 12.4 | 19 |

The scene module of PR #15 reproduces all 225 rows identically as printed
(both evidence estimates, both effective sample sizes, both log odds).

## What it says

The two gaps narrow but do not close: 100 Hz harmonic 1 is 26.6% against
about 45, and 200 Hz harmonic 2 is 15.7% against about 20. 200 Hz harmonic 3
moves past the paper's (24.8 against about 21). The 400 Hz harmonic 2 value
rests on 9 of 25 levels. A6 shows that the 200 Hz harmonic 2 gap comes from
the threshold rule, and 100 Hz harmonic 1 from the model.

## Reproduce

Re-run on main through the scene module (identical outputs, PR #15):

```
python tools/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --percents 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50
python tools/compare_mistuned_harmonic.py --f0 100 --harmonic 1 --whistle-timing inferred \
    --trajectories --harmonic-timing inferred --percents 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50     # the original script
python tools/mistuning_thresholds.py data/A5/f0-*.txt
```

Raw outputs: [`data/A5/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A5).

## Open

- 100 Hz harmonic 1. Untested candidates: BASS infers each source's GP
  variance and lengthscale per sound (we fix them at the medians), and its
  evidence estimate differs from ours (see A6, A7).
- Many fits are left out (48 of 225), most at 400 Hz harmonic 2.
