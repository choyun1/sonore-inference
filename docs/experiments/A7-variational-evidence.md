# A7. Variational evidence on the full model

> **Under review.** This is a working record, not reviewed results. The work was
> done with AI assistance (Claude), and Adrian Cho has not yet checked it
> carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-10). **Paper target:** Fig. 7D (right); the
paper's inference is variational (App. B). **Code:** PR [#20](https://github.com/choyun1/sonore-inference/pull/20),
first on main at `73da5f7`.

## Question

Shared background: [how the experiments work](index.md#how-the-experiments-work).


Milestone (b) needed a more robust evidence estimate than Laplace (see B1).
Before using it there: does variational evidence, as the paper estimates
it, change milestone (a)'s thresholds?

## Setup

A5's full model through the scene module, plus a variational fit: a
diagonal Gaussian in Laplace-whitened coordinates, fitted by the ELBO
(Adam, learning rate 0.01, 10 draws per step, 2000 steps), then importance
sampling from it. Four conditions at the paper's 7 levels.

## Results

Thresholds, % of f0, by four evidence estimates. All four agree in every
condition.

| Condition | Paper's model (by eye) | Eqn 2 | BASS's rule |
| --- | --- | --- | --- |
| 100 Hz, harmonic 1 | about 45 | 25 | 30 |
| 200 Hz, harmonic 2 | about 20 | 15 | 20 |
| 200 Hz, harmonic 3 | about 21 | 25 | 30 |
| 400 Hz, harmonic 3 | about 13 | 15 | 20 |

The four estimates are Laplace, importance sampling from Laplace, the ELBO,
and importance sampling from the variational fit (the four log-odds
columns of each output in [`data/A7/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A7)). All 7 levels are used,
including 3 rows where a Laplace Hessian was floored. The Eqn 2 values are
`sonore_inference.thresholds` applied to those four columns (checked again
for this page). The BASS-rule values come from a local script in the
project's files that re-implements BASS's rule, so it is not committed.
`tools/mistuning_thresholds.py` does not read the extra columns and drops
floored rows, so it is not used here.

## What it says

The variational estimates give the same thresholds as Laplace here, so the
switch made for (b) does not move (a)'s results. The log odds differ most
where Laplace was invalid or large (200 Hz harmonic 3 at 50%: importance
sampling from Laplace +22578, from the variational fit +59).

## Reproduce

```
python tools/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --variational-steps 2000
```

## Open

- Importance sampling's effective sample size stays low (mostly 1–8 of 128).
- GP variance and lengthscale are still fixed, not inferred as in the paper.
