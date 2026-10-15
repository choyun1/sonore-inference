# A7. Variational evidence on the full model

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-10). **Paper target:** Fig. 7D (right); the
paper's inference is variational (App. B). **Code:** PR [#20](https://github.com/choyun1/sonore-infer/pull/20),
first on main at `73da5f7`.

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).


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
columns of each output in [`data/A7/`](data/A7/)). All 7 levels are used,
including 3 rows where a Laplace Hessian was floored. The Eqn 2 values are
`sonore_infer.thresholds` applied to those four columns (checked again
for this page). The BASS-rule values come from a local script in the
project's files that re-implements BASS's rule, so it is not committed.
`tools/bass/mistuning_thresholds.py` does not read the extra columns and drops
floored rows, so it is not used here.

## Plots

![A7-variational-evidence](plots/A7-variational-evidence.svg)

*p(two) by four evidence estimates at the paper's 7 levels, from [`data/A7/`](data/A7/). The four lines mostly lie on top of each other.*

## What it says

The variational estimates give the same thresholds as Laplace here, so the
switch made for (b) does not move (a)'s results. The log odds differ most
where Laplace was invalid or large (200 Hz harmonic 3 at 50%: importance
sampling from Laplace +22578, from the variational fit +59).

## Fits run to convergence (2026-10-15)

The fits above stop after 300 Adam steps. In [C2](C2-aba-streaming.md) that
left an ABA one-stream fit far from its mode, so the same conditions were
rerun with each fit continuing until its log joint rose by less than 0.5 over
1000 steps (at most 15000), seed 0, on the same cloud machine, on a 0-30%
grid in steps of 5 (`data/A7/converged/`, `tools/run_seeds.py --seeds 0
--tolerance 0.5`, commit c3a2968).

Eqn 2 thresholds (%), Laplace / variational:

| Condition | 300 steps, shared points | Converged, shared points | Converged, 0-30% in 5s |
| --- | --- | --- | --- |
| 100 Hz, harmonic 1 | 25.0 / 25.0 | 25.0 / 25.2 | 27.4 / 25.2 |
| 200 Hz, harmonic 2 | 15.0 / 15.0 | 15.0 / 15.0 | 17.5 / 12.5 |
| 200 Hz, harmonic 3 | 25.0 / 25.0 | 25.0 / 24.9 | 22.8 / 22.5 |
| 400 Hz, harmonic 3 | 15.0 / 15.0 | 15.0 / 12.8 | 12.5 / 11.1 |

"Shared points" are the mistunings both runs have: 0, 5, 10, 20 and 30%.
On them only 400 Hz harmonic 3's variational threshold moves (by 2.2); the
finer grid moves the others by up to 2.5, the size of its step. Every log
odds at a shared point has the same sign as with 300 steps. Two values moved
by more than seed noise: 100 Hz harmonic 1 at 10%, variational −13.9 to
−28.9 (still one source), and at 25% the variational estimate failed (−inf;
Laplace −3.9). Milestone (a)'s conclusions do not depend on the 300-step
fits.

## Reproduce

```
python tools/bass/scene_mistuned_harmonic.py --f0 100 --harmonic 1 --variational-steps 2000
```

## Open

- Importance sampling's effective sample size stays low (mostly 1–8 of 128).
- GP variance and lengthscale are still fixed, not inferred as in the paper.
- Ten seeds of the same run: [A8](A8-ten-seeds.md). Kernel hyperparameters inferred: [C1](C1-kernel-inferred.md).
