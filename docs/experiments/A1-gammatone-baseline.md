# A1. Mistuned harmonic with a gammatone cochleagram

> **Under review.** This is a working record, not reviewed results. The work was
> done with AI assistance (Claude), and Adrian Cho has not yet checked it
> carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-05). **Paper target:** Fig. 7D (right), stimuli
and analysis of App. C.6. **Code:** PRs [#8](https://github.com/choyun1/sonore-inference/pull/8) (priors, evidence)
and [#9](https://github.com/choyun1/sonore-inference/pull/9) (thresholds), first on main at `ce484ce`.

## Question

Shared background: [how the experiments work](index.md#how-the-experiments-work).


Does a re-implementation built only from the paper show a mistuned-harmonic
effect, and at what thresholds?

## Setup

- Stimuli as App. C.6: f0 100, 200 and 400 Hz; harmonic 1, 2 or 3 mistuned;
  60 dB components, 400 ms.
- Two hypotheses: one harmonic source (H1), or a harmonic source plus one
  whistle (H2). Fitted by 300 Adam steps from the paper's two starting points.
- Cochleagram: 64 true half-ERB gammatone channels, 20–9423 Hz, 20 dB floor.
- Priors from the paper; GP hyperparameters fixed at the medians of
  Table A.2. Whistle timing given (equal to the stimulus's).
- Evidence by Laplace and by importance sampling from it (128 samples).
- Mistuning: the paper's 7 levels (0–50%), plus a 1% grid inside each crossing.

## Results

Thresholds, % of f0 (`data/A1/thresholds.txt`). 1 row left out (unreliable
evidence estimate).

| f0 (Hz) | Harmonic | Paper's model (by eye) | Ours, Laplace | Ours, importance sampling | Levels used |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | about 45 | 6.7 | 6.6 | 11 |
| 100 | 2 | n.m. | 10.1 | 10.1 | 16 |
| 100 | 3 | n.m. | 15.4 | 15.3 | 16 |
| 200 | 1 | about 10 | 3.5 | 3.5 | 11 |
| 200 | 2 | about 20 | 7.4 | 7.3 | 11 |
| 200 | 3 | about 21 | 11.1 | 10.9 | 16 |
| 400 | 1 | about 5 | 2.5 | 2.5 | 10 |
| 400 | 2 | about 10 | 5.1 | 4.9 | 15 |
| 400 | 3 | about 13 | 7.1 | 6.8 | 11 |

## What it says

There is a mistuned-harmonic effect, with thresholds that rise with harmonic
number and fall with f0, but they are well below the paper's model
everywhere, and still above listeners (about 1–5%, Fig. 7D left, by eye).
Where the paper's model cannot measure a threshold (100 Hz, harmonics 2 and
3), ours can. A2 to A5 trace the gap.

## Reproduce

On today's main the gammatone cochleagram is no longer the default; select
it with `--cochleagram gammatone`. Not re-run on today's main.

```
python tools/compare_mistuned_harmonic.py --f0 200 --harmonic 3 --cochleagram gammatone > f0-200_h3.txt
python tools/compare_mistuned_harmonic.py --f0 200 --harmonic 3 --cochleagram gammatone \
    --percents 11 12 13 14 15 16 17 18 19 > f0-200_h3_fine.txt
python tools/mistuning_thresholds.py data/A1/coarse/f0-*.txt data/A1/fine/f0-*.txt
```

Raw outputs: [`data/A1/coarse/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A1/coarse) (7 levels) and
[`data/A1/fine/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A1/fine) (the 1% grids).

## Open

- Importance sampling's effective sample size for H2 ranges from 5 to 112 of 128, so the importance estimate sometimes rests on few samples. The two estimates still agree closely.
