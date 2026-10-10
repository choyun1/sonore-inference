# Computational experiments

Each experiment has an ID and its own page. A page states what the
experiment asks, which part of Cusimano, Hewitt & McDermott (2024),
*Listening with generative models* (Cognition 253, 105874, CC BY 4.0), it is
compared with, the paper's values as we read them, ours, the code and
command that made them, and what is still open.

Conventions on every page:

- **Paper values are read by eye** from the paper's figures, unless a page
  says otherwise. No figures are copied here; see the paper for them.
- **Thresholds** are in % of f0 and use Eqn 2 of the paper, read as
  threshold = lowest level + area under p(one source)
  ([`thresholds.py`](https://github.com/choyun1/sonore-inference/blob/main/src/sonore_inference/thresholds.py)).
  Some pages also give the value by the rule in BASS's analysis code; those
  numbers come from a local analysis that is not in this repository,
  because BASS's code has no licence.
- **One seed** for all our runs and for the BASS run. The paper averaged 10.
- Every number comes from a committed output under [`data/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data) and a
  script under `tools/`, or is marked as an estimate.

## Experiments

| ID | Experiment | Paper target | Status | Result in one line |
| --- | --- | --- | --- | --- |
| [A1](A1-gammatone-baseline.md) | Mistuned harmonic, gammatone cochleagram | Fig. 7D, App. C.6 | done | Thresholds 2.5–15.4%, below the paper's model |
| [A2](A2-fft-cochleagram.md) | + BASS's FFT cochleagram | Fig. 7D | done | Most of the gap closes; 100 Hz h2/h3 near 40% |
| [A3](A3-whistle-timing.md) | + whistle onset and duration inferred | Fig. 7D | done | 7 of 9 conditions near the paper; 100/1 and 200/2 still low |
| [A4](A4-bass-gain.md) | + BASS's uncalibrated channel gain | Fig. 7D | done | 100/1 rises from 15.0 to 23.5% |
| [A5](A5-full-model.md) | + f0 and level trajectories, harmonic timing (full model) | Fig. 7D | done | 100/1 26.6% (paper about 45), 200/2 15.7% (about 20) |
| [A6](A6-bass-own-code.md) | BASS's own code on two conditions | Fig. 7D | done | 200/2 gap was the threshold rule; 100/1 is a model gap |
| [A7](A7-variational-evidence.md) | Variational evidence on the full model | Fig. 7D, App. B | done | Same thresholds as Laplace on 4 conditions |
| [B1](B1-two-notes-laplace.md) | Two simultaneous notes, Laplace evidence | none (closest: §3.1.2) | superseded by B2 | Octave margins within Laplace's own noise |
| [B2](B2-two-notes-variational.md) | Two simultaneous notes, variational evidence | none (closest: §3.1.2) | done | Tritones and fifth: two notes; octave: one, up to 240 ms |
| [R1](R1-source-and-room.md) | Source and room (RT60) from one sound | none in this paper | in progress | Building blocks checked; inference not run yet |

## Milestone (a) at a glance

Mistuned-harmonic thresholds (% of f0, Eqn 2, Laplace estimate). Each column
adds one change to the one before. "n.m." is not measurable below 50%, as
the paper reports it.

| f0 (Hz) | Harmonic | Paper's model (Fig. 7D, by eye) | A1 | A2 | A3 | A4 | A5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 1 | about 45 | 6.7 | 12.2 | 15.0 | 23.5 | 26.6 |
| 100 | 2 | n.m. | 10.1 | 39.5 | n.m. | n.m. | n.m. |
| 100 | 3 | n.m. | 15.4 | 41.6 | n.m. | n.m. | n.m. |
| 200 | 1 | about 10 | 3.5 | 5.6 | 7.4 | 7.6 | 8.2 |
| 200 | 2 | about 20 | 7.4 | 11.1 | 13.5 | 13.6 | 15.7 |
| 200 | 3 | about 21 | 11.1 | 17.8 | 22.8 | 22.1 | 24.8 |
| 400 | 1 | about 5 | 2.5 | 2.6 | 3.6 | 4.5 | 4.0 |
| 400 | 2 | about 10 | 5.1 | 6.5 | 7.9 | 7.7 | 6.8 |
| 400 | 3 | about 13 | 7.1 | 9.5 | 12.9 | 12.5 | 13.0 |

Each column is the `threshold_laplace` column of that experiment's
`thresholds.txt`, made by `tools/mistuning_thresholds.py` from the committed
outputs. Rows whose evidence estimate was unreliable are left out first;
each page says how many.

Listeners in Moore, Glasberg & Peters (1986), shown in Fig. 7D (left), are
at about 1–5% of f0 (by eye), below every model column here and below the
paper's model.

## Design documents

The decisions behind these experiments are in
[design v1](../design/design-v1.md), [milestone (b)](../design/milestone-b.md)
and [source and room](../design/source-reverb.md).
