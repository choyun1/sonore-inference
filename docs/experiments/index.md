# Computational experiments

> **Under review.** This is a working record, not reviewed results. The work was
> done with AI assistance (Claude), and Adrian Cho has not yet checked it
> carefully. Numbers, methods and interpretations may change.

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

## How the experiments work

### The main idea

The paper treats hearing as inference in a generative model. The model
says how a scene is made: a few sound sources, each of a type (harmonic
tone, whistle, noise), each with its own unknowns (f0, level, spectrum,
onset, duration, and how these change over time), each with a prior. A
renderer turns a scene into a waveform, and a cochleagram turns the
waveform into a time-frequency picture. Hearing a sound means finding the
scenes that could have made it.

To ask "one source or two?", each answer is a separate hypothesis with its
own set of sources. For each hypothesis we find the best-fitting scene and
estimate its **marginal likelihood** (the evidence): how well the
hypothesis explains the sound, averaged over its unknowns. Extra sources
fit better but cost prior probability, so the evidence includes an Occam
penalty. The **log posterior odds** of two sources over one is the
difference of the two log evidences plus the log prior of each structure.
Positive favours two sources.

Sweeping a stimulus parameter (mistuning, onset asynchrony) gives a curve
of odds, and the paper's Eqn 2 turns that curve into a **threshold**: the
point where the model switches from hearing one source to two. These
thresholds are what the paper compares with listeners, and what we
compare with the paper's model.

The paper's own code is BASS (github.com/mcusi/bass). We re-implement the
model from the paper alone, because BASS has no licence. We ran BASS only
locally, as a check (A6).

### The common setup

What every experiment here shares unless its page says otherwise:

- **Representation.** A cochleagram of 64 channels from 20 to 9423 Hz, in
  dB with a 20 dB floor, 10 ms frames. A1 uses true gammatone filters. From
  A2 on it is BASS's FFT approximation, and from A4 on with BASS's
  uncalibrated channel gain.
- **Likelihood.** Each cochleagram cell is Gaussian around the rendered
  scene's value, with an SD of 10 dB (BASS's value).
- **Priors.** The paper's (Tables A.1 and A.2): the number of sources
  (Poisson, 1 per second, at least one), source types, frequencies, levels
  and the harmonic spectrum. Gaussian-process hyperparameters are fixed at
  the medians of Table A.2, where the paper infers them per sound.
- **Fitting.** For each hypothesis, Adam gradient ascent on the log
  posterior from the starting points the paper gives (App. C), keeping the
  best mode. The question is the evidence for each hypothesis, not whether
  a blind search finds it.
- **Evidence.** Laplace (a Gaussian at the mode, from the Hessian) and
  importance sampling from that Gaussian. From A7 on, also a variational
  fit, as the paper does (App. B), and importance sampling from it. A
  Laplace estimate is left out where the fit has no true peak (the Hessian
  had to be floored).
- **Thresholds.** Eqn 2 as we read it: lowest level plus the area under
  p(one source). Where noted, also the rule in BASS's analysis code.

### The three groups

- **A, mistuned harmonic (milestone (a)).** Reproduce the paper's Fig. 7D.
  A complex tone (f0 100, 200 or 400 Hz, equal 60 dB harmonics, 400 ms)
  with harmonic 1, 2 or 3 mistuned by 0 to 50% of f0. Hypotheses: one
  harmonic source, or a harmonic source plus a whistle at the mistuned
  component. A1 to A5 add BASS's details one at a time, to find what makes
  the thresholds match. A6 runs BASS itself, and A7 checks the evidence
  method.
- **B, two simultaneous notes (milestone (b)).** Not in the paper. A
  200 Hz note with a second note a tritone, a just fifth or an octave
  above, starting 0 to 240 ms later. Hypotheses: one harmonic source, two,
  or one plus whistles. The question is whether the model hears two notes
  and how much onset asynchrony it needs.
- **R, source and room.** Not in the paper. One noise-like source in one
  room. A different model: the source is a Gaussian spectrogram
  (McDermott, Wrobleski & Oxenham, 2011), the room is a band-by-band
  exponential decay (RT60) with direct sound, and the likelihood is on band
  power in 10 ms blocks. The question is whether RT60 and the source can
  be told apart from one sound.

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
