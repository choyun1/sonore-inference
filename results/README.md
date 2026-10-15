# Computational experiments

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

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
  ([`thresholds.py`](../src/sonore_infer/thresholds.py)).
  Some pages also give the value by the rule in BASS's analysis code; those
  numbers come from a local analysis that is not in this repository,
  because BASS's code has no license.
- **One seed** for our runs and for the BASS run, except A8 and B4, which
  average 10 seeds as the paper did.
- Every number comes from a committed output under [`data/`](data/) and a
  script under `tools/`, or is marked as an estimate.
- **Plots** in [`plots/`](plots/) are drawn from those outputs by
  `tools/results_plots.py`; rerun it after changing any output.

**Experiment IDs and decision numbers.** The design documents number their
decisions with the same letters (decision B1 is in milestone-b.md, decision
C1 in milestone-c.md, decision R1 in source-reverb.md). On these pages a bare
ID such as B1 is an experiment; a decision is always written "decision B1".
Inside a design document a bare number is one of its decisions; elsewhere
both are written with their word.

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
model from the paper alone, because BASS has no license. We ran BASS only
locally, as a check (A6).

### The common setup

What every experiment here shares unless its page says otherwise:

- **Representation.** A cochleagram of 64 channels from 20 to 9423 Hz, in
  dB with a 20 dB floor, 25 ms time windows every 10 ms. A1 uses true
  gammatone filters. From A2 on it is BASS's FFT approximation, and from
  A4 on with BASS's uncalibrated channel gain.
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

### The four groups

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
  be told apart from one sound (R1), and what the model hears in rooms
  that break its assumptions (R2).
- **C, scene grammar (milestone (c)).** The model grows toward the
  paper's full grammar (several events per source, inferred
  Gaussian-process hyperparameters) and is rerun on A and B, then on the
  paper's further experiments, starting with ABA streaming.

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
| [A8](A8-ten-seeds.md) | A7 over ten seeds | Fig. 7D | done | No threshold changes; 100/1 gap is not seed noise |
| [B1](B1-two-notes-laplace.md) | Two simultaneous notes, Laplace evidence | none (closest: §3.1.2) | superseded by B2 | Octave margins within Laplace's own noise |
| [B2](B2-two-notes-variational.md) | Two simultaneous notes, variational evidence | none (closest: §3.1.2) | done | Tritones and fifth: two notes; octave: one, up to 240 ms |
| [B3](B3-octave-sigma5.md) | Octave with likelihood SD 5 dB | none | done | Still one source up to 240 ms; odds rise faster |
| [B4](B4-two-notes-seeds.md) | B2 over seeds (5 on close cases, then 10 on all) | none | done | Every sign unchanged; seed SD 1–5 |
| [C1](C1-kernel-inferred.md) | GP sigma and lengthscale inferred (100/1 and octave) | Fig. 7D, Table A.2 | done | 100/1 still 25%; octave still one source |
| [C2](C2-aba-streaming.md) | ABA sequences, one stream or two (30 stimuli) | Fig. 8D, App. C.10 | map done; seeds 0–4 on the 2 cells under 10 | One stream at 1 semitone, two from 3 up (paper: from about 6) |
| [C3](C3-search.md) | Search over structures from detected events | App. B.2 | first pass on 4 (b) and 3 ABA stimuli | Same structure as enumerated on 6 of 7; fifth at 0 ms gives note + whistle |
| [O1](O1-online-buildup.md) | Buildup of streaming, batch and online | Fig. 8F, App. C.11 | first pass, sweeps | Converged batch odds rise with triplets; online with per-window convergence reproduces them in one pass; the lag, not the window, sets the slope |
| [R1](R1-source-and-room.md) | Source and room (RT60) from one sound | none in this paper | done | RT60 within 3%; 90% intervals cover 66% (overconfident) |
| [R2](R2-atypical-rooms.md) | Atypical rooms, exponential room model | none in this paper | done | Biased like T20 × 3; time-reversed rooms do not fit |

## Against the paper

Where our results match Cusimano et al. (2024), where they differ, and
which differences we chose not to chase. Our code, seeds and machines
differ from the paper's, and BASS's code cannot be copied, so matching the
paper is not a goal in itself; listeners are the reference. A difference
is chased only if it is larger than seed and machine noise (seed SD 1–5 in
[B4](B4-two-notes-seeds.md) and [A8](A8-ten-seeds.md), a few more between
machines) and either suggests our own inference is not doing what we claim,
or changes how the model compares with listeners.

| Comparison | Paper | Ours | Status |
| --- | --- | --- | --- |
| Mistuned harmonic thresholds, 8 of 9 conditions (Fig. 7D) | by eye | within one grid step ([A5](A5-full-model.md)–[A8](A8-ten-seeds.md)) | **matches** |
| Mistuned harmonic, 100 Hz harmonic 1 | about 45% (BASS's code, one seed: 40%) | 25% Eqn 2, 30% BASS's rule ([A8](A8-ten-seeds.md), [C1](C1-kernel-inferred.md)) | **differs, not chased**: listeners are about 1–5%, closer to ours; the cause is in BASS's code, which we cannot copy |
| Threshold rule (Eqn 2) | Eqn 2 as printed; BASS's analysis code uses another rule ([A6](A6-bass-own-code.md)) | both reported | **differs by method**; both kept on every page |
| Evidence method | variational, 250 steps per hypothesis then full fits for the top 5 | Laplace and variational importance ([A7](A7-variational-evidence.md)) | **differs by method**; A7 found the same thresholds by both |
| Gaussian-process sigma and lengthscale inferred (Table A.2) | inferred per sound | inferred in C1 and C2, fixed at Table A.2's medians elsewhere | **matches where inferred**; Table A.2's quartiles imply an f0 lengthscale bound of 0.01 s, not the 0.1 s printed ([C1](C1-kernel-inferred.md)) |
| ABA streaming, sign at 1 and 6–12 semitones (Fig. 8D) | one stream at 1, two from about 6 | the same ([C2](C2-aba-streaming.md)) | **matches** |
| ABA streaming, 3 semitones | slightly one stream (about −5) | two streams (+8 to +13, seeds 0–4) | **differs, not chased**: listeners are near the two-stream boundary there, so neither sign is evidence |
| ABA streaming, size of the log odds and its fall with slower rates | changes little across intervals (its color scale stops at ±20) | up to +217, falling with slower rates | **differs, deferred**: matters only once graded listener data are compared |
| Buildup of streaming (Fig. 8F) | odds rise with triplets | see [O1](O1-online-buildup.md) | see that page |
| Search over structures (App. B.2) | learned proposals, no per-stimulus results reported | hand-built detector ([C3](C3-search.md)) | **not applicable**: no paper numbers to compare; the test is against our own enumeration |
| Two simultaneous notes (B), source and room (R) | not in the paper | | **not applicable** |

One check is of our own inference, not of the paper: milestones (a) and (b)
fitted each hypothesis with 300 Adam steps, which in C2 left the one-stream
fit far from its mode. Rerun with fits to convergence (2026-10-15), every
sign and every threshold on the shared grid is unchanged, except 400 Hz
harmonic 3's variational threshold (15.0 to 12.8%); see
[A7](A7-variational-evidence.md#fits-run-to-convergence-2026-10-15) and
[B2](B2-two-notes-variational.md#fits-run-to-convergence-2026-10-15).

## Milestone (a) at a glance

![Milestone (a) thresholds by experiment](plots/milestone-a-thresholds.svg)

*Each dot is one experiment's threshold (Eqn 2, Laplace); darker is later in the sequence. Not measurable (n.m.) is drawn at 50. The diamonds are the paper's model, read by eye. Drawn by `tools/results_plots.py`.*

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
`thresholds.txt`, made by `tools/bass/mistuning_thresholds.py` from the committed
outputs. Rows whose evidence estimate was unreliable are left out first;
each page says how many.

Listeners in Moore, Glasberg & Peters (1986), shown in Fig. 7D (left), are
at about 1–5% of f0 (by eye), below every model column here and below the
paper's model.

## Design documents

The decisions behind these experiments are in
[design v1](../docs/design/design-v1.md), [milestone (b)](../docs/design/milestone-b.md),
[milestone (c)](../docs/design/milestone-c.md) and
[source and room](../docs/design/source-reverb.md).

### Paths cited in the design documents

The design documents were written before these pages were in the repository
and cite outputs by their path in the project's working folder. This table
says where each one is now.

| Path cited | Now in |
|---|---|
| `results/two_notes/README.md` | [B1](B1-two-notes-laplace.md), [B2](B2-two-notes-variational.md) |
| `results/two_notes/variational/tritone.txt` | [`data/B2/tritone.txt`](data/B2/tritone.txt) |
| `results/two_notes/sigma5/README.md` | [B3](B3-octave-sigma5.md) |
| `results/two_notes/seeds/` | [B4](B4-two-notes-seeds.md) |
| `source-reverb/results/recovery_summary.txt` | [`data/R1/recovery_summary.txt`](data/R1/recovery_summary.txt) |
| `source-reverb/results/atypical_rooms_summary.txt` | [`data/R2/atypical_rooms_summary.txt`](data/R2/atypical_rooms_summary.txt) |
| `results/aba/raw/` | not here yet; it belongs to the milestone (c) branch |
