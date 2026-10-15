# B2. Two simultaneous notes, variational evidence

> **Under review.** This is a working record, not reviewed
> results. The work was done with AI assistance (Claude) and has not yet been
> checked carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-10). **Paper target:** none (see
[B1](B1-two-notes-laplace.md)); no paper values are compared.
**Code:** `tools/bass/compare_two_notes.py --variational-steps`, PR
[#23](https://github.com/choyun1/sonore-infer/pull/23) (`121aab4`), using the variational evidence of
PR [#20](https://github.com/choyun1/sonore-infer/pull/20). Design: [milestone (b)](../docs/design/milestone-b.md).

## Question

Shared background: [how the experiments work](README.md#how-the-experiments-work).


B1's question, with an evidence estimate that does not depend on reaching
a true peak, and with longer octave asynchronies (120, 160, 240 ms).

## Setup

As B1, but 300 Adam steps then a variational fit (2000 steps), as in A7.
The main estimate is importance sampling from the variational fit; the ELBO
is reported as a check.

## Results

Log posterior odds over the best one-source fit, variational estimate
(`data/B2/summary.txt`); positive favours the alternative.

Two sources vs one:

| Interval | 0 ms | 10 | 20 | 40 | 80 | 120 | 160 | 240 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tritone, different spectra | 44.0 | 39.2 | 49.9 | 57.0 | 48.4 | | | |
| Tritone | 31.1 | 35.7 | 45.2 | 48.4 | 56.4 | | | |
| Just fifth | 56.0 | 58.6 | 61.9 | 66.1 | 54.4 | | | |
| Octave | −33.6 | −33.1 | −28.3 | −27.0 | −26.3 | −23.4 | −25.4 | −21.9 |

Controls (one note split into two sources an octave apart, vs one): 200 Hz
−19.5, 282.8 Hz −17.2, 300 Hz −19.8, 400 Hz −26.2.

Harmonic plus whistles vs one: −57.0 to −125.3 at the tritones; +0.1 to
+10.5 at the just fifth (ELBO −3.1 to +8.9), where two harmonic sources beat
whistles by 50 to 59.

Asynchrony thresholds by Eqn 2: 0 ms for both tritones and the fifth; none
within 240 ms for the octave (Eqn 2 returns the largest asynchrony).

## Plots

![B2-two-notes-variational](plots/B2-two-notes-variational.svg)

*Log odds of two sources over the best one-source fit (variational estimate), from [`data/B2/summary.txt`](data/B2/summary.txt). Above zero favors two notes.*

## What it says

- Tritones and the just fifth: two sources at every asynchrony, synchronous
  included.
- Octave: one source at every asynchrony up to 240 ms, by 22 to 34. The odds
  rise slowly with asynchrony.
- Controls: one source, including 200 and 400 Hz, where B1's Laplace
  estimate had favoured two.
- At the fifth, whistles are about as good as one source; two notes are far
  ahead of both.
- At the octave, two sources fit better at their mode (log joint about
  −9906 against −9919 to −9926 for one), but not by enough to pay for the
  second source. One source's fit hardly worsens with asynchrony; my
  reading, not tested, is that with the likelihood's 10 dB noise (BASS's
  value) the late +6 dB on shared harmonics costs little.

The design expected the octave to become two sources with enough
asynchrony (milestone (b), B5). Up to 240 ms it did not.

## Fits run to convergence (2026-10-15)

The fits above stop after 300 Adam steps. In [C2](C2-aba-streaming.md) that
left an ABA one-stream fit far from its mode, so all 27 stimuli were rerun
with each fit continuing until its log joint rose by less than 0.5 over 1000
steps (at most 15000), seed 0, on the same cloud machine
(`data/B2/converged/`, summary in `data/B2/converged/summary.txt`;
`tools/run_seeds.py --seeds 0 --tolerance 0.5`, commit c3a2968).

Every sign is unchanged, and so is every Eqn 2 threshold: 0 ms for the
tritones and the fifth, none within 240 ms for the octave. Most variational
log odds move by less than 4, within seed noise ([B4](B4-two-notes-seeds.md)).
The larger moves:

- 400 Hz control: −26.2 to −12.1, still one source.
- Harmonic plus whistles against one source at the tritones rise by 4 to 13,
  most of which is the structure prior's labelling count (decision C6), new
  since these outputs were made, not the fits.
- Fifth, whistles against one source at 10 ms: 0.1 to 7.7; the fifth's
  whistles stay far behind two notes (by 47 to 52).

Milestone (b)'s conclusions do not depend on the 300-step fits.

## Reproduce

```
python tools/bass/compare_two_notes.py --interval octave --variational-steps 2000 \
    --only one@400 one@200 one@100 two > octave.txt
python tools/bass/compare_two_notes.py --interval octave --asynchronies 0.12 0.16 0.24 \
    --variational-steps 2000 --only one@400 one@200 one@100 two > octave_long.txt
python tools/bass/compare_two_notes.py --interval just_fifth --variational-steps 2000 --only whistles > just_fifth_whistles.txt
python tools/bass/two_notes_summary.py data/B2/*.txt
```

Raw outputs: [`data/B2/`](data/B2/).

## Open

- One seed and one variational fit per hypothesis. Two-source ELBOs were
  still rising slowly at 2000 steps (1 to 2 more by 4000 steps at the
  octave, 20 ms), so the octave's margin may be a few units smaller.
- Importance sampling's effective sample size is 1 to 6 of 256 for most fits.
- Not compared with listeners, and BASS was not run on these stimuli
  (decision B6 of [milestone (b)](../docs/design/milestone-b.md)).
- Followed up by [B3](B3-octave-sigma5.md) (likelihood SD 5 dB), [B4](B4-two-notes-seeds.md) (seeds) and [C1](C1-kernel-inferred.md) (kernel inferred).
