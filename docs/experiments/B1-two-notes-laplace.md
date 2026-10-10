# B1. Two simultaneous notes, Laplace evidence (first pass)

**Status:** done (2026-10-07), **superseded by [B2](B2-two-notes-variational.md)**:
the octave's margins here are within Laplace's own run-to-run noise.
**Paper target:** none. The paper has no two-note experiment; the closest
are its simultaneous-grouping illusions (§3.1.2, Fig. 7) and onset
asynchrony (Fig. 7E–F, App. C.7), which use different stimuli, so no values
are compared. **Code:** stimuli PR [#16](https://github.com/choyun1/sonore-inference/pull/16), scene module
PR [#15](https://github.com/choyun1/sonore-inference/pull/15), tool `tools/compare_two_notes.py` (on main since
PR [#23](https://github.com/choyun1/sonore-inference/pull/23), `121aab4`). Design: [milestone (b)](../design/milestone-b.md).

## Question

Can the model tell two harmonic notes sounding together from one note, and
how does that depend on the interval and on onset asynchrony?

## Setup

- Lower note 200 Hz; upper note at a tritone (282.8 Hz) with a falling
  spectrum (−6 dB per octave of harmonic number), a tritone with flat
  spectra, a just fifth (300 Hz) and an octave (400 Hz). Every harmonic up to
  2400 Hz at 60 dB; coinciding components add in phase.
- The upper note starts 0, 10, 20, 40 or 80 ms late and ends with the lower
  one (assumed; lower note 400 ms). Four controls play one note alone.
- Three hypotheses: one harmonic source (started at 200 Hz, at the upper f0
  and at 100 Hz; the best is kept), two harmonic sources, and one harmonic
  source plus a whistle for each upper component that is not a harmonic of
  200 Hz (8 at the tritones, 4 at the fifth, none at the octave).
- A5's full model (FFT cochleagram with BASS's gain, trajectories, all timing
  inferred). 3000 Adam steps per fit; Laplace and importance-sampling evidence.

## Results

Log posterior odds over the best one-source fit; positive favours the
alternative (`data/B1/summary.txt`, Laplace). `*` marks a value resting
on a fit with no true peak (floored Hessian), where Laplace is not valid.

Two sources vs one:

| Interval | 0 ms | 10 ms | 20 ms | 40 ms | 80 ms |
| --- | --- | --- | --- | --- | --- |
| Tritone, different spectra | 36.2 | 46.9* | 40.4 | 60.5* | 58.7* |
| Tritone | 26.9 | 31.4 | 32.7 | 40.0 | 52.6 |
| Just fifth | 70.2 | 70.6 | 75.9 | 84.0 | 77.3 |
| Octave | −4.9 | −6.6 | −15.7* | −7.4 | −5.7 |

Controls (one note split into two sources an octave apart, vs one): 200 Hz
−14.1, 282.8 Hz −1.5*, 300 Hz −19.7, 400 Hz +7.6*.

Harmonic plus whistles vs one: −104 to −179 at the tritones; −8.7 to +21.5
at the fifth, all but one resting on fits with no true peak.

## What it says

Tritones and the fifth come out as two sources at every asynchrony,
synchronous included. The octave comes out as one source up to 80 ms, but
by margins (5 to 16) no larger than the Laplace estimate's own movement
between 300 and 3000 steps (up to about 15 for the same start). Many fits
had no true peak even after 3000 steps; the cause found was a rugged
likelihood in the f0 trajectory. This is why B2 switched to variational
evidence (Cho's choice, 2026-10-10).

## Reproduce

```
python tools/compare_two_notes.py --interval octave --steps 3000 > octave.txt
python tools/compare_two_notes.py --controls --steps 3000 > controls.txt
python tools/two_notes_summary.py data/B1/*.txt
```

Raw outputs: [`data/B1/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/B1). The 300-step pass is in the project's
files, not here.

## Open

- Superseded by B2.
