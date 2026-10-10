# Milestone (b): two simultaneous notes — design

Status: accepted by Cho on 2026-10-07, all as recommended except B2, where Cho
chose to include option (B) as well. Follows design v1, D3 (b‴).

## 1. Goal

Can the model tell two harmonic notes sounding together from one note? As in
milestone (a), the answer is the log posterior odds of two sources over one,
from each hypothesis's marginal likelihood. Design D3 grades the stimuli from
easy to hard: an unrelated interval with different spectra, the same with
equal spectra, a just fifth (3:2), and an octave.

What carries over from (a) unchanged: the FFT cochleagram with BASS's gain,
the Gaussian likelihood, the paper's priors with GP hyperparameters fixed at
the Table A.2 medians, f0 and level trajectories, inferred onsets and
durations, MAP fits with Laplace and importance-sampling evidence, float64.

## 2. Decisions

### B1. Stimuli

**Options.**

- **(A) Fixed set.** The four D3 steps once each, synchronous onsets.
- **(B) Each step swept along one cue.** The four D3 steps, each at onset
  asynchronies of 0, 10, 20, 40 and 80 ms (the upper note starts later). This
  gives a log-odds curve and a threshold per interval, as mistuning did in (a).
- **(C) Interval swept continuously** (for example 0–12 semitones in
  half-semitone steps), synchronous.

**Recommendation: (B).** It keeps (a)'s method (a threshold along one
parameter), and onset asynchrony is the cue D3 names for the octave, where
nothing else can separate the notes. Proposed values, all assumptions:

| Parameter | Value | Why |
| --- | --- | --- |
| Lower note f0 | 200 Hz | Where (a) matched the paper best |
| Upper note f0 | 282.8 Hz (tritone), 300 Hz (just fifth), 400 Hz (octave) | D3 steps |
| Harmonics | All up to 2400 Hz, equal level: 12, 8, 8 and 6 for 200, 282.8, 300 and 400 Hz | One ceiling for both notes, so that at the octave every upper component lands on a lower harmonic; 2400 Hz is the mistuned harmonic's top at 200 Hz |
| Level | 60 dB per component | As (a) |
| Spectra, step 1 | lower flat; upper falling 6 dB per octave of harmonic number | "Different spectra" |
| Spectra, steps 2–4 | both flat | "Same spectrum" |
| Duration | 400 ms each note, 10 ms ramps, cosine phases | As (a) |
| Asynchrony | 0, 10, 20, 40, 80 ms, upper note late, both end together | Cue for steps 3–4 |

That is 4 intervals × 5 asynchronies = 20 stimuli, plus 4 single-note
controls (each upper note alone and the lower note alone), which must come out
as one source.

### B2. Hypotheses

**Options.**

- **(A) One harmonic source against two.**
- **(B) Also a harmonic source plus whistles**, BASS's alternative explanation
  in (a).

**Recommendation: (A).** With several components per note, explaining one
note as whistles costs far more prior mass than a second harmonic source.

**Decision (Cho, 2026-10-07): (A) and (B), three hypotheses**, as BASS
compared h, hw and hwq in (a), so the guess above is checked rather than
assumed. (B) is one harmonic source plus one whistle for each component of the
upper note that does not coincide with a harmonic of the lower note: 8
whistles for the tritone, 4 for the fifth (harmonics 1, 3, 5 and 7 of 300 Hz).
At the octave none is left, so (B) equals one source and is not fitted.

One known competitor for one source: at the just fifth, every component is a
harmonic of 100 Hz, so one source at 100 Hz with some harmonics missing
explains the sound exactly. The spectrum GP's smoothness prior is what should
penalize that comb, which makes step 3 a direct test of the spectrum prior.
So one source is fitted from both 200 Hz and 100 Hz starting points.

### B3. Fitting and starting points

**Options.**

- **(A) Start at the true parameters**, plus the known alternatives (B2's
  100 Hz; octave errors for each note), keep the best posterior mode.
- **(B) Blind search**: a grid of f0 pairs, no knowledge of the stimulus.

**Recommendation: (A).** The question is the evidence for each hypothesis,
not whether search finds the notes; (a) worked the same way. Blind search is
its own problem (the paper reports octave errors), better taken up in (c).

### B4. Where the scene code lives

**Options.**

- **(A) Keep hypotheses in tools/ scripts**, as `compare_mistuned_harmonic.py`
  does now (352 lines, one-off).
- **(B) Move a general scene into the package**: a list of sources (harmonic
  or whistle), each with its own parameters, prior and renderer, plus one
  function that fits a scene and returns its evidence. Tools then only build
  stimuli and hypotheses.

**Recommendation: (B).** Two harmonic sources would otherwise mean copying
most of the mistuned-harmonic script, and milestone (c) needs general scenes
anyway. The mistuned-harmonic results are re-run through the new code as its
test: same log odds to within fitting noise.

### B5. What counts as success

Measured, per interval: log odds at each asynchrony and the asynchrony
threshold by both rules from (a) (Eqn 2 and BASS's). Expected, to be checked
rather than assumed:

- Controls: one source, clearly (log odds well below 0).
- Tritone with different spectra: two sources even when synchronous.
- Octave: one source when synchronous; two only with enough asynchrony.
- The order of thresholds follows the D3 grading.

No comparison with listeners in this milestone. The literature on onset
asynchrony and harmonic grouping would be cited first, if Cho wants that.

### B6. Running BASS on the same stimuli

**Options.** (A) Not in this milestone. (B) Write a BASS hypothesis file
locally (never committed) and run it on Cho's PC, about 1 h per stimulus
for two hypotheses (estimate from the (a) run's 25 and 36 min per hypothesis).

**Recommendation: (A) for now.** Decide after our results, and only for the
stimuli where they are surprising.

## 3. Size (estimate)

A two-source fit has about twice the parameters of (a)'s. (a)'s fits took
about 22 s each on the cloud CPU. With 24 stimuli, 2 hypotheses and 2–3
starting points each, that is about 120 fits, roughly 1.5 h on one CPU, plus about 1 h for
B2's third hypothesis (its whistle fits are larger).
To be measured on the first stimulus before the rest are run.

## 4. Order of work

1. B4: package scene code; re-run (a) through it as the test (PR).
2. B1: the two-note stimuli and controls, with tests (PR).
3. B2/B3: fit and compare one source against two; results and log in the
   step log (PR plus results page).
