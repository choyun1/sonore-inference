# Faster online inference — design

Status: **accepted by Cho, 2026-10-15** (all six recommendations, S1 to
S6). §7 keeps the questions as they were asked. Cho chose on 2026-10-15
to do the speed-ups before the order-effects runs ([order-effects.md](order-effects.md), parked), since
speed compounds across every later experiment.

## Decisions at a glance

All six accepted by Cho on 2026-10-15. The options and reasons are in §2.

| Decision | Accepted |
|---|---|
| S1. Render each event over its own span | Yes: exact, no approximation; first |
| S2. Take the evidence less often than the fit | Every 0.5 s and at each triplet's end; fit still every 50 ms. **Outcome:** passed S6 at 8 and 4 st, six to eight times faster (results/O2); default of `online_buildup.py` since 2026-10-16 |
| S3. Fewer optimizer steps per window | L-BFGS on the free latents, Adam kept as the default until it is shown equal. **Outcome:** tried in two forms, failed S6, dropped (results/O2); (C) and (D) remain |
| S4. Evaluate only the recent sound | The past summarized by a Gaussian in the whole-sound latents; after S1 to S3 are measured |
| S5. Hardware | Measure the PC's GPU; float64 throughout until float32 is shown safe |
| S6. How each is checked | Against the O1 runs: log odds within 3 at every triplet, cost measured with `online_cost.py` |

Numbers are marked as in [online.md](online.md): **[measured]** with its
file, **[code]** with file and line, **[estimate]** with its basis.

---

## 1. What this is

[online-realtime.md](online-realtime.md) measured where an online window's
time goes. At 3.05 s of sound, 8 semitones, a 1 s lag and 50 ms windows,
one window took 69 to 179 s (one stream) and 86 to 121 s (two streams)
[measured: `results/data/O1/st8_w50_p50_lag1.txt`]. Of that, the Laplace
Hessian took 54 s (one stream) and 108 s (two), one Adam step 0.20 to
0.24 s, and the render of the whole sound so far 54 to 62 ms of each
step's 65 to 72 ms forward pass [measured:
`results/data/O1/online_cost_st8.txt`].

This design turns that note's §4 into changes, in the order that keeps each
one checkable on its own. Each keeps the model as it is (priors, likelihood,
hypotheses); S2 to S4 change the approximation, and are accepted only if
they reproduce the O1 results (S6).

## 2. Decisions

### S1. Render each event over its own span

Each event is synthesized over the whole sound so far and then gated to
zero outside it [code: `sources.py` `trajectory_event`, `n_samples` from
`total_duration`; `_gate`]. A source with n events costs n times the whole
sound, and n grows with the sound: render time rose 16 times for 5.5 times
the length [measured: online-realtime.md §3].

**Options.**

- **(A) Keep it.**
- **(B) Render each event only over the samples where its gate is not
  zero**, from its current onset and duration (taken without gradient to
  choose the slice; the gate itself keeps its gradient), and add it into
  the sound at its place.

**Recommendation: (B).** The gate is exactly zero outside the event
[code: `_gate`], so the waveform is the same; the phase is still the
running integral of frequency from the onset, so it differs only by the
order of floating-point sums. Checked by a test: whole-sound and per-span
renders agree to 1e-9 for random scenes, and the gradients too. Expected
gain: render time linear in the sound rather than faster than linear,
about 5 to 10 times less at 3 s [estimate: 12 events of 50 ms against
12 renders of 3.05 s; the interpolation of the trajectories stays whole].
It helps the batch fits of milestones (a) to (c) as well.

### S2. Take the evidence less often than the fit

**Options.**

- **(A) Every window**, as now.
- **(B) Every T seconds and at named times** (each triplet's end), the fit
  still every window. Between evidence times, latents freeze as the lag
  passes, as now; at an evidence time, the term is the Laplace estimate now
  minus the last one restricted to what is still free, the same formula
  with a longer step [code: `FixedLagEvidence.add`].
- **(C) Only at the end**, which is the batch estimate.

**Recommendation: (B), T = 0.5 s.** With no lag the terms still sum
exactly to the batch Laplace estimate, whatever T (the telescoping of O5
does not depend on the step). With a lag, latents that froze between two
evidence times were fitted after the last Hessian, so the restriction is
taken at slightly different values; how much that matters is what S6
measures. The decision needs the evidence at the leaky integrator's scale
(τ 1 to 5 s), not every 50 ms. Expected gain: ten times fewer Hessians,
which are most of the time [estimate: count of Hessians].

### S3. Fewer optimizer steps per window

One stream took up to 1,556 Adam steps in a window, two streams at least
50, the stopping rule's floor [measured: O1 page, "Lag and window"].

**Options.**

- **(A) Adam**, as now.
- **(B) L-BFGS** (torch's, with a line search) on the free latents, which
  uses curvature from the gradients it has seen and needs no Hessian.
- **(C) Newton steps with the Hessian** of S2's evidence times.
- **(D) A relative stopping rule**, so two streams can stop below 50 steps.

**Recommendation: (B), with (D) for whichever wins.** L-BFGS usually needs
far fewer steps than Adam on smooth problems like these fits, and costs
about one gradient per step plus a few in the line search [estimate; to be
measured]. (C) would need a Hessian in every window, which S2 removes.
Adam stays the default until L-BFGS reproduces the O1 fits (S6), because
the learning rates per latent were tuned for Adam in milestones (a) to (c).

**Outcome (2026-10-15).** Tried twice on the O1 case and dropped
([O2](../../results/O2-online-speed.md)): plain L-BFGS drifts along
directions the sound so far does not constrain and the next window never
recovers (odds −1.4 million against 132); held near the window's start and
finished by Adam, it lands the one-stream fit in a worse basin (odds 167
against 132) and saves 6% of the time. (D), a stopping rule that can stop
below 50 steps, is the option left here.

### S4. Evaluate only the recent sound

Even with S1, every step renders and compares the whole sound so far,
because the means of each trajectory (and the kernel's latents, when
inferred) stay free over the whole sound (online.md O3).

**Options.**

- **(A) Keep it.**
- **(B) Freeze the whole-sound latents too** after the first L seconds.
- **(C) Summarize the frozen past by a Gaussian in the whole-sound latents**:
  the part of the log joint from frames before the lag, as a quadratic in
  the means, from the Hessian of the last evidence time (assumed-density
  filtering). Each step then renders and compares only the last lag plus
  the window.

**Recommendation: (C), but only after S1 to S3 are measured.** (B) would
stop the means from learning from new tones, which matters for the
one-stream whistle that keeps meeting A–B jumps. (C) keeps them free at a
bounded cost per step. It is the largest change in the approximation, and
its own check against the full sum is needed, as O5 did for the window
terms. On 3 s stimuli S1 to S3 may be enough; S4 is what longer stimuli
(order effects with 8 s gaps, and anything towards real time) need.

### S5. Hardware

Cho's PC has an RTX 2070 SUPER; the cochleagram's forward and backward on
3.1 s took 0.011 s in float32 and 0.037 s in float64 there
[measured 2026-10-05: `/mnt/project-files/results/machines/`]. The cloud CPU is what every O1 number used.

**Options.**

- **(A) Cloud CPU only.**
- **(B) Measure the PC's GPU** with `online_cost.py`, in float64.
- **(C) float32** for the fit, float64 for the Hessian and its log
  determinant.

**Recommendation: (B) now, (C) only if measured safe.** The Laplace log
determinant over hundreds of latents is where float32 could change the
answer; the fit is less sensitive. The 2070 runs float64 slowly, so (C)
may matter on that card [estimate: consumer cards' float64 rate].

### S6. How each change is checked

**Recommendation.** Each change is run on the O1 reference case (8 st,
6 triplets, lag 1 s, 50 ms windows, stopping rule O9) and accepted when:

- the online log odds at every triplet are within 3 of the O1 run's
  (10.0 to 131.9 [measured: O1 page]), and the leaky odds (τ 2.5 s) within
  3 as well; 3 is below the differences already seen between window
  lengths (up to 7) and the Laplace estimate's own sensitivity (26 at 8 st,
  6 triplets);
- for S1 only, equality to rounding instead, by a test;
- the cost per window is measured with `tools/online/online_cost.py` (with
  a new column for the change) and the whole run's time reported.

Then 4 st as well before the change becomes the default.

## 3. Order of work

1. S1, with its test: one PR. Every batch tool gets it too, so their
   outputs change by rounding only; the PR says so (no silent output
   changes).
2. S2 and S3 behind options of `run_online` and `online_buildup.py`, each
   run against O1: one PR with a results page (experiment O2) of cost and
   agreement.
3. S5's measurement on the PC, if Cho can run it.
4. S4, its own design detail and check, once 1 to 3 are measured.

## 4. What is not changed

The priors, the likelihood, the hypotheses, the lag rule of O3, the
squared-exponential kernel (online.md O4 (A)). A state-space kernel
(O4 (C)) is a model change and stays a later decision.

## 5. Size

S1 is a change in `sources.py` and a test. S2 and S3 are options in
`online.py` and the tool. Each check is one O1-sized run of both
hypotheses, about 1.5 h of cloud CPU today and less after S1 [measured:
3026 + 2572 s for the O1 run].

## 6. Risks

- S2: the frozen-between-evidence-times effect may be larger than 3 at
  8 st, where freezing already cost up to 22.5 [measured: O1 page,
  "Freezing costs little here"]. Then T shrinks, or S4 comes first.
- S3: L-BFGS can stall on the one-stream fit's A–B jumps, where Adam
  needed its many steps. Then (C) or (D) instead.

## 7. Questions for Cho, short form

1. **S1** Render each event over its own span, exact to rounding?
   (recommended: yes)
2. **S2** Evidence every 0.5 s and at each triplet's end, fit every 50 ms?
   (recommended: yes)
3. **S3** Try L-BFGS against Adam, Adam stays default until shown equal?
   (recommended: yes)
4. **S4** A Gaussian summary of the past, after 1 to 3 are measured?
   (recommended: yes, later)
5. **S5** Measure on your PC's GPU in float64; float32 only if safe?
   (recommended: yes)
6. **S6** Accept a change when log odds stay within 3 of O1 at every
   triplet? (recommended: yes)
