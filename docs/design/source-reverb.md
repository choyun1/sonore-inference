# Source and room inference: design document

Status: **draft for Cho**, written 2026-10-10. AI-assisted (Claude), from
HANDOFF-source-reverb.md, the 2017-18 archive, McDermott, Wrobleski & Oxenham
(2011) and the evaluation of the same date. No library code until Cho accepts
this.

**How numbers are marked.**

- **[probe N]**: measured by one of the four scratch probe scripts behind the
  evaluation (project files, `source-reverb/probes/`). These are single-seed
  numpy scripts and are not library code. Each one will be redone as a
  `tools/` script before anything rests on it.
- **[MWO]**: McDermott, Wrobleski & Oxenham (2011), *PNAS* 108, 1188-1193,
  Methods and SI Methods.
- **[handoff]**: HANDOFF-source-reverb.md, measured there on 2026-10-10.
- **[code]**: read in the archive's `sri/sigtools.py` or sonore 0.5.0.
- **[estimate]**: my estimate, with its basis next to it.

---

## 1. Problem and scope

One source plays in one room, and the observer hears only the result. The
model infers the source's spectrotemporal statistics and the room's RT60 from
that one sound. The work has two experiments. (a) Recover the parameters
from sounds the model could have made. (b) Infer with an exponential-room
model from rooms that are not exponential, and measure how RT60 is biased.
A comparison with Traer & McDermott's (2016) listeners is optional, and if
done it is labelled a comparison, not a validation.

Out of scope: several sources, binaural cues, real recordings as observations
(except as an optional later test), and amortized inference.

## 2. Already settled with Cho (2026-10-10)

| # | Question | Answer |
|---|---|---|
| S1 | Direct sound | Included at a typical DRR. Tail only is a later control. |
| S2 | Source timing | 400 ms source ending inside the recording, onset and offset known to the model. A sound with no offset is a later experiment. |
| S3 | Source generator | Write the corrected generator here, with a test. Add a parity test against sonore's version when it lands, then switch. |
| S4 | RT60 prior data | Traer & McDermott's published 271-IR survey, which can be checked, not the archive's 469 values of unknown origin. |
| S5 | Source variance | The 2017 value of 0.5 had no source. Make it a prior range. |

## 3. What the 2018 failure becomes

The handoff's lesson was to freeze or average out every random draw, render
the observations with the model, and check the likelihood surface before
sampling. Here is how each draw is handled:

| Random draw | 2018 | This design |
|---|---|---|
| Source carrier noise | frozen file | averaged out: the model predicts expected band power (R3) |
| Source log-power grid | frozen standard-normal draw | **inferred**: latent with a Gaussian-process prior (R1) |
| IR noise | redrawn on every call (the main failure [handoff]) | averaged out: expected band energy envelope (R3) |
| Observation | made with unrecorded settings | rendered as waveforms by the generator and `sonore.synth_ir`, with every setting saved |

The observations are full waveform renders, and the inference model predicts
expected power. So experiment (a) includes the real gap between the two
(3.4 dB sd residual [probe 3]) instead of hiding it.

## 4. Decisions for Cho

### R1. Source model

Cho asked whether we need this stimulus model at all. The source model is
both the stimulus generator and the observer's prior, the thing that stops
the inferred source from absorbing the reverb [probe 4]. Experiment (a)
needs sounds drawn from that prior.

**Options.**
- **(A) MWO-style spectrotemporal source.** The log power on an ERB-band by
  time-window grid is a Gaussian process with separable exponential
  correlations, imposed on white noise. For: the source has statistics worth
  inferring, the prior is Gaussian in log power so evidence is computable,
  and it is the 2017 design. Against: the sounds are synthetic.
- **(B) Fixed-spectrum noise burst.** For: simplest. Against: no source
  statistics are left to infer, so it answers only the room half of the
  question.
- **(C) Natural sounds (speech, impacts).** For: realistic and close to
  Traer & McDermott. Against: no prior fits them exactly, so they cannot be
  the in-model check.

**Recommendation: (A) now, and (C) later as an out-of-model test.** The
specification, with corrected construction (cells ordered as the covariance
assumes, each window divided by its noise RMS):

- Bands: 30 half-cosine ERB filters from 50 Hz to 8 kHz, about 1 ERB apart
  [estimate: MWO used 39 over 20 Hz to 4 kHz (about 0.66 ERB apart); 8 kHz
  covers the room's frequency profile].
- Windows: 20 ms raised cosines at 50% overlap, so the grid step is 10 ms
  [MWO].
- Correlation: exp(−a·|Δ ERB-number|) · exp(−b·|Δ grid step|), with
  defaults a = 0.11 per ERB and b = 0.065 per step. 0.11 is MWO's 0.075 per
  filter converted at their spacing of about 0.66 ERB. My probes used 0.075
  per roughly 1 ERB band, a smoother source than MWO's.
- Mean: flat spectrum, which MWO achieves by setting the mean proportional
  to bandwidth [MWO]. The overall level is normalised out.
- Hyperparameters inferred: a, b, and the log-power sd s. a and b are
  log-uniform over a factor of 10 around their defaults, as in 2017. s is
  log-uniform from 2 to 20 dB [estimate: brackets the probe's assumed 7 dB;
  MWO does not state theirs].
- Check: realised against intended grid correlation. The probe version gives
  0.935 [probe 3], against 0.72 for the 2017 code [handoff].

### R2. Room model

**Options.** (A) `sonore.synth_ir` with the direct impulse and a DRR. (B) Its
own room model in core.

**Recommendation: (A) for observations, and a core reimplementation of only
its expected band energy for the model.** Per design-v1 D2, core never imports
sonore, so R3 implements the Traer & McDermott band RT60 and onset-level
regressions in torch and tests them against `sonore.band_rt60s` and averaged
`synth_ir` renders.

- Latent: broadband (median) RT60.
- DRR: fixed and known in (a), at a typical value. Inferring it is a later
  variant. **The typical value comes from the 271-IR survey** (S1, S4),
  measured at 1.5 m source distance as reported by the survey page (search
  result, not yet read in full).
- Frequency profile: ecological in the model. The atypical profiles appear
  only in observations, for (b).

### R3. Forward model for inference

**Options.**
- **(A) Expected band power.** In band b, E[y_b(t)²] = (s_b² ∗ e_b)(t),
  where s_b is the source's band power and e_b is the room's expected band
  energy, including the direct impulse.
- **(B) Waveform render with frozen noise**, as in BASS and 2018.
- **(C) Waveform render with the noise as latents.**

**Recommendation: (A).** It averages out both noises at about the cost of
frozen noise, and it is smooth in RT60. Measured against waveform renders, it
leaves −1.1 dB mean and 3.4 dB sd, and each IR draw's RT60 peak falls at 0.38
to 0.40 s for a true 0.40 s [probe 3]. The −1.1 dB is the expected offset of
the log of a chi-square variable. It will be corrected analytically, and the
correction checked against renders. One thing is not yet measured: probe 3
used the source's actual waveform power, so the residual from the carrier
noise inside each window still has to be measured. It is the first test in
§5.

### R4. Representation and likelihood

**Options.** (A) Block band power in dB: the same cosine ERB bands, 10 ms
blocks, linear in power before the log, so R3 applies exactly. (B) The
existing `Cochleagram` or `FFTCochleagram`. These are envelope magnitudes
with their own smoothing, so R3 would only approximate them.

**Recommendation: (A), as a new class, tested against sonore's `Filterbank`.**
The Gaussian noise in dB has sigma set from the measured model-to-render
residual (3.4 dB [probe 3]) rather than a round number. The floor is a
parameter, applied to both observation and prediction, as BASS does. Two
things follow from this:

- The residuals are correlated across cells, so evidence gaps are
  overconfident [evaluation §2]. Results report posteriors at two sigmas
  (the measured sigma and twice it), so readers can see how much depends on
  that choice.
- (b) depends on the floor: a linear decay matched at the end reads 2.3 to
  4.3 times the true RT60 as the floor goes from 60 to 10 dB [probe 2]. So
  (b) is reported at floors of 20, 40 and 60 dB. 20 dB is BASS's floor.

### R5. Inference

**Options.** (A) A grid over the hyperparameters and RT60, with the source
grid integrated out by Laplace at each point. (B) Joint MAP over everything.
(C) MCMC, SMC or SVI over everything.

**Recommendation: (A).** Joint MAP is ruled out: without an offset it prefers
a dry room by 102 nats while the evidence does not [probe 4, case B]. With an
offset, the evidence is sharp in RT60 (130 nats lower at 0.35 s and 81 lower
at 0.45 s) whatever source tcorr is used [probe 4, case A].

The grid is RT60 (15 points, log-spaced from 0.1 to 2 s) times a, b and s
(5 points each), 1,875 points. Each point needs one L-BFGS fit of 1,200 grid
latents and a log-determinant. 27 fits took 45 s on a 4-core container
[probe 4], so the full grid takes about 1 h [estimate]. The existing
`evidence.log_evidence` (PR #8) is reused unchanged, and its importance
sampling checks each Laplace estimate (ESS reported). A sampler is added only
if the grid shows multimodality. Before any fit, likelihood slices through
the truth are plotted, as the handoff asks.

### R6. Framework and placement

PyTorch, per design-v1 D1. The numpy probes are ported, not reused. New
modules go under `src/sonore_inference/`: `spectrotemporal.py` (R1 source and
prior), `room.py` (R2/R3 expected energy) and `blockpower.py` (R4). Tools go
in `tools/`. The milestone (b) modules (`scene.py`, `cochleagram.py`,
`evidence.py`) are reused without edits, so the two lines of work do not
collide.

### R7. Experiments

- **(a) Recovery.** 10 seeds × RT60 in {0.2, 0.4, 0.8} s × three settings of
  (a, b, s): the defaults and two corners. Success: the truth falls in the
  90% posterior interval in about 90% of runs, for each parameter. The width
  of each interval is reported; wide intervals for a and b are an expected
  finding [probe 4: a factor of 2 in b changes evidence by 4-5 nats].
- **(b) Atypical rooms.** Exponential, ecological model against observations
  with each atypical decay shape (time_reversed, linear_matched_start,
  linear_matched_end) and profile (inverted, exaggerated, reduced). Reported:
  the posterior median RT60 divided by the true RT60, at three floors, next
  to `measure_rt60`'s Schroeder T20×3 reading of the same IR as a reference
  [probe 2: 1.79 for linear_matched_end]. For time_reversed, an exponential
  model's RT60 has no clear meaning, so it is reported as a model-fit
  failure (evidence relative to exponential rooms), not as a bias.
- **(c) Optional.** Comparison with Traer & McDermott's listeners.

### R8. Priors and the survey data

**RT60:** lognormal, fitted to the RT60s that core's own `measure_rt60`
equivalent measures from the 271 survey IRs (S4). A lognormal beat an
exponential by 258 AIC on the archive's 469 values [probe 1]; the fit is
redone on the survey. **The survey files are not redistributed.** A `tools/`
script downloads them, and only the two fitted numbers go in the code, with
a citation.

Two things are open:
- The licence of the original files is unchecked. A mirror says CC BY 4.0,
  but that was not read on the MIT page.
- The MIT site is blocked from this container by its network policy, so the
  download has to happen on Cho's machine (§6).

## 5. Order of work after acceptance

1. Generator (R1) and a test of its realised correlation.
2. Expected-energy model (R2/R3) with tests against `band_rt60s` and
   averaged renders; measure the carrier-noise residual and the log-bias
   correction.
3. Block-power likelihood (R4); likelihood slices at the truth.
4. Survey download and RT60 prior fit (R8).
5. Laplace grid (R5); experiment (a).
6. Experiment (b).

Each step is one PR and one row in the step record.

## 6. What Cho needs to do

1. Accept, or change, R1 to R8.
2. When step 4 arrives, run the survey download on your machine. This
   container cannot reach mcdermottlab.mit.edu.

## References

- McDermott, J. H., Wrobleski, D., & Oxenham, A. J. (2011). Recovering
  sound sources from embedded repetition. *PNAS* 108(3), 1188-1193.
- Traer, J., & McDermott, J. H. (2016). Statistics of natural reverberation
  enable perceptual separation of sound and space. *PNAS* 113(48),
  E7856-E7865. doi:10.1073/pnas.1612524113. Survey:
  https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
- Evaluation, 2026-10-10 (project files: `source-reverb/evaluation.md`).
