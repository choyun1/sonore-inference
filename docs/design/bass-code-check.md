# BASS code check against design-v1.md

Checked 2026-10-04 against https://github.com/mcusi/bass at commit `48e3755`
("Update README.md"), cloned from this environment (github.com was reachable).
The repository has no LICENSE file, so only short excerpts are quoted below and
nothing was copied into the project. Line numbers are at that commit.

Scope: the two items design-v1.md §1.1 marks "check in code" (items 1 and 7),
plus item 8, which compares the paper with the earlier code reading.

## Item 1. Number of sources: is the Poisson rate scaled by scene duration?

**Partly. Sampling scales it; scoring does not.**

- Every shipped config sets `n_sources: {dist: poisson, args: 1.0}`
  (e.g. `config/full_sequential.yaml:78-80`; the same in all 11 configs).
- Sampling from the prior, `model/scene.py:164-173` (`sample_discrete`), draws
  from `Poisson(scene_duration * n_sources_lambda)` and redraws while the
  result is 0. So generation uses a rate of 1 source per second and is
  zero-truncated.
- Scoring under the prior, `model/scene.py:181-189` (`log_p_discrete`), uses
  `Poisson(n_sources_lambda).log_prob(n_sources)`: rate 1 per scene, no
  duration factor and no zero-truncation correction. This is the term that
  enters `log_p` (`model/scene.py:195`) and therefore the ELBO used in inference.
- `Scene.sample` (the sampling path) is called from
  `inference/amortized/dataset.py:131`, i.e. when generating training scenes
  for the event-proposal network. Inference on observed sounds goes through
  `log_p`.

Implication for design: what the paper calls "rate 1 per second" holds for the
training-data generator; the prior that scored hypotheses during inference is
Poisson(1) over the whole scene, whatever its length. For the illusion stimuli
(roughly 0.5 to 2 s) the two differ modestly; for longer everyday sounds they
differ more. sonore-inference should pick one deliberately and state it.

## Item 7. Variational objective: standard ELBO or importance-weighted?

**The code implements an importance-weighted bound, but every shipped config
runs it with K = 1, which is the standard ELBO. The paper's description matches
what ran.**

- The loss is `importance_weighted_bound` (`inference/metrics.py:17-29`),
  docstring citing eq. 8 of Burda et al. (arXiv 1509.00519) and paper eq. B.4.
  It reshapes the batch of scores to `[n_importance_samples, batch_size /
  n_importance_samples]`, takes `logsumexp` over the first axis, subtracts
  `log K`, and averages.
- It is selected when `loss_type == "elbo"` (`inference/optimize.py:37-39`).
  Other options there are `map` (`:40-44`) and `ml` (`:45-49`).
- All 11 configs set `loss_type: elbo`, `n_importance_samples: 1`,
  `batch_size: 10` (e.g. `config/full_sequential.yaml:339-341`). With K = 1 the
  `logsumexp` over a length-1 axis is the identity, so the loss is the mean of
  `-(ll - lq + lp)` over 10 samples: the standard ELBO with 10 samples per
  step, as the paper says. The per-sample score is defined at
  `model/scene.py:342` (`score = self.ll - self.lq + self.lp`).
- The hypothesis-comparison number is `elbo()` in `inference/metrics.py:43-44`:
  `logsumexp(scores) - log(len(scores))` over the logged samples, i.e. an
  importance-sampling estimate of log marginal likelihood, despite the name.
  `basic_loop_from_scene` returns it as `importance_sampled_elbo`
  (`inference/optimize.py:136-137`). This matches the paper's "importance
  sampling afterwards" [App. B.1.1, eqs B.3-B.5].
- No code path other than the configs sets `n_importance_samples` (grep over
  the repo).

Implication for design: the handoff's "importance-weighted ELBO" describes the
code's option; the runs used the plain ELBO. sonore-inference can start with
the plain ELBO and keep K as a parameter.

## Item 8. Harmonic excitation (paper vs earlier code reading)

**The code reading is right on the count; the high-frequency taper is wider
than "near Nyquist".**

- Default `n_harmonics=200` (`renderer/excitation.py:16`, passed from
  `model/source.py:242` as `renderer_options.get("n_harmonics", 200)`; no
  config overrides it), but the indices are `torch.arange(1, n_harmonics)`
  (`renderer/excitation.py:36`), i.e. harmonics 1 to 199.
- Limits (`renderer/excitation.py:20-23`): `hi_lim_freq = audio_sr/2 - 1`,
  `lo_lim_freq = 30`, `hi_ramp_hz = 1000`, `lo_ramp_hz = 25`.
- Per-harmonic weight (`renderer/excitation.py:135-139`): a linear taper from
  1 to 0 over the 1000 Hz below `hi_lim_freq`, times a linear taper from 0 to 1
  over 30 to 55 Hz. At the configs' `audio_sr: 20000`
  (`config/full_sequential.yaml:18`) that means full weight up to about
  9 kHz, falling to zero at 9999 Hz. Frequencies are also clamped to
  [30, 9999] Hz inside `components` (`renderer/excitation.py:99-100`).
- So "samples above Nyquist set to zero" [paper App. A.5] is the end point of a
  1 kHz linear roll-off, not a hard cut, and the count is 199, not 200.

Implication for design: minor. If sonore-inference reproduces BASS renders as a
test oracle, match 199 harmonics and the two linear tapers; otherwise any
smooth band-limit is fine.
