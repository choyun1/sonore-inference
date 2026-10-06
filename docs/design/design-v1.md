# sonore-inference: design document, version 1

Status: **accepted by Cho on 2026-10-05** (D1 to D8 as recommended; see §5 for
the answers to D3 and D6).
Written 2026-10-04. AI-assisted (Claude), from HANDOFF.md, bass-vs-sonore.md and
a full read of the paper.

**How numbers are marked.** Nothing in this document was measured by a
`tools/` script yet, because there is no code. Every number is one of:

- **[paper §x]**: quoted from Cusimano, Hewitt & McDermott (2024), *Listening
  with generative models*, Cognition 253, 105874, at the section given;
- **[code]**: from the earlier reading of github.com/mcusi/bass at 48e3755,
  as reported in HANDOFF.md, or from bass-code-check.md where cited (§1.3);
- **[estimate]**: my own estimate, with its basis stated next to it.

---

## 1. What reading the paper changes in the handoff

The handoff's code reading holds up. The paper corrects or sharpens it in the
places below. Where paper and code may disagree, the code is what ran, so those
items are marked "check in code".

### 1.1 Corrections

1. **Number of sources.** The paper gives Poisson with *rate 1 source per
   second* [paper Table A.1]. The code is not consistent with itself
   [bass-code-check.md, item 1]: *sampling* (used to generate training scenes
   for the proposal network) draws from Poisson(duration × 1) and redraws on
   zero (`model/scene.py:164-173`), while *scoring* (the prior term in the
   ELBO, so what inference used) is plain Poisson(1) per scene, with no
   duration factor and no zero-truncation correction
   (`model/scene.py:181-189`). So the handoff's "Poisson(1)" describes
   inference and the paper's "rate 1 per second" describes data generation.
   The authors report that the parameters of this prior had little effect on
   their results [App. A.4]. The number of events per source is
   Geometric(p = 0.5) and the source type is uniform over noise, harmonic and
   whistle [Table A.1]. See D8.
2. **Event ramps.** Sigmoid on and off ramps, 18 ms in total, with the sampled
   onset and offset at the ramp maxima, chosen for differentiability
   [paper App. A.5]. bass-vs-sonore.md said "10 ms-scale ramps".
3. **Trajectory time step.** 10 ms for the illusions, but **20 ms for everyday
   sounds**, to save memory [App. A.5]. The handoff gives only 10 ms. By the
   handoff's own reasoning the amplitude modulation ceiling for everyday sounds
   is about 25 Hz rather than 50 Hz [estimate: half the trajectory sampling
   rate, 1 / (2 × 20 ms)].
4. **Fast modulation is limited by the likelihood too, not only the renderer.**
   For co-modulation masking release the authors lowered the masker's
   modulation cutoff from the original 50 Hz to 10 Hz "because the relatively
   coarse temporal resolution of the cochleagram representation limited the
   rate of modulations that could be resolved" [App. C.4]. So adding fast
   modulation to a renderer does nothing unless the likelihood can see it
   (see D4).
5. **Why the noise is frozen.** The paper says why: the pink noise sample "was
   frozen with a fixed random seed to enable differentiability", and the pink
   spectrum is "an arbitrary choice" given the degeneracy of excitation and
   filter [App. A.5]. This replaces the handoff's inferred "variance reduction"
   reason. The open question for Maddie narrows to why one draw is *shared by
   all* noise sources rather than one frozen draw per source.
6. **Likelihood details.** The likelihood uses an FFT-based gammatone-like
   spectrogram (Ellis 2009), chosen for speed: 64 channels with half-ERB
   bandwidths and centre frequencies from 20 to 9423 Hz, 25 ms window, 10 ms
   hop, truncated at 20 dB relative to a fixed reference RMS of 1e-6, compared
   under isotropic Gaussian noise with σ = 10, which was set by hand
   [paper §2.2.3, App. A.5]. The chcochleagram model (Feather et al. 2023;
   40 half-cosine ERB filters, 42 to 7327 Hz, Hilbert envelopes at 200 Hz) is
   the **input to the event-proposal network**, not the likelihood
   [App. B.2.1]. The handoff's "gammatonegram or cochleagram" for the
   likelihood may describe a code option; the paper uses the gammatonegram.
7. **Variational objective.** The paper optimizes the standard ELBO with Adam,
   10 samples from the guide per step, and estimates each hypothesis's marginal
   probability afterwards by importance sampling [App. B.1.1, eqs B.3–B.5]. The
   handoff says "importance-weighted ELBO". Both are right
   [bass-code-check.md, item 7]: the loss is an importance-weighted bound
   (`inference/metrics.py:17-29`), but every shipped config sets the number of
   importance samples to 1 with batch size 10, which is the plain ELBO over
   10 samples, as the paper says. The number used to compare hypotheses is an
   importance-sampling estimate of the log marginal likelihood, despite being
   named `elbo` (`inference/metrics.py:43-44`). GP trajectories use
   a variational inducing-point guide (Hensman et al. 2015); everything else is
   mean-field [App. B.1.1].
8. **Harmonic excitation.** The paper says 200 harmonics with samples above
   Nyquist set to zero [App. A.5]. The code renders harmonics 1 to 199
   (`torch.arange(1, 200)`), with a linear taper from 0 to 1 over 30 to 55 Hz
   and a linear taper from 1 to 0 over the 1000 Hz below 9999 Hz at 20 kHz
   sampling, so full weight ends near 9 kHz [bass-code-check.md, item 8;
   `renderer/excitation.py:20-23, 36, 135-139`]. This matters only if a test
   reproduces BASS renders exactly.

### 1.2 Additions that matter for design

- **Speech is the gap, and it is time-varying spectra, not only fixed
  formants.** The authors left speech out of fitting the harmonic spectrum
  prior "because the time-varying formants of speech could not be well-modeled
  with the model's constant spectrum constraint" [App. A.3], and excluded the
  Speech category from the everyday-sound test set for the same reason
  [App. F.1]. A voice source therefore needs a filter that is fixed in
  frequency (not shifted with f0) *and* that can change over time. The
  handoff stressed only the first.
- **The authors already name most of the proposed extensions.** Reverberation
  and spatial effects [§4.3.3], composite sources mixing noise and harmonic
  parts (breathy flute, fluttertongue) [§3.3.3.1, §4.1.3], time-varying
  spectra and frequency-dependent decay [§3.3.3.2, §4.2.1], textures with a
  statistics-based likelihood [§4.3.2], and a periodicity representation in
  the likelihood [§4.2.3]. The handoff's "opening" is real, but it is the
  authors' own future-work list. Any claim of novelty should be modest and
  should wait for the conversation with Maddie and Josh.
- **Prior art the handoff missed (both cited by the paper).** Nix & Hohmann
  (2007) jointly estimated spectral envelopes *and source direction* of
  concurrent voices by sequential Monte Carlo (IEEE TASLP 15(3), 995–1008).
  That is a direct precedent for voices plus space. Turner (2010, UCL PhD
  thesis) used Gaussian processes on raw audio for scene analysis with point
  estimates [§4.1.1]; the handoff listed Turner & Sahani only as single-signal
  demodulation. I have not read either; both belong on the reading list.
- **Structure matters more than fitted priors.** Replacing the fitted source
  priors with uniform ones did not measurably worsen the match to humans
  (p = 0.30), while fixing source parameters did (p < 0.01) [§3.2.1.1–2]. So
  fitting priors to recorded sounds can come late.
- **Enumerative inference did most of the psychophysics.** All but one of the
  psychophysical experiments compared a few experimenter-defined hypotheses
  with SVI, without the proposal network [§2.4]. This is the cheapest route to
  meaningful results (D5).
- **The stimuli are fully specified in the paper.** Appendix C gives levels,
  durations, ramps, frequencies and filter specs for every illusion. Milestone
  (a) can be built from the paper text, which is CC BY 4.0, without touching
  the unlicensed code (D7). The Darwin & Sutherland vowels were made with a
  Python Klatt synthesizer (Sprouse's klsyn) [App. C.7], which sonore's
  `klatt_synthesize` could reproduce.
- **Reported compute.** Sequential inference tested "hundreds" of hypotheses
  per sound, each optimized on one GPU for 2 minutes to an hour: "tens or
  hundreds of GPU hours" per sound [App. B.2.5]. Enumerative inference used
  1–10 hypotheses per sound at 30 minutes to 2 hours each, 8000 iterations
  each [App. B.3.1–2]. The GPU model is not stated.

### 1.3 What I could not check

- Items 1, 7 and 8 above were settled against the code at 48e3755 by a
  separate session that could reach GitHub; its notes, with file and line
  references, are in [bass-code-check.md](bass-code-check.md). The global
  GitHub name search (§2) was still not run.
- I have not checked chcochleagram's licence.

---

## 2. Name check (sonore-inference)

| Where | Result | How |
|---|---|---|
| PyPI | **Free.** `sonore-inference`, `sonore_inference` and `sonoreinference` all return 404 (PyPI normalizes these to one name anyway). `sonore` 0.4.0 is Cho's. | `curl https://pypi.org/pypi/<name>/json`, 2026-10-04 |
| GitHub, Cho's account | **Free.** The only repository matching "sonore" that Cho's GitHub connection can see is `choyun1/sonore`. | Claude Code repository listing, 2026-10-04 |
| GitHub, everyone else | **Not checked.** github.com returns 403 from this environment. | — |
| Web | No hits for the exact string "sonore-inference". | web search, 2026-10-04 |

The import name would be `sonore_inference`. A repository-wide check on
GitHub is a 10-second search Cho can do in a browser.

---

## 3. Scope, in one paragraph

A PyTorch library (if D1 goes that way) of **differentiable renderers** for
source types, each verified against sonore's NumPy output, plus a
**likelihood** on an auditory representation, plus enough **inference** to
compare a few hypotheses about a sound. It keeps BASS's structure (sources
emit events; smooth trajectories; a time-frequency likelihood) and adds source
types BASS lacks. It does not try to reproduce BASS's full sequential
inference or its proposal network until the smaller pieces work.

---

## 4. Decisions for Cho

Each decision lists the options at their best, a recommendation and the
reason. D1 to D7 come from the handoff; D8 was added after the code check.
Cho's answer is needed on D1 to D7 before code; D8 only matters at
milestone (c) and can wait. The short form of each question is in §5.

### D1. Framework

**Options.**

- **(A) PyTorch**, plain autograd first, Pyro or gpytorch added only when a
  milestone needs them. For: BASS is PyTorch plus gpytorch, so its numbers and
  any shared code later are directly comparable; chcochleagram from the same
  lab is PyTorch; eager execution handles hypotheses whose structure (number
  of sources and events) changes, which is what a scene grammar needs; Colab
  supports it with no setup. Against: batching many hypotheses needs manual
  padding; less elegant functional transforms.
- **(B) JAX with NumPyro.** For: `vmap` would batch many hypotheses of the
  same structure on one GPU, which is exactly the enumerative-inference
  workload, whereas BASS optimized one hypothesis per GPU; float64 is one flag;
  NumPyro's HMC/NUTS is strong if sampling is chosen in D5. The sonore trial
  found JAX no faster, but that trial measured sonore's own NumPy-style CPU code,
  not batched gradient loops on a GPU, so it says little about this workload.
  Against: each new hypothesis structure triggers recompilation under `jit`;
  no BASS comparability; float32 broke bit-for-bit output in the trial.

**Recommendation: (A) PyTorch.** Comparison with BASS and conversations with
its authors are the project's first uses, and variable structure is central.
Tests against sonore run in float64; whether training runs in float32 is
measured later, not assumed. If D5 later lands on HMC for fixed structures,
revisit.

### D2. Relation to sonore

**Options.**

- **(A) Test oracle only.** sonore is a test dependency, pinned to a commit.
  Each renderer and the likelihood representation must reproduce sonore's
  output to a stated tolerance in float64.
- **(B) Oracle plus optional runtime use** for data preparation (making
  stimuli, HRIRs, rooms) and plotting, behind an optional install extra;
  the differentiable core never imports sonore.
- **(C) Runtime dependency of the core.** Rejected: sonore is 0.x with a
  changing API, and NumPy calls inside a torch graph break gradients.

**Recommendation: (B).** It is the handoff's suggestion made explicit. One
detail the tests must handle: renderers with random excitation cannot match
sonore draw for draw, so tests inject the same excitation array into both
implementations and compare the deterministic part exactly, and compare
stochastic outputs only by statistics stated in the test.

### D3. First milestone

**Options** (from the handoff, sharpened):

- **(a) Stimuli.** Regenerate three of the paper's stimuli from the
  specifications in Appendix C using sonore, in a notebook: mistuned harmonic
  [C.6], ABA bistability [C.10] and the Darwin & Sutherland vowels [C.7] (a
  harmonic, a tone sequence and a Klatt vowel). No inference. Comparing with
  BASS's generators means running their code, which needs github.com access
  from wherever it runs.
- **(b) One richer source, recovered by gradient descent.** A harmonic source
  whose filter is **fixed in frequency** (the opposite of Panpipes), first with
  static formants, then with formants that move over time; recover f0 track and
  filter from a synthetic sound by minimizing the likelihood of D4, no priors.
  This tests the gap the paper itself names for speech (§1.2), and checks the
  code comment that the fixed filter is "less numerically stable" [code].
- **(b′) The spatial alternative.** A spatialized harmonic source. Harder than
  it looks: a magnitude cochleagram per ear can carry level differences but
  discards the fine timing that carries ITD, so space also needs a new
  likelihood (D4).
- **(b″) The instrument alternative** (added after Cho's question; see §7). A
  harmonic source whose spectrum is the product of two parts: an excitation
  spectrum indexed by **harmonic number** (Panpipes' assumption) and a body
  filter **fixed in Hz** (the fixed-filter assumption). BASS's Panpipes and
  option (b) are both special cases. Step 1 recovers both parts, and the f0
  track, from synthetic notes. Step 2 is two such sources in the same f0 range,
  the violin-and-clarinet case BASS merged into one source [Fig. G.4], and
  asks whether the two-part spectrum keeps them apart. It uses the same renderer
  work as (b), framed around instruments instead of voices.
- **(b‴) Two simultaneous notes** (added after Cho's question). BASS never
  tested two harmonic sources at unrelated pitches sounding together; its
  closest case is the FM illusion, an octave apart [§3.1.2.1], and it failed on
  overlapping piano notes [Fig. G.7C]. No grammar change is needed: compare a
  one-source and a two-source hypothesis by enumerative inference, as BASS did
  for the mistuned harmonic [App. C.6]. Graded synthetic stimuli, easiest
  first: (1) an unrelated interval (for example a tritone) with different
  spectra; (2) the same interval with the same spectrum; (3) a just fifth (3:2),
  where every second harmonic of the upper note coincides with every third
  harmonic of the lower one; (4) an octave, where every harmonic of the upper note coincides
  with one of the lower note and only cues like onset asynchrony or vibrato can
  separate them. Expected obstacle: two interleaved harmonic series leave fewer
  components resolved in a half-ERB cochleagram, so steps (3) and (4) are where
  D4's periodicity question is decided.
- **(c) Scene grammar with priors and inference.**

**Recommendation: (a) then (b) or (b″), with (b′) and (c) after both work.**
(b) and (b″) need the same renderer; Cho chooses whether the first target is a
voice or an instrument. (a) is
small and gives the stimuli that (b) and (c) are later tested on. (b) is the
narrowest piece that is new and testable against sonore. Expected difficulty
to watch in (b): gradient descent on f0 gets stuck at octave errors, which the
authors also report [§4.2.2]; restarts are part of the milestone.

### D4. Likelihood representation

**Options.**

- **(A) BASS's gammatonegram** as in §1.1 item 6. For: same numbers as the
  paper. Against: the FFT pooling has no exact counterpart in sonore to test
  against, and the authors call its resolution "a major limitation" [§4.2.3].
- **(B) sonore's ERB filterbank envelopes, made differentiable**, with dB
  compression, a floor and Gaussian noise as in BASS. For: testable against
  sonore's `Filterbank`/`Envelopes` (which have exact inverses and adjoints);
  close in kind to chcochleagram. Against: not identical to BASS, so BASS
  numbers do not transfer directly.
- **(C) Add a second representation later**: periodicity (the authors'
  first-named fix), modulation (for fast AM), or binaural cues (for space).

**Recommendation: (B) now, behind one likelihood interface so (A) or (C) can
be added without changing renderers.** Floor and σ are parameters, defaulting
to BASS's values (20 dB, σ = 10) so results stay comparable in kind. Which
second representation to add is decided after milestone (b) shows where (B)
fails.

**Update 2026-10-05.** Both are implemented: `Cochleagram` (B) and
`FFTCochleagram` (A, BASS's FFT approximation, PR #11). Below about 600 Hz
(A)'s channels are 2 to 3.6 times wider than half-ERB gammatones, and with it
the mistuned-harmonic thresholds move most of the way to the paper's. Cho
chose (A) as the default for reproducing the paper's results; (B) stays
available.

**Update 2026-10-06.** BASS does not calibrate (A)'s channels: its pooled
magnitudes are divided by the FFT size, so a 60 dB tone reads about 47 dB at
100 Hz and 56 dB at 3 kHz, nearer the 20 dB floor. `FFTCochleagram` now does the same by default (within 0.6 dB of BASS's code
per channel); Cho chose this default on 2026-10-06. `bass_gain=False` keeps
the calibrated channels.

### D5. Inference method

**Options.** Point estimates by gradient; stochastic variational inference as
in BASS; sampling (HMC, SMC); amortized or simulation-based inference.

**Points that constrain the choice.**

- Comparing hypotheses with different numbers of sources or events needs
  **marginal probabilities, not point estimates**, because densities of
  different dimensionality are not comparable [App. B.1.1]. Any method used at
  (c) must estimate them.
- The paper's own amortized proposals (detectron2) were the costliest part to
  build and still failed for wrong sound types [§4.2.2].

**Recommendation.** For (b): MAP point estimates by gradient with several
restarts; no marginal likelihood needed. For (c): start with **enumerative
inference** (Cho or the experiment defines the few hypotheses; SVI fits each;
importance sampling compares them), which covers most of the paper's
psychophysics and needs no proposal network. Choose between SVI and sampling
at (c) by measuring both on one illusion. Amortized proposals only if
open-ended scenes become a goal.

### D6. Compute

**What the paper needed:** see §1.2. As a rough scale, the mistuned-harmonic
experiment alone is 63 stimuli × 3 optimizations (one single-source
hypothesis, two initializations of the two-source one) × 10 seeds = 1,890
optimizations [App. C.6, §2.4]; at the reported 30 minutes to 2 hours each
that is about 950 to 3,800 GPU hours [estimate: the counts are from the paper,
the per-optimization time is the paper's range for enumerative inference
App. B.3.2, and our implementation's speed is unknown].

**Options.** Laptop CPU; Colab GPU; a university cluster; paid cloud GPUs.

**Recommendation.** Size milestones (a) and (b) to run in minutes on Colab or
a laptop, then measure seconds per iteration with a `tools/` script before
planning (c). **Question for Cho: what do you have access to?** The answer
bounds scene length and the number of hypotheses at (c).

### D7. Licence and reuse

**Options for this project:** MIT (as sonore), BSD-3-Clause, or Apache-2.0
(adds an explicit patent grant).

**Reuse of BASS.** The repository has no licence [code], so its code cannot
be copied without permission. The paper is CC BY 4.0 [paper, p. 1], and the
model, algorithms and stimulus specifications it describes can be
re-implemented from the text with citation. Running BASS's code to compare
outputs copies nothing into this project.

**Recommendation: MIT, matching sonore.** Implement only from the paper and
sonore, cite the paper wherever an equation or parameter comes from it, and
keep names taken from the paper unchanged. Ask Maddie whether the code could
carry a licence anyway, since later comparison (sequential inference, the
fitted priors) would benefit. chcochleagram's licence is to be checked before
it is used even as a test dependency.

### D8. Prior on the number of sources

BASS samples with one prior and scores with another (§1.1 item 1).

**Options.**

- **(A) Rate per second, used everywhere.** Poisson(λ × duration) with λ = 1
  per second, zero-truncated, and the same distribution in sampling and
  scoring; the truncation adds a normalizing term,
  −log(1 − exp(−λ × duration)), to log p(n). For: matches the paper's text;
  a longer sound plausibly holds more sources; one distribution, so samples
  and scores agree. Against: BASS's inference numbers were produced under
  (B), so log odds will not match BASS exactly.
- **(B) Poisson(1) per scene, used everywhere.** For: matches what BASS's
  inference actually scored. Against: contradicts the paper's text; the prior
  expectation of one source does not grow with scene length.
- **(C) Reproduce BASS exactly**, (A) for sampling and (B) for scoring. For:
  bit-level comparison with BASS. Against: an inconsistent model by design.

**Recommendation: (A), with λ as a parameter.** One distribution for sampling
and scoring is the clean design, and the authors report that this prior's
parameters had little effect on their results [App. A.4]. If a comparison with
BASS log odds ever needs it, (B) is a one-line scoring option, not the default.
This only bites at milestone (c), so it can be decided then. It is also worth
asking Maddie whether the mismatch was intended (§6).

---

## 5. Decisions as accepted (2026-10-05)

Cho accepted D1 to D8 as recommended. Two answers needed detail:

- **D3:** milestone (a), the three stimuli, then (b‴), two simultaneous notes,
  which Cho singled out as the simplest new problem. (b) and (b″) share its
  renderer and follow from it.
- **D6:** compute is a personal laptop (CPU only) and a Windows 10 PC with an
  NVIDIA GPU, no cluster. Everything up to and including (b‴) is sized for
  CPU; the GPU is an optional `device` argument, never required. The code
  stays OS-independent (no SLURM, no POSIX-only paths) so it runs on the
  Windows PC. Seconds per iteration on both machines are measured by a
  `tools/` script before anything larger is planned.

## 5a. The questions, short form (as put to Cho)

1. **D1** PyTorch or JAX? (recommended: PyTorch)
2. **D2** sonore as a test oracle plus optional extra, never in the core?
   (recommended: yes)
3. **D3** First milestone (a) three stimuli, then (b) a fixed-filter voice-like
   source or (b″) an instrument source with a harmonic-number spectrum times a
   fixed body filter, recovered by gradient? (recommended: (a) then (b) or
   (b″), your choice of target; space comes after)
4. **D4** sonore's ERB envelopes as the first likelihood, BASS's floor and σ
   as defaults? (recommended: yes)
5. **D5** MAP for (b), enumerative inference first at (c)? (recommended: yes)
6. **D6** What compute do you have: laptop, Colab, a cluster, paid cloud?
7. **D7** MIT, and implement only from the paper? (recommended: yes)
8. **D8** Source-count prior: rate per second, zero-truncated, the same in
   sampling and scoring? (recommended: yes; can wait until milestone (c))

Also for Cho, not decisions: create the empty repository `choyun1/sonore-inference`
when ready (I can push the design doc there as the first commit on a branch),
and the updated questions for Maddie and Josh are in §6.

---

## 6. Questions for Maddie and Josh, revised

- Why is one frozen noise draw shared by all noise sources, rather than one
  frozen draw per source? (The paper explains freezing, not sharing.)
- Why does the harmonic spectrum shift with f0? The code calls the fixed
  filter "less numerically stable"; was that the only reason?
- The code samples the number of sources at a rate of 1 per second
  (zero-truncated) but scores it as Poisson(1) per scene. Was that mismatch
  intended, and which one did you mean as the model?
- Could the code carry a licence (MIT or BSD)?
- Is anyone extending BASS to voices with time-varying formants, to binaural
  scenes or to reverberation, which the paper lists as future work?
- What GPU did a typical hypothesis optimization run on?

---

## 7. Instruments, timbre and track extraction

Added 2026-10-04 after Cho asked whether instrument identity could be the
first target, with extracting individual tracks from a multi-instrument
recording as the long-term goal.

**What the paper says.** The word "timbre" does not appear, and there is no
instrument-identification or timbre experiment. Instrument identity is
implicit: a source has one spectral shape shared by all its events (the
paper's example is "a filter corresponds to the instrument body") plus its own
priors on f0, level and timing [App. A.2], so notes are grouped into a source,
not labelled. The harmonic spectrum prior was fitted to classical-instrument
recordings (URMP) and birdsong [App. A.3]. Instruments appear among the
everyday-sound failures, and each failure points to a missing piece:

- A plucked violin note and the clarinet notes after it were merged into one
  source, "because the model cannot represent the spectral differences that
  distinguish the two" [§3.3.3.2, Fig. G.4]. This is a timbre failure.
- A bowed cello note, a trumpet note and overlapping piano notes were
  explained as noise rather than as periodic sounds, which the authors put
  down to the cochleagram's poor periodicity resolution [Fig. G.7, §4.2.3].
- A struck bell was split into parts that decay at different rates [Fig. G.3],
  and a flute note's breath noise was split from its tone [Fig. G.2C].
- Hybrid Transformer Demucs, a music stem separator, was one of the comparison
  networks. It matched the illusions worse than BASS did, but the paper
  evaluated it on illusions, not on separation quality [§3.2.2].

**What this means for the source model.** Neither BASS's Panpipes nor a fixed
filter fits instruments on its own. Body resonances (violin, guitar, piano
soundboard) are fixed in frequency, which favours a fixed filter. Some spectral
features are tied to harmonic number instead; the clarinet's weak even
harmonics in its low register are the textbook case (Fletcher & Rossing, *The
Physics of Musical Instruments*, 1998). A two-part spectrum, harmonic-number
excitation times a fixed body filter, covers both. That is option (b″) in D3.
Steady spectrum is also not the whole of timbre. Attack time and how the
spectrum changes over a note are among the main dimensions found in timbre
dissimilarity studies (McAdams et al., 1995, *Psychological Research* 58,
177–192). Decay shape, noise components (bow, breath) and inharmonicity (piano
strings) are also absent from BASS. Those are later additions, chosen once (b″)
shows what the steady-spectrum model can and cannot separate.

**Relation to music source separation.** Supervised separators such as
Demucs, trained on MUSDB, are far ahead on waveform quality for fixed stem
classes (drums, bass, vocals, other) and are fast. An analysis-by-synthesis
approach is unlikely to match them on that measure. Its possible advantages are
different: per-instrument parameters you can read (f0 track, spectrum,
onsets), no training data for the particular instruments, and a direct link to
perception. Two practical points:

- BASS's output sources are *renderings* of the inferred scene, not separated
  audio. Extracting real tracks would mean using the rendered sources as
  time-frequency masks on the mixture, a standard step that is not in BASS.
- Prior art to read before claiming anything: Bayesian harmonic models for
  music with MCMC (Davy & Godsill, early 2000s) and unsupervised separation
  with differentiable source-filter synthesizers (Schulze-Forster et al.,
  IEEE/ACM TASLP, 2023). I cite both from memory and have not yet checked
  them.

URMP, which BASS used for its priors, has separate instrument tracks and their
mixes, so it could serve as test data for both (b″) and track extraction
(its licence is not yet checked).

---

## References

- Cusimano, M., Hewitt, L. B., & McDermott, J. H. (2024). Listening with
  generative models. *Cognition*, 253, 105874.
  doi:10.1016/j.cognition.2024.105874 (CC BY 4.0).
- Nix, J., & Hohmann, V. (2007). Combined estimation of spectral envelopes and
  sound source direction of concurrent voices by multidimensional statistical
  filtering. *IEEE TASLP*, 15(3), 995–1008.
- Turner, R. E. (2010). *Statistical models for natural sounds*. PhD thesis,
  University College London.
- Hensman, J., Matthews, A., & Ghahramani, Z. (2015). Scalable variational
  Gaussian process classification. *AISTATS*.
- Feather, J., Leclerc, G., Mądry, A., & McDermott, J. H. (2023). Model
  metamers reveal divergent invariances between biological and artificial
  neural networks. *Nature Neuroscience*, 26(11), 2017–2034.
- Ellis, D. P. W. (2009). Gammatone-like spectrograms (software).
