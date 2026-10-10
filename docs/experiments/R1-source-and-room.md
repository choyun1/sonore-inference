# R1. Source and room (RT60) from one reverberant sound

**Status: in progress.** The model's building blocks are written and
checked; the inference experiments have not been run. This page will be
updated as they are.

**Paper target:** none in Cusimano et al. (2024), which has no room model;
its §4.3 names reverberation as one of the aspects of sound generation a
model of scenes should include, citing Traer & McDermott (2016). A
comparison with Traer & McDermott's listeners is optional and, if done, will
be labelled a comparison, not a validation. Design:
[source and room](../design/source-reverb.md) (R1–R8, accepted by Cho
2026-10-10).

## Question

One source plays in one room. From the single sound that results, can the
model infer the source's spectrotemporal statistics and the room's RT60?
Planned, from the design (R7):

- **(a) Recovery:** 10 seeds × RT60 in {0.2, 0.4, 0.8} s × three source
  settings. Success: the truth falls in the 90% posterior interval in about
  90% of runs, per parameter.
- **(b) Atypical rooms:** an exponential-room model given rooms with other
  decay shapes and spectral profiles; reported as the posterior median RT60
  over the true RT60, at three floors.

## Done so far

| Step | What | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Source prior (`SpectrogramPrior`), matching sonore's `gaussian_spectrogram` | Whitening sonore's drawn levels gives back its standard normals (to 1e-9, 3 seeds) | PR [#18](https://github.com/choyun1/sonore-inference/pull/18), tests |
| 2 | Room's expected band energy (`RoomGain`) | Mean of 64 rendered noise bursts within 0.1 dB of the model on average (after the fix found in step 3); a single draw is off by −0.7 to −1.6 dB on average, sd 2.8–4.2 dB | PR [#19](https://github.com/choyun1/sonore-inference/pull/19); `data/R1/room_gain_check.txt` (`tools/room_gain_check.py`) |
| 3 | Block-power likelihood (`BlockPower`) | One render against the model at RT60 0.4 s, DRR 10 dB, all cells: bias −1.3 dB, sd 3.6 dB (the likelihood uses these) | PR [#21](https://github.com/choyun1/sonore-inference/pull/21); `data/R1/block_power_check.txt` (`tools/block_power_check.py`) |
| 3 | Likelihood slices through the truth, one sound (RT60 0.4 s, DRR 10 dB) | At floors 60 and 40 dB every slice peaks at the truth; at a 20 dB floor (BASS's) RT60 is nearly flat (0.1 s costs 1.3 nats) | `data/R1/likelihood_slices.txt` (`tools/likelihood_slices.py`) |
| 4 | RT60 prior from Traer & McDermott's survey (270 IRs, run by Cho) | Gamma, shape 1.97, scale 0.220 s (median 0.362 s), better than a lognormal by 18 AIC; typical DRR 9.9 dB | PR [#22](https://github.com/choyun1/sonore-inference/pull/22); constants in `src/sonore_inference/room.py` (`tools/rt60_prior_fit.py`) |

The survey files are not redistributed; the prior fit runs where they are
downloaded, so its raw output is not here.

## Open

- Step 5 (Laplace grid over RT60 and the source, experiment (a)) and step 6
  (experiment (b)) are next.
- At BASS's 20 dB floor one sound barely constrains RT60, which matters for
  combining this with milestone (b)'s model.
- Licence of the survey's original files is unchecked.
