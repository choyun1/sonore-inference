# A6. BASS's own code on two conditions

**Status:** done (2026-10-07). **Paper target:** Fig. 7D (right).
**Code:** BASS (github.com/mcusi/bass at `48e3755`), run locally on Cho's
PC. BASS has no licence, so none of its code, or our local fixes to it, is
in this repository; only its printed results are.

## Question

A5 leaves two gaps: 100 Hz harmonic 1 and 200 Hz harmonic 2. Which come from
our model, and which from how the paper turned evidence into a threshold?

## Setup

- BASS's `full_enumerative` configuration (8000 variational steps per
  hypothesis), seed 0, one job at a time on an RTX 2070 SUPER; all 42 jobs
  finished (about 23 h).
- Stimuli: 100 Hz harmonic 1 and 200 Hz harmonic 2, the paper's 7 mistuning
  levels; hypotheses h, hw and hwq (harmonic; harmonic plus whistle; harmonic
  plus whistle plus noise).
- Log odds as in BASS's analysis: max(ELBO hw, ELBO hwq) − ELBO h.
- Thresholds by two rules: Eqn 2 as we read it, and the rule in BASS's
  analysis code, which snaps the threshold to a level or the midpoint between
  two levels.

## Results

Thresholds, % of f0 (`data/A6/thresholds.txt`, which also lists every
ELBO). Ours is A5's full model on the same 7 levels.

| Condition | Paper's model (by eye) | BASS run, BASS's rule | BASS run, Eqn 2 | Ours, BASS's rule | Ours, Eqn 2 |
| --- | --- | --- | --- | --- | --- |
| 100 Hz, harmonic 1 | about 45 | 40 | 39.6 | 30 | 25.0 |
| 200 Hz, harmonic 2 | about 20 | 20 | 14.8 | 20 | 15.0 |

Ours by BASS's rule on all 9 conditions, 7 levels:
`data/A6/ours_bass_rule.txt`.

## What it says

At 200 Hz harmonic 2, BASS's run and ours agree by both rules, and BASS's
rule gives the paper's value. So that gap was the threshold rule, not the
model. At 100 Hz harmonic 1, BASS's run gives 40 and ours 30 by the same
rule: that one is a real difference between the models.

## How the numbers were made

The ELBOs are BASS's `metrics.json` outputs. The Eqn 2 values apply
`sonore_inference.thresholds` to the log odds. The BASS-rule values come
from a local script in the project's files that re-implements BASS's
analysis rule; it is not committed, for the licence reason above.

## Open

- One seed against the paper's 10. At 100 Hz harmonic 1 the 50% level swings
  back to one source (log odds −9.0), so more seeds would show whether 40
  holds.
- Which rule made the paper's figure: BASS's code suggests its own rule.
- Candidate causes of the 100 Hz harmonic 1 difference, untested: BASS
  infers the GP variance and lengthscale per sound, and its evidence is the
  log-mean-exp of all importance scores seen during training.
