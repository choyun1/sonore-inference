# A4. BASS's uncalibrated channel gain

> **Under review.** This is a working record, not reviewed results. The work was
> done with AI assistance (Claude), and Adrian Cho has not yet checked it
> carefully. Numbers, methods and interpretations may change.

**Status:** done (2026-10-06). **Paper target:** Fig. 7D (right).
**Code:** PR [#13](https://github.com/choyun1/sonore-inference/pull/13), first on main at `ceccaee`.

## Question

Shared background: [how the experiments work](index.md#how-the-experiments-work).


BASS's FFT cochleagram divides magnitudes by the FFT length and does not
calibrate each channel. Run through BASS's own code, a 60 dB tone reads
about 47 dB at 100 Hz and about 56 dB at 3 kHz, closer to the 20 dB floor at
low frequencies. Does matching that change the thresholds?

## Setup

As A3, with the FFT cochleagram's gain left uncalibrated, as in BASS. This
became the default on 2026-10-06 (Cho's choice).

## Results

Thresholds, % of f0 (`data/A4/thresholds.txt`). 7 rows left out.

| f0 (Hz) | Harmonic | Paper's model (by eye) | A3 | A4, Laplace | A4, importance sampling |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | about 45 | 15.0 | 23.5 | 22.9 |
| 100 | 2 | n.m. | n.m. | n.m. | n.m. |
| 100 | 3 | n.m. | n.m. | n.m. | n.m. |
| 200 | 1 | about 10 | 7.4 | 7.6 | 7.4 |
| 200 | 2 | about 20 | 13.5 | 13.6 | 13.2 |
| 200 | 3 | about 21 | 22.8 | 22.1 | 22.0 |
| 400 | 1 | about 5 | 3.6 | 4.5 | 4.4 |
| 400 | 2 | about 10 | 7.9 | 7.7 | 7.6 |
| 400 | 3 | about 13 | 12.9 | 12.5 | 11.6 |

## What it says

The gain moves 100 Hz harmonic 1 from 15.0% to 23.5% and barely changes the
rest. The 47 and 56 dB readings were measured with BASS's code on Cho's
machine and are recorded in PR #13, not in committed outputs here.

## Reproduce

`--cochleagram fft` (the default) is this cochleagram. Not re-run on
today's main.

```
python tools/compare_mistuned_harmonic.py --f0 100 --harmonic 1 --whistle-timing inferred --percents 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50
python tools/mistuning_thresholds.py data/A4/f0-*.txt
```

Raw outputs: [`data/A4/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A4).

## Open

- 100 Hz harmonic 1 still about 20 points below the paper.
