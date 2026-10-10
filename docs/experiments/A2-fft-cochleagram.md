# A2. BASS's FFT cochleagram

**Status:** done (2026-10-05). **Paper target:** Fig. 7D (right).
**Code:** PR [#11](https://github.com/choyun1/sonore-inference/pull/11), first on main at `40f7da4`.

## Question

A1's thresholds were far below the paper's model. Does the cochleagram
explain it? BASS computes its cochleagram with an FFT approximation (Ellis,
2009): a 25 ms spectrogram pooled into gammatone-shaped channels, rather
than a gammatone filterbank.

## Setup

As A1, with one change: the FFT cochleagram, calibrated per channel (the
calibration is removed in A4). Mistuning on a 22-level grid (0–10% in 1%
steps, then 12–20 in 2s, then 25–50 in 5s).

Measured effect on channel width (`data/A2/bandwidths.txt`, from
`tools/channel_bandwidths.py`): below about 600 Hz the FFT channels are
2.1–3.6 times wider than true half-ERB channels (64.4 Hz against 18.0 Hz
at 104 Hz; 72.1 Hz against 34.0 Hz at 401 Hz).

## Results

Thresholds, % of f0 (`data/A2/thresholds.txt`). 1 row left out.

| f0 (Hz) | Harmonic | Paper's model (by eye) | A1 | A2, Laplace | A2, importance sampling |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | about 45 | 6.7 | 12.2 | 12.0 |
| 100 | 2 | n.m. | 10.1 | 39.5 | 39.6 |
| 100 | 3 | n.m. | 15.4 | 41.6 | 42.8 |
| 200 | 1 | about 10 | 3.5 | 5.6 | 5.5 |
| 200 | 2 | about 20 | 7.4 | 11.1 | 11.1 |
| 200 | 3 | about 21 | 11.1 | 17.8 | 17.3 |
| 400 | 1 | about 5 | 2.5 | 2.6 | 2.5 |
| 400 | 2 | about 10 | 5.1 | 6.5 | 6.2 |
| 400 | 3 | about 13 | 7.1 | 9.5 | 9.1 |

## What it says

The wider low-frequency channels move every threshold up, most at 100 Hz,
where harmonics 2 and 3 come close to the paper's "not measurable". The
reading of Eqn 2 and the component phases were also checked at this stage
and ruled out as causes (a plain 50% crossing agrees within 0.4%; the
paper's coarser 7-level grid gives the same thresholds within 2% of f0).
Those two checks are recorded in the PR, not in committed outputs here.

## Reproduce

On today's main, `--cochleagram fft-calibrated` selects this cochleagram
(`fft` now means A4's uncalibrated version). Not re-run on today's main.

```
python tools/compare_mistuned_harmonic.py --f0 100 --harmonic 1 --cochleagram fft-calibrated \
    --percents 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50
python tools/mistuning_thresholds.py data/A2/f0-*.txt
```

Raw outputs: [`data/A2/`](https://github.com/choyun1/sonore-inference/tree/main/docs/experiments/data/A2). The first run of 400 Hz harmonic 2
stopped at 7% with an error (its traceback is kept in
`f0-400_h2.txt`); `f0-400_h2_rest.txt` is the rerun of the remaining
levels.

## Open

- 100 Hz harmonic 1 is still far below the paper (12.2 against about 45).
