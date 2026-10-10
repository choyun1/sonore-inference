"""RoomGain against sonore.synth_ir renders (docs/design/source-reverb.md, R3).

A 400 ms white-noise burst (a new draw each time) is convolved with a
``synth_ir`` room (a new draw each time). Both are split into the analysis
bands (sonore's cosine filterbank, 39 bands from 20 to 4000 Hz) and their
energy summed over 10 ms blocks. The model predicts the reverberant energy
as the burst's own block energies convolved, block by block, with
:class:`RoomGain`. The burst's energies are measured, not modelled, so what
is compared is the room alone.

Printed, over cells within 60 dB of each band's peak:

- the mean over draws against the model, in dB: how well the expected
  energy is modelled;
- single draws against the model, in dB: the residual one observation
  leaves, whose mean is the log bias and whose sd sets the likelihood's
  noise for the room (the source's own carrier adds more, step 3).

(Comparing the impulse response alone, without a source, is misleading in
the lowest bands: the narrow analysis filters ring for tens of ms, which in
a reverberant sound belongs to the source's band energy, not to the room.)

    python tools/room_gain_check.py [--draws 64]
"""

import argparse

import numpy as np
import sonore as so
import torch
from scipy.signal import fftconvolve

from sonore_inference.room import RoomGain

FS = 20_000.0
BLOCK = round(0.010 * FS)
BURST = 0.4


def block_energy(x: np.ndarray, bank, n_blocks: int) -> np.ndarray:
    """Band energy per 10 ms block, (bands, n_blocks), without the edge filters."""
    padded = np.zeros(n_blocks * BLOCK)
    padded[: min(len(x), len(padded))] = x[: len(padded)]
    bands = np.asarray(bank.analyze(so.Sound(padded, FS)).data).reshape(len(padded), -1)[:, 1:-1]
    return (bands**2).reshape(n_blocks, BLOCK, -1).sum(1).T


def predicted(source: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Block-by-block convolution of source energy with the room's gain."""
    n_blocks = gain.shape[1]
    return np.stack([np.convolve(s, g)[:n_blocks] for s, g in zip(source, gain, strict=True)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=64)
    args = parser.parse_args()
    bank = so.cosine_filterbank(39, 20.0, 4000.0)
    print(f"sonore {so.__version__}; {args.draws} draws; 39 bands, 20-4000 Hz; 10 ms blocks; 400 ms burst")
    print("rt60  drr   mean-of-draws vs model [dB]     single draw vs model [dB]")
    print("            mean    sd    max|err|         mean    sd")
    for rt60 in (0.2, 0.4, 0.8):
        n_blocks = round(BURST / 0.010) + int(np.ceil(1.2 * rt60 / 0.010))
        for drr in (None, 10.0, 0.0):
            gain = RoomGain(fs=FS, n_blocks=n_blocks, drr_db=drr).gain(torch.tensor(rt60)).numpy()
            observed, model = [], []
            for seed in range(args.draws):
                burst = so.gaussian_noise(BURST, FS, rng=1000 + seed).data.ravel()
                ir = so.synth_ir(rt60, FS, drr_db=drr, rng=seed).data.ravel()
                observed.append(block_energy(fftconvolve(burst, ir), bank, n_blocks))
                model.append(predicted(block_energy(burst, bank, n_blocks), gain))
            observed, model = np.stack(observed), np.stack(model)
            mean_model = model.mean(0)
            keep = mean_model > mean_model.max(1, keepdims=True) * 1e-6
            mean_err = 10 * np.log10(observed.mean(0)[keep] / mean_model[keep])
            single = 10 * np.log10(observed[:, keep] / model[:, keep])
            label = "tail" if drr is None else f"{drr:4.0f}"
            worst = np.abs(mean_err).max()
            print(
                f"{rt60:4.1f}  {label}  {mean_err.mean():6.2f} {mean_err.std():6.2f} {worst:7.2f}"
                f"          {single.mean():6.2f} {single.std():6.2f}"
            )


if __name__ == "__main__":
    main()
