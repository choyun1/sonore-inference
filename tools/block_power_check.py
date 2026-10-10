"""BlockPower's expected energies against sonore renders (docs/design/source-reverb.md, R3-R4).

Each observation is a 400 ms ``sonore.gaussian_spectrogram`` (sonore's
defaults) on a Gaussian noise carrier, ramped by 10 ms, optionally convolved
with a ``synth_ir`` room, analysed into 39 bands and 10 ms blocks over 1 s.
The model is :meth:`BlockPower.source_energy` from the drawn cell levels,
convolved with :class:`RoomGain`. Every draw of carrier and room is new; the
spectrogram is redrawn every ``--draws`` observations. Spectrogram, carrier
and room never share a seed: sonore draws each from ``default_rng(seed)``, so
a shared seed makes them the same noise and the energies correlate.

Printed, over cells the model puts within 60 dB of its peak, separately for
blocks while the source plays and after it ends:

- the mean of the draws against the model [dB]: how well the expected
  energy is modelled;
- single draws against the model [dB]: the mean is the log bias and the sd
  the noise the likelihood has to allow (design R4), also pooled over all
  cells (the defaults ``BIAS_DB`` and ``SIGMA_DB`` in blockpower.py are the
  pooled values at RT60 0.4 s and DRR 10 dB).

Then single-draw bias and sd by thirds of the bands (low, mid, high).

    python tools/block_power_check.py [--sources 4] [--draws 16]
"""

import argparse

import numpy as np
import sonore as so
import torch
from scipy.signal import fftconvolve
from sonore.core.utils import as_rng
from sonore.sources.gaussian_spectrogram import _correlated_field

from sonore_inference.blockpower import BlockPower
from sonore_inference.cochleagram import erb_to_freq, freq_to_erb
from sonore_inference.room import RoomGain
from sonore_inference.spectrotemporal import SpectrogramPrior

FS = 20_000.0
SETTINGS = [(None, None), (0.2, 10.0), (0.4, 10.0), (0.8, 10.0), (0.4, 0.0), (0.4, "tail")]


def drawn_levels(prior: SpectrogramPrior, n_windows: int, seed: int) -> np.ndarray:
    """The cell levels [dB] ``gaussian_spectrogram(rng=seed)`` draws."""
    field = _correlated_field(prior.n_bands, n_windows, prior.rho_band, prior.rho_time, as_rng(seed))
    return prior.mean_db()[:, None] + prior.sd_db * field


def summary(errors: np.ndarray) -> str:
    return f"{errors.mean():6.2f} {errors.std():5.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--draws", type=int, default=16)
    args = parser.parse_args()
    power = BlockPower(n_blocks=100)
    prior = SpectrogramPrior()
    playing = np.arange(power.n_blocks) < round(power.duration / power.block)
    print(
        f"sonore {so.__version__}; {args.sources} spectrograms x {args.draws} draws; "
        "39 bands 20-4000 Hz, 10 ms blocks, 400 ms source, 1 s analysed"
    )
    print("               while playing               after the offset            all cells")
    print("               draws-mean    single        draws-mean    single        single")
    print("rt60  drr      mean    sd   mean    sd     mean    sd   mean    sd     mean    sd")
    by_band = {}
    for rt60, drr in SETTINGS:
        rows = {"playing": ([], []), "after": ([], [])}
        thirds = [[], [], []]
        for source in range(args.sources):
            env = so.gaussian_spectrogram(power.duration, FS, rng=source)
            model = power.source_energy(torch.from_numpy(drawn_levels(prior, power.n_windows, source)))
            if rt60 is not None:
                room = RoomGain(fs=FS, n_blocks=power.n_blocks, drr_db=None if drr == "tail" else drr)
                model = power.reverberant(model, room.gain(rt60))
            model = model.numpy().clip(1e-300)
            observed = []
            for draw in range(args.draws):
                # distinct seeds: the same seed would give carrier and room the same noise
                seed = 1000 * (source + 1) + draw
                carrier = so.gaussian_noise(power.duration, FS, rng=seed)
                x = env.to_sound(carrier).ramp(power.ramp).data.ravel()
                if rt60 is not None:
                    ir = so.synth_ir(rt60, FS, drr_db=None if drr == "tail" else drr, rng=10**6 + seed)
                    x = fftconvolve(x, ir.data.ravel())
                observed.append(power.analyze(x).numpy())
            observed = np.stack(observed)
            keep = model > model.max() * 1e-6
            mean_error = 10 * np.log10(observed.mean(0) / model)
            single = 10 * np.log10(observed / model)
            for name, blocks in (("playing", playing), ("after", ~playing)):
                cells = keep & blocks[None, :]
                rows[name][0].append(mean_error[cells])
                rows[name][1].append(single[:, cells].ravel())
            for i, bands in enumerate(np.array_split(np.arange(power.n_bands), 3)):
                cells = np.zeros_like(keep)
                cells[bands] = keep[bands]
                thirds[i].append(single[:, cells].ravel())
        label = "dry " if rt60 is None else f"{rt60:4.1f}"
        drr_label = "" if rt60 is None else ("tail" if drr == "tail" else f"{drr:4.0f}")
        line = f"{label}  {drr_label:4s} "
        for name in ("playing", "after"):
            means, singles = (np.concatenate(v) for v in rows[name])
            line += f"  {summary(means)} {summary(singles)}" if means.size else "        (no cells)"
        pooled = np.concatenate(rows["playing"][1] + rows["after"][1])
        print(line + f"    {summary(pooled)}")
        by_band[(label, drr_label)] = [summary(np.concatenate(t)) for t in thirds]
    print("\nsingle draws by thirds of the bands (mean, sd) [dB]")
    centers = erb_to_freq(freq_to_erb(prior.f_lo) + prior.band_spacing_erb * np.arange(1, power.n_bands + 1))
    thirds = np.array_split(np.arange(power.n_bands), 3)
    thirds_hz = [f"{centers[b[0]]:.0f}-{centers[b[-1]]:.0f} Hz" for b in thirds]
    print("rt60  drr       " + "     ".join(f"{t:17s}" for t in thirds_hz))
    for (label, drr_label), values in by_band.items():
        print(f"{label}  {drr_label:4s}     " + "     ".join(values))


if __name__ == "__main__":
    main()
