"""Likelihood slices through the truth (docs/design/source-reverb.md, R4-R5).

One observation: a 400 ms ``sonore.gaussian_spectrogram`` on a noise
carrier in a ``synth_ir`` room (RT60 0.4 s, DRR 10 dB), analysed into 39
bands and 10 ms blocks over 1.2 s. The model is :class:`BlockPower` at the
drawn cell levels and :class:`RoomGain`. One thing at a time is moved off the
truth, the rest held there:

- RT60;
- the level of every cell, by the same number of dB;
- the source's offset, by moving the last cells' levels down (the source
  ending early) or the sound continuing at the last cell's level (later);
- the DRR, which (a) holds fixed at the truth.

Printed: log likelihood relative to the truth [nats], at floors of 60, 40
and 20 dB below the observation's peak and at the measured noise sigma and
twice it (design R4). Bias and sigma are from tools/block_power_check.py.

    python tools/likelihood_slices.py [--seed 0] [--plot slices.png]
"""

import argparse

import numpy as np
import sonore as so
import torch
from scipy.signal import fftconvolve
from sonore.core.utils import as_rng
from sonore.sources.gaussian_spectrogram import _correlated_field

from sonore_inference.blockpower import BIAS_DB, SIGMA_DB, BlockPower, log_likelihood
from sonore_inference.room import RoomGain
from sonore_inference.spectrotemporal import SpectrogramPrior

FS = 20_000.0
RT60, DRR = 0.4, 10.0
FLOORS = (60.0, 40.0, 20.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="save the slices as an image here")
    args = parser.parse_args()
    power, prior = BlockPower(), SpectrogramPrior()
    rng = as_rng(args.seed)
    field = _correlated_field(prior.n_bands, power.n_windows, prior.rho_band, prior.rho_time, rng)
    levels = torch.from_numpy(prior.mean_db()[:, None] + prior.sd_db * field)
    env = so.gaussian_spectrogram(power.duration, FS, rng=args.seed)
    carrier = so.gaussian_noise(power.duration, FS, rng=1000 + args.seed)
    x = env.to_sound(carrier).ramp(power.ramp).data.ravel()
    ir = so.synth_ir(RT60, FS, drr_db=DRR, rng=2000 + args.seed).data.ravel()
    observed = power.analyze(fftconvolve(x, ir))

    def expected(levels=levels, rt60=RT60, drr=DRR):
        gain = RoomGain(fs=FS, n_blocks=power.n_blocks, drr_db=drr).gain(rt60)
        return power.reverberant(power.source_energy(levels), gain)

    def early_or_late(shift):
        """Offset moved by ``shift`` windows: earlier cells silenced, or the
        last level held for later windows (the grid is cut at the sound's
        end, so a later offset is drawn as more sound at the end level)."""
        if shift <= 0:
            moved = levels.clone()
            if shift < 0:
                moved[:, shift:] = -200.0
            return expected(moved)
        longer = BlockPower(duration=power.duration + shift * power.block)
        moved = torch.cat([levels, levels[:, -1:].expand(-1, shift)], 1)
        gain = RoomGain(fs=FS, n_blocks=power.n_blocks, drr_db=DRR).gain(RT60)
        return power.reverberant(longer.source_energy(moved), gain)

    slices = {
        "RT60 [s]": (
            [0.1, 0.2, 0.3, 0.35, 0.38, 0.4, 0.42, 0.45, 0.5, 0.6, 0.8, 1.2, 2.0],
            RT60,
            lambda v: expected(rt60=v),
        ),
        "level [dB]": ([-6, -3, -1, -0.5, 0, 0.5, 1, 3, 6], 0, lambda v: expected(levels + v)),
        "offset [blocks]": ([-4, -2, -1, 0, 1, 2, 4], 0, lambda v: early_or_late(int(v))),
        "DRR [dB]": ([0, 5, 8, 9, 10, 11, 12, 15, 20], DRR, lambda v: expected(drr=v)),
    }
    print(
        f"sonore {so.__version__}; seed {args.seed}; RT60 {RT60} s, DRR {DRR} dB; "
        f"bias {BIAS_DB} dB, sigma {SIGMA_DB} dB and twice it"
    )
    print("log likelihood minus its value at the truth [nats]")
    results = {}
    for name, (values, truth, model) in slices.items():
        predictions = [model(v) for v in values]
        print(f"\n{name:16s}" + "".join(f"{v:8g}" for v in values))
        for sigma in (SIGMA_DB, 2 * SIGMA_DB):
            for floor in FLOORS:
                scores = np.array(
                    [log_likelihood(observed, p, sigma, floor, BIAS_DB).item() for p in predictions]
                )
                scores -= scores[values.index(truth)]
                results[(name, sigma, floor)] = (values, scores)
                label = f"sigma {sigma:.1f} floor {floor:.0f}"
                print(f"{label:16s}" + "".join(f"{s:8.1f}" for s in scores))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(slices), figsize=(4 * len(slices), 3.2))
        for ax, (name, (_, truth, _)) in zip(axes, slices.items(), strict=True):
            for sigma, style in ((SIGMA_DB, "-"), (2 * SIGMA_DB, "--")):
                for floor in FLOORS:
                    v, s = results[(name, sigma, floor)]
                    ax.plot(v, s, style, marker=".", label=f"σ {sigma:.1f}, floor {floor:.0f}")
            ax.axvline(truth, color="k", lw=0.5)
            ax.set_xlabel(name)
            ax.set_ylim(-400, 20)
            if name == "RT60 [s]":
                ax.set_xscale("log")
        axes[0].set_ylabel("log likelihood re truth [nats]")
        axes[-1].legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)


if __name__ == "__main__":
    main()
