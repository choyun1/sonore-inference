"""Effective bandwidth of cochleagram channels, measured with steady sinusoids.

For channels near the given frequencies, sweeps a 0.5 s sinusoid across
frequency and records each channel's level in its steady frames. The
equivalent rectangular bandwidth is the integral of the channel's power
response divided by its peak. Compares :class:`Cochleagram` (gammatone
filtering) with :class:`FFTCochleagram` (BASS's FFT approximation).

    python tools/channel_bandwidths.py
"""

import numpy as np
import torch

from sonore_inference.cochleagram import Cochleagram, FFTCochleagram, erb_bandwidth

FS = 20_000
TARGETS = (100, 200, 300, 400, 600, 800, 1200, 2400)


def response(cochleagram, channel, freqs):
    t = np.arange(int(0.5 * FS)) / FS
    levels = []
    for chunk in np.array_split(freqs, max(1, len(freqs) // 20)):
        tones = np.sqrt(2) * 1e-6 * 10**3 * np.cos(2 * np.pi * chunk[:, None] * t)  # 60 dB
        with torch.no_grad():
            levels.append(cochleagram(torch.tensor(tones))[:, channel, 10:-10].mean(-1).numpy())
    return 10 ** ((np.concatenate(levels) - 60) / 10)


def main():
    kinds = {
        "gammatone": Cochleagram(floor_db=-200),
        "fft": FFTCochleagram(floor_db=-200),
    }
    print("cf_hz  half_erb_hz  erb_gammatone_hz  erb_fft_hz")
    cfs = Cochleagram().cfs
    for target in TARGETS:
        channel = int(np.argmin(np.abs(cfs - target)))
        cf = cfs[channel]
        freqs = np.linspace(max(1.0, cf - 600), cf + 600, 601)
        widths = []
        for cochleagram in kinds.values():
            power = response(cochleagram, channel, freqs)
            widths.append(np.trapezoid(power, freqs) / power.max())
        print(f"{cf:6.0f}  {0.5 * erb_bandwidth(cf):11.1f}  {widths[0]:16.1f}  {widths[1]:10.1f}")


if __name__ == "__main__":
    main()
