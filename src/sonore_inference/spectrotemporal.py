"""Prior on a random spectrogram after McDermott, Wrobleski & Oxenham (2011).

The source of the source-and-room problem (docs/design/source-reverb.md, R1)
is a grid of levels in dB, one per ERB band and time window, drawn from a
Gaussian whose correlation is ``exp(-band distance [ERB] /
band_correlation_erb) * exp(-time distance [s] / time_correlation)``. sonore
makes the test sounds (``sonore.gaussian_spectrogram``); this module is the
observer's prior on the same grid, so that the grid can be inferred.

An exponential correlation on a regular grid is a first-order
autoregression along each axis. So the field is whitened exactly by the
innovations ``(x[k] - rho x[k-1]) / sqrt(1 - rho^2)`` along time and then
along bands, and the log density needs no matrix of size (bands x windows)^2.
A prefix of the windows is itself such a field, so a grid cut short at the
end of a sound is scored correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .cochleagram import erb_to_freq, freq_to_erb


@dataclass(frozen=True)
class SpectrogramPrior:
    """Gaussian prior on a grid of band levels in dB, shape (n_bands, n_windows).

    Defaults are sonore's ``gaussian_spectrogram`` defaults, which follow
    McDermott, Wrobleski & Oxenham (2011): 39 half-cosine ERB bands from 20
    to 4000 Hz, 20 ms windows overlapping by half (a 10 ms grid step),
    correlation lengths 8.78 ERB and 154 ms (their 0.075 per filter and
    0.065 per window), and 14.1 dB per cell (the paper gives none; this is
    the 2017 implementation's variance of 0.5 in log10 amplitude).
    """

    n_bands: int = 39
    f_lo: float = 20.0
    f_hi: float = 4000.0
    window: float = 0.020
    band_correlation_erb: float = 8.78
    time_correlation: float = 0.154
    sd_db: float = 14.1

    @property
    def band_spacing_erb(self) -> float:
        """ERB distance between neighbouring band centers; the bands sit at
        equally spaced points strictly between ``f_lo`` and ``f_hi``."""
        return float(freq_to_erb(self.f_hi) - freq_to_erb(self.f_lo)) / (self.n_bands + 1)

    @property
    def step(self) -> float:
        """Grid step in time [s]: half a window."""
        return self.window / 2

    @property
    def rho_band(self) -> float:
        return math.exp(-self.band_spacing_erb / self.band_correlation_erb)

    @property
    def rho_time(self) -> float:
        return math.exp(-self.step / self.time_correlation)

    def mean_db(self) -> np.ndarray:
        """Mean level of each band [dB re the average band], rising with its
        width in Hz so that the long-term spectrum is flat on average."""
        spacing = self.band_spacing_erb
        centers = freq_to_erb(self.f_lo) + spacing * np.arange(1, self.n_bands + 1)
        widths = erb_to_freq(centers + spacing / 2) - erb_to_freq(centers - spacing / 2)
        return 10 * np.log10(widths / widths.mean())

    def whiten(self, deviation_db: torch.Tensor) -> torch.Tensor:
        """Standard-normal innovations of a grid of deviations from the mean
        [dB], shape (n_bands, n_windows). Inverse of :meth:`color`."""
        x = deviation_db / self.sd_db
        x = torch.cat([x[:, :1], (x[:, 1:] - self.rho_time * x[:, :-1]) / math.sqrt(1 - self.rho_time**2)], 1)
        x = torch.cat([x[:1], (x[1:] - self.rho_band * x[:-1]) / math.sqrt(1 - self.rho_band**2)], 0)
        return x

    def color(self, innovations: torch.Tensor) -> torch.Tensor:
        """Deviations from the mean [dB] from standard-normal innovations, by
        the same recursion as sonore's draw: along time, then along bands."""
        x = innovations.clone()
        a_t, a_b = math.sqrt(1 - self.rho_time**2), math.sqrt(1 - self.rho_band**2)
        for k in range(1, x.shape[1]):
            x[:, k] = self.rho_time * x[:, k - 1] + a_t * x[:, k]
        for k in range(1, x.shape[0]):
            x[k] = self.rho_band * x[k - 1] + a_b * x[k]
        return self.sd_db * x

    def log_prob(self, deviation_db: torch.Tensor) -> torch.Tensor:
        """Log density of a grid of deviations from the mean [dB]."""
        n_bands, n_windows = deviation_db.shape
        z = self.whiten(deviation_db)
        log_det = (
            n_bands * n_windows * math.log(self.sd_db)
            + n_bands * (n_windows - 1) * 0.5 * math.log(1 - self.rho_time**2)
            + (n_bands - 1) * n_windows * 0.5 * math.log(1 - self.rho_band**2)
        )
        return -0.5 * (z**2).sum() - 0.5 * z.numel() * math.log(2 * math.pi) - log_det

    def covariance(self, n_windows: int) -> torch.Tensor:
        """Full covariance [dB^2] of the flattened grid (band-major), for tests
        and small problems."""
        band = torch.arange(self.n_bands, dtype=torch.float64)
        time = torch.arange(n_windows, dtype=torch.float64)
        c_band = self.rho_band ** (band[:, None] - band[None, :]).abs()
        c_time = self.rho_time ** (time[:, None] - time[None, :]).abs()
        return self.sd_db**2 * torch.kron(c_band, c_time)
