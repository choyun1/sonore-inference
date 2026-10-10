"""Band energy in short blocks, and its expected value for a random
spectrogram in a room (docs/design/source-reverb.md, R3-R4).

The representation is the energy of each band of sonore's half-cosine ERB
filterbank (by default the source's 39 bands, 20 to 4000 Hz) summed over
10 ms blocks, in dB. It is linear in power before the log, so the expected
energy of a reverberant sound is the source's expected energy convolved,
band by band, with the room's expected gain (:class:`~sonore_inference.room.RoomGain`).

The source is ``sonore.gaussian_spectrogram`` on a Gaussian noise carrier
(``env.to_sound(noise).ramp(0.01)``). In band ``c`` the envelope
``e_c(t) = sum_k w_k(t) a_ck`` (raised-cosine windows ``w_k`` every 10 ms,
cell amplitudes ``a_ck = 10^(level/20)``) multiplies the fine structure
``cos(phase)`` of that band of the noise; the bands are re-filtered and
summed, cut to the sound's length and ramped. Its expected energy in
analysis band ``b`` treats the envelopes as slow:

    E[y_b(t)^2] = sum_{c, c'} A_bcc' e_c(t) e_c'(t),
    A_bcc' = mean_f |H_b|^2 H_c H_c' S_cc',

where ``S_cc'`` is the cross-spectrum of the fine structures of bands ``c``
and ``c'``. For Gaussian noise it is exact: the phases of two jointly
Gaussian analytic signals with complex correlation ``rho`` have
``E[cos phi cos phi'] = pi/8 Re[rho 2F1(1/2, 1/2; 2; |rho|^2)]``. Only
neighbouring ``c, c'`` overlap. Two time smearings are added: each
synthesis filter's energy impulse response before the cut to the sound's
length, and the analysis filter's after it. In the lowest bands both last
tens of ms.

What is left out is the carrier's fluctuation inside each block, and the
room's (R3). tools/block_power_check.py measures both against renders; the
likelihood takes them as a bias and a noise in dB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from .room import cosine_responses


def phase_correlation(z: np.ndarray) -> np.ndarray:
    """``2F1(1/2, 1/2; 2; z)`` for ``0 <= z <= 1``, from 1 to 4/pi:
    ``4 / (pi z) (E(k) - (1 - z) K(k))`` with ``k^2 = z``, the complete
    elliptic integrals by the arithmetic-geometric mean."""
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    a, b = np.ones_like(z), np.sqrt(1 - z)
    c_sum, power = 0.5 * z, 0.5
    for _ in range(40):
        c = (a - b) / 2
        a, b = (a + b) / 2, np.sqrt(a * b)
        power *= 2
        c_sum = c_sum + power * c**2
    k_int = np.pi / (2 * a)
    e_int = k_int * (1 - c_sum)
    small, one = z < 1e-4, z >= 1.0
    safe = np.where(small, 1.0, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = 4 / (np.pi * safe) * (e_int - (1 - z) * k_int)
    return np.where(small, 1 + z / 8 + 3 * z**2 / 64, np.where(one, 4 / np.pi, exact))


@dataclass(frozen=True)
class BlockPower:
    """Block band energies of a recording ``n_blocks`` blocks long, and the
    expected ones for a random spectrogram of ``duration`` seconds that
    starts the recording, with raised-cosine ramps of ``ramp`` [s]."""

    fs: float = 20000.0
    block: float = 0.010
    n_blocks: int = 120
    n_bands: int = 39
    f_lo: float = 20.0
    f_hi: float = 4000.0
    window: float = 0.020
    duration: float = 0.4
    ramp: float = 0.010
    n_freqs: int = 1 << 16
    kernel_tail: float = 1e-5

    @property
    def block_samples(self) -> int:
        return int(round(self.block * self.fs))

    @property
    def n_samples(self) -> int:
        return self.n_blocks * self.block_samples

    @property
    def source_samples(self) -> int:
        return int(round(self.duration * self.fs))

    @property
    def n_windows(self) -> int:
        """Cells per band of the source grid, as ``gaussian_spectrogram`` draws them."""
        return int(np.ceil(((self.source_samples - 1) / self.fs) / (self.window / 2))) + 1

    def _responses(self, freqs: np.ndarray) -> np.ndarray:
        """Analysis responses without the edge filters, ``(len(freqs), n_bands)``."""
        return cosine_responses(np.abs(freqs), self.n_bands, self.f_lo, self.f_hi)[:, 1:-1]

    def analyze(self, waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Energy of each band in each block, ``(n_bands, n_blocks)``. The
        waveform is cut or zero-padded to the recording's length and padded
        by 0.5 s more at both ends, so the filters' ringing cannot wrap.
        Equal to sonore's ``Filterbank.analyze(sound, pad=0.5)``."""
        n = self.n_samples
        x = torch.as_tensor(np.array(waveform, float)).reshape(-1)[:n]
        pad = int(round(0.5 * self.fs))
        n_fft = n + 2 * pad
        x = torch.nn.functional.pad(x, (pad, n_fft - pad - x.numel()))
        response = torch.from_numpy(self._responses(np.fft.rfftfreq(n_fft, 1 / self.fs)))
        bands = torch.fft.irfft(torch.fft.rfft(x)[:, None] * response, n=n_fft, dim=0)[pad : pad + n]
        return (bands**2).reshape(self.n_blocks, self.block_samples, self.n_bands).sum(1).T

    @cached_property
    def _envelope_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The raised-cosine windows ``(source_samples, n_windows)`` and the
        squared ramp ``(source_samples,)``, as sonore makes them."""
        t = np.arange(self.source_samples) / self.fs
        offset = (t[:, None] - self.window / 2 * np.arange(self.n_windows)[None, :]) / self.window
        windows = np.where(np.abs(offset) < 0.5, 0.5 * (1 + np.cos(2 * np.pi * offset)), 0.0)
        gain = np.ones(self.source_samples)
        n_ramp = int(round(self.ramp * self.fs))
        if n_ramp:
            phase = np.linspace(0, 1, n_ramp, endpoint=False) + 0.5 / n_ramp
            up = (1 - np.cos(np.pi * phase)) / 2
            gain[:n_ramp], gain[-n_ramp:] = up, up[::-1]
        return torch.from_numpy(windows), torch.from_numpy(gain**2)

    @cached_property
    def _mixing(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``A0[b, c] = A_bcc`` and ``A1[b, c] = A_bc(c+1)`` (module docstring),
        on a two-sided grid of ``n_freqs`` frequencies."""
        freqs = np.fft.fftfreq(self.n_freqs, 1 / self.fs)
        h = self._responses(freqs)
        analytic = np.where(freqs[:, None] > 0, 2 * h, 0.0)
        power = (analytic**2).mean(0)
        mixing = []
        for lag in (0, 1):
            out = np.zeros((self.n_bands, self.n_bands - lag))
            for c in range(self.n_bands - lag):
                pair = analytic[:, c] * analytic[:, c + lag]
                rho = np.fft.ifft(pair) / np.sqrt(power[c] * power[c + lag])
                correlation = np.pi / 8 * np.real(rho * phase_correlation(np.abs(rho) ** 2))
                spectrum = np.real(np.fft.fft(correlation))
                out[:, c] = (h**2).T @ (h[:, c] * h[:, c + lag] * spectrum) / self.n_freqs
            mixing.append(torch.from_numpy(out))
        return mixing[0], mixing[1]

    @cached_property
    def _kernels(self) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energy impulse responses of the filters, zero phase, centered,
        cut where less than ``kernel_tail`` of the energy lies beyond:
        ``(half, synthesis (n_bands, 2 half + 1), synthesis of neighbouring
        pairs (n_bands - 1, ...), analysis)``. Synthesis and analysis use the
        same filters, so the first and last are equal."""
        n = 1 << int(math.ceil(math.log2(2 * self.fs)))
        h = self._responses(np.fft.rfftfreq(n, 1 / self.fs))
        energy = np.fft.fftshift(np.fft.irfft(h, n=n, axis=0) ** 2, axes=0)
        energy /= energy.sum(0, keepdims=True)
        lag = np.abs(np.arange(n) - n // 2)
        order = np.argsort(lag, kind="stable")
        outside = 1 - np.cumsum(energy[order], axis=0)
        half = int(lag[order][np.argmax((outside < self.kernel_tail).all(1))])
        kernel = energy[n // 2 - half : n // 2 + half + 1].T
        kernel = torch.from_numpy(kernel / kernel.sum(1, keepdims=True))
        return half, kernel, 0.5 * (kernel[:-1] + kernel[1:]), kernel

    def _smear(self, x: torch.Tensor, kernel: torch.Tensor, half: int, length: int) -> torch.Tensor:
        """Each row of ``x`` (starting at t = 0) convolved with its centered
        kernel; ``length`` samples from t = 0 (what rings before 0 is cut)."""
        n_fft = 1 << int(math.ceil(math.log2(max(x.shape[1] + kernel.shape[1], half + length))))
        out = torch.fft.irfft(torch.fft.rfft(x, n=n_fft) * torch.fft.rfft(kernel, n=n_fft), n=n_fft)
        return out[:, half : half + length]

    # Products of two cell amplitudes that the envelopes' products contain:
    # (band step c' - c, window step k' - k, factor). e_c^2 has a_ck^2 and
    # 2 a_ck a_c(k+1); e_c e_c+1 (counted twice) has a_ck a_c+1,k' for k' = k, k+1
    # and k - 1, the last written as a_c(k+1) a_c+1,k.
    _TERMS = ((0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 2.0), (1, -1, 2.0))

    def _sample_power(self, products: list[torch.Tensor]) -> torch.Tensor:
        """Expected power at each sample ``(n_bands, n_samples)`` from the
        envelope products ``[e_c^2, e_c e_c+1]`` at each sample of the source."""
        _, ramp = self._envelope_weights
        a0, a1 = self._mixing
        half, synthesis, synthesis_pairs, analysis = self._kernels
        m = self.source_samples
        same = self._smear(products[0], synthesis, half, m) * ramp
        cross = self._smear(products[1], synthesis_pairs, half, m) * ramp
        return self._smear(a0 @ same + 2 * a1 @ cross, analysis, half, self.n_samples)

    def _blocks(self, power: torch.Tensor) -> torch.Tensor:
        return power.reshape(power.shape[0], self.n_blocks, self.block_samples).sum(-1)

    def source_energy_direct(self, levels_db: torch.Tensor) -> torch.Tensor:
        """:meth:`source_energy` computed sample by sample (slow; the reference)."""
        windows, _ = self._envelope_weights
        envelopes = 10 ** (levels_db.to(torch.float64) / 20) @ windows.T
        products = [envelopes**2, envelopes[:-1] * envelopes[1:]]
        return self._blocks(self._sample_power(products))

    @cached_property
    def _responses_to_products(self) -> list[torch.Tensor]:
        """For each term of ``_TERMS``, the block energies that a unit product
        of cell amplitudes makes: ``(bands c, windows k, b - c + 1, n_blocks)``
        for analysis bands ``b = c - 1 .. c + 2`` (the mixing reaches no
        further). The model is linear in these products, so this is
        :meth:`source_energy_direct` taken apart."""
        windows, ramp = self._envelope_weights
        a0, a1 = self._mixing
        half, synthesis, synthesis_pairs, analysis = self._kernels
        m, n_c = self.source_samples, self.n_bands
        out = []
        for band_step, window_step, factor in self._TERMS:
            rows = n_c - band_step
            kernel, mixing = (synthesis, a0) if band_step == 0 else (synthesis_pairs, a1)
            n_k = self.n_windows - abs(window_step)
            first = max(0, -window_step)
            response = torch.zeros(rows, n_k, 4, self.n_blocks, dtype=torch.float64)
            for k in range(n_k):
                shape = windows[:, k + first] * windows[:, k + first + window_step]
                smeared = self._smear(shape.expand(rows, m), kernel, half, m) * ramp
                for offset in range(-1, 3):
                    c = torch.arange(rows)
                    c = c[(c + offset >= 0) & (c + offset < n_c)]
                    if len(c) == 0:
                        continue
                    weights = mixing[c + offset, c]
                    power = weights[:, None] * smeared[c]
                    power = self._smear(power, analysis[c + offset], half, self.n_samples)
                    response[c, k, offset + 1] = factor * self._blocks(power)
            out.append(response)
        return out

    def source_energy(self, levels_db: torch.Tensor) -> torch.Tensor:
        """Expected energy ``(n_bands, n_blocks)`` of the source alone in each
        analysis band and block, from its cell levels in dB
        ``(n_bands, n_windows)``, mean spectrum included."""
        amplitude = 10 ** (levels_db.to(torch.float64) / 20)
        energy = torch.zeros(self.n_bands + 3, self.n_blocks, dtype=torch.float64)
        terms = zip(self._TERMS, self._responses_to_products, strict=True)
        for (band_step, window_step, _), response in terms:
            n_k = self.n_windows - abs(window_step)
            first = max(0, -window_step)
            rows = self.n_bands - band_step
            products = (
                amplitude[:rows, first : first + n_k]
                * amplitude[band_step : band_step + rows, first + window_step : first + window_step + n_k]
            )
            by_offset = torch.einsum("ck,ckoj->ocj", products, response)
            for offset in range(-1, 3):
                energy = energy.index_add(0, torch.arange(rows) + offset + 1, by_offset[offset + 1])
        return energy[1 : self.n_bands + 1]

    # Derivatives of source_energy with respect to the cell levels, for the
    # Laplace approximation (docs/design/source-reverb.md, R5), where
    # autograd's Hessian takes about a minute. Each product of two amplitudes
    # is P = exp(kappa (l_x + l_y)), kappa = ln 10 / 20, so dP/dl_x = dP/dl_y
    # = kappa P and every second derivative in (l_x, l_y) is kappa^2 P.

    def _cells_and_products(self, levels_db: torch.Tensor):
        """For each term of ``_TERMS``: flat indices of its two cells (x, y),
        each ``(rows, n_k)``, their products of amplitudes, and its response."""
        amplitude = 10 ** (levels_db.detach().to(torch.float64) / 20)
        index = torch.arange(self.n_bands * self.n_windows).reshape(self.n_bands, self.n_windows)
        terms = zip(self._TERMS, self._responses_to_products, strict=True)
        for (band_step, window_step, _), response in terms:
            n_k = self.n_windows - abs(window_step)
            first = max(0, -window_step)
            rows = self.n_bands - band_step
            x = index[:rows, first : first + n_k]
            y = index[band_step : band_step + rows, first + window_step : first + window_step + n_k]
            yield x, y, amplitude.reshape(-1)[x] * amplitude.reshape(-1)[y], response

    def source_energy_jacobian(self, levels_db: torch.Tensor) -> torch.Tensor:
        """``d source_energy / d levels_db``, ``(n_bands, n_blocks, n_bands * n_windows)``
        (cells flattened band-major)."""
        kappa = math.log(10) / 20
        n_cells = self.n_bands * self.n_windows
        out = torch.zeros((self.n_bands + 3) * self.n_blocks * n_cells, dtype=torch.float64)
        blocks = torch.arange(self.n_blocks)
        for x, y, products, response in self._cells_and_products(levels_db):
            rows = torch.arange(x.shape[0])[:, None, None, None]
            offsets = torch.arange(4)[None, None, :, None]
            values = (kappa * products[:, :, None, None] * response).reshape(-1)
            out_row = ((rows + offsets) * self.n_blocks + blocks) * n_cells
            for cell in (x, y):
                out.index_add_(0, (out_row + cell[:, :, None, None]).reshape(-1), values)
        out = out.reshape(self.n_bands + 3, self.n_blocks, n_cells)
        return out[1 : self.n_bands + 1]

    def source_energy_curvature(self, levels_db: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``sum_bj weights[b, j] d^2 source_energy[b, j] / d levels_db^2``,
        ``(n_cells, n_cells)``, for weights ``(n_bands, n_blocks)``."""
        kappa = math.log(10) / 20
        n_cells = self.n_bands * self.n_windows
        padded = torch.zeros(self.n_bands + 3, self.n_blocks, dtype=torch.float64)
        padded[1 : self.n_bands + 1] = weights
        out = torch.zeros(n_cells * n_cells, dtype=torch.float64)
        for x, y, products, response in self._cells_and_products(levels_db):
            rows = x.shape[0]
            reached = torch.stack([padded[offset : offset + rows] for offset in range(4)], 1)
            q = (kappa**2 * products * torch.einsum("ckoj,coj->ck", response, reached)).reshape(-1)
            x, y = x.reshape(-1), y.reshape(-1)
            for i, j in ((x, x), (y, y), (x, y), (y, x)):
                out.index_add_(0, i * n_cells + j, q)
        return out.reshape(n_cells, n_cells)

    def reverberant(self, source: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """Source energy ``(n_bands, n_blocks)`` convolved block by block with
        a room's expected gain ``(n_bands, n_blocks)``."""
        n = self.n_blocks
        spectrum = torch.fft.rfft(source, n=2 * n) * torch.fft.rfft(gain, n=2 * n)
        return torch.fft.irfft(spectrum, n=2 * n)[..., :n]


# Single renders against the model at experiment (a)'s setting (RT60 0.4 s,
# DRR 10 dB), pooled over cells within 60 dB of the peak: tools/block_power_check.py,
# 4 spectrograms x 16 draws. Both grow toward the low bands (-1.9 and 4.3 dB
# below 400 Hz, -0.6 and 2.5 dB above 1.4 kHz) and with less direct sound.
BIAS_DB = -1.3
SIGMA_DB = 3.6


def log_likelihood(
    observed: torch.Tensor,
    expected: torch.Tensor,
    sigma_db: float | torch.Tensor = SIGMA_DB,
    floor_db: float = 60.0,
    bias_db: float | torch.Tensor = BIAS_DB,
) -> torch.Tensor:
    """Gaussian log likelihood in dB of observed block energies given the
    expected ones (both ``(n_bands, n_blocks)``).

    The prediction is ``10 log10(expected) + bias_db``: the mean of the log
    of a fluctuating energy lies below the log of its mean. Observation and
    prediction are both floored at ``floor_db`` below the observation's
    peak, as BASS does, so cells far below the peak count only as quiet.
    ``sigma_db`` and ``bias_db`` may be one value per band.
    """
    observed_db = 10 * torch.log10(observed.to(torch.float64).clamp_min(1e-300))
    floor = observed_db.max() - floor_db
    bias = torch.as_tensor(bias_db, dtype=torch.float64)
    sigma = torch.as_tensor(sigma_db, dtype=torch.float64)
    bias = bias[:, None] if bias.ndim == 1 else bias
    sigma = sigma[:, None] if sigma.ndim == 1 else sigma
    predicted_db = 10 * torch.log10(expected.clamp_min(1e-300)) + bias
    residual = torch.maximum(observed_db, floor) - torch.maximum(predicted_db, floor)
    density = -0.5 * (residual / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return density.expand_as(residual).sum()
