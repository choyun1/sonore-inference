"""Expected band energy of a synthetic room, for inference (docs/design/source-reverb.md, R2-R3).

sonore's ``synth_ir`` (Traer & McDermott, 2016) makes the rooms of the test
sounds: Gaussian noise is split into ERB bands, each band is multiplied by an
exponential decay whose RT60 and onset level follow the paper's regressions on
real rooms, the bands are re-filtered and summed, and a unit direct impulse is
added at a given direct-to-reverberant ratio. Each draw's noise differs. The
observer does not know the draw, so this module averages it out: it gives the
*expected* energy of the impulse response in each band of an analysis
filterbank and each time block, as a differentiable function of the median
RT60.

The output is a block-to-block gain: in band ``b``, for energy spread
evenly over one block, the expected energy the room puts ``k`` blocks later,
relative to the energy band ``b`` passes from a unit impulse. A sound whose
spectrum is flat across band ``b`` then has expected reverberant energy in
block ``k`` of ``sum_j S_b(j) gain_b(k - j)``, ``S_b(j)`` being its own
energy in block ``j``. The direct impulse is a gain of 1 in block 0. (Up to
step 2 the gain was the impulse response's energy in each block, which puts
every echo half a block early: 0.4 to 1.4 dB too little energy in the tail
for RT60s of 0.8 to 0.2 s.)

Only the natural room is modelled (exponential decay, ecological frequency
profile and onset levels); the atypical rooms appear only in observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from .cochleagram import freq_to_erb

# Regression fits of Traer & McDermott (2016, Eq. S11 and the onset-level
# fits), as shipped with sonore (sonore/data/fit_*.npy, also in the 2017
# archive): at each frequency, log10(band RT60) = a log10(median RT60) + b, and
# onset level [dB] = a log10(median RT60) + b, before removing the median.
FIT_FREQS_HZ = np.array(
    [
        20.000000000000018, 52.53452072806606, 89.32266484496157, 130.9205589365129,
        177.95703860652745, 231.14315458822676, 291.2829217022629, 359.28547315337823,
        436.17880390228174, 523.1253108713707, 621.439364905682, 732.6071801245017,
        858.3092810287386, 1000.4459069994534, 1161.1657382274939, 1342.8983773242708,
        1548.3910776383907, 1780.7502735002947, 2043.4885402078687, 2340.577693647385,
        2676.508832257061, 3056.36022898794, 3485.8740995852363, 3971.5434076967304,
        4520.710019041513, 5141.675688436205, 5843.827557470186, 6637.780059978897,
        7535.535380501079, 8550.664891371327, 9698.51431123361, 10996.435686358136,
        12464.04970162439, 14123.542286527143, 16000.000000000007,
    ]
)
FIT_RT60 = np.array(
    [
        [-0.12454164574428174, -0.40758451806207396],
        [0.9148902293525073, -0.08755363867346294],
        [1.0131189585354163, -0.030778671020215965],
        [1.0694213573511135, 0.011377094712102902],
        [1.0985139015393468, 0.04099871700826336],
        [1.1086909401950698, 0.059008994921429005],
        [1.1083449422630347, 0.07023252645037022],
        [1.116670271138292, 0.08041977324129952],
        [1.1232229437486865, 0.08686232352333526],
        [1.1224371191611873, 0.09006023640626078],
        [1.1156623047477177, 0.09142489180723772],
        [1.1058561672691787, 0.09111601411637878],
        [1.09007410506939, 0.08789729931519732],
        [1.0660173556336983, 0.08019029099414976],
        [1.0400585387747558, 0.06918402021237186],
        [1.020107202777094, 0.058423670252776096],
        [1.0105253122268343, 0.04942513932519645],
        [1.0003674066822605, 0.03898892003305426],
        [0.9828518542969673, 0.025904838791924383],
        [0.9622009717999747, 0.01136123398042603],
        [0.9381557076149112, -0.005963473581982978],
        [0.9113659805700268, -0.026904933269604372],
        [0.8825565120960427, -0.049893817385875186],
        [0.8549946712277564, -0.07330266890609902],
        [0.8348950938987265, -0.1022897703184788],
        [0.7746798007719943, -0.16458162981162075],
        [0.6635433818504952, -0.26243956353558845],
        [0.5893045427041337, -0.3531859729225119],
        [0.5473088722574062, -0.43451923864042097],
        [0.4814735842354519, -0.5299577140349812],
        [0.3840283332753858, -0.6633545362574712],
        [0.29934272930077355, -0.8295566622360822],
        [0.2673046927116262, -0.9302956947794316],
        [0.24813759796969, -0.9320172927117075],
        [0.23409255927876726, -0.917820804924765],
    ]
)
FIT_ONSET_DB = np.array(
    [
        [-8.929523391096096, -68.14275497502757],
        [-8.929523391096096, -56.190141203366224],
        [-8.929523391096096, -51.390866446894705],
        [-8.929523391096096, -49.06188734869524],
        [-8.929523391096096, -47.78279776394398],
        [-8.929523391096096, -46.898585962158776],
        [-8.929523391096096, -46.276320776029856],
        [-8.929523391096096, -45.81790448083072],
        [-8.929523391096096, -45.451368970324665],
        [-8.929523391096096, -45.19968154545155],
        [-8.929523391096096, -45.10286179635085],
        [-8.929523391096096, -45.26243237992839],
        [-8.929523391096096, -45.91411830538582],
        [-8.929523391096096, -47.09433356016085],
        [-8.929523391096096, -48.19315194442546],
        [-8.929523391096096, -48.79287556680199],
        [-8.929523391096096, -49.035699508014794],
        [-8.929523391096096, -49.127344422029076],
        [-8.929523391096096, -49.24596440863902],
        [-8.929523391096096, -49.386145553630776],
        [-8.929523391096096, -49.42354399632921],
        [-8.929523391096096, -49.38202795413957],
        [-8.929523391096096, -49.41245089014797],
        [-8.929523391096096, -49.6683973694879],
        [-8.929523391096096, -51.138336660807354],
        [-8.929523391096096, -54.74711900023305],
        [-8.929523391096096, -58.88334146628265],
        [-8.929523391096096, -61.36786893798176],
        [-8.929523391096096, -62.27219864316856],
        [-8.929523391096096, -62.52111553969047],
        [-8.929523391096096, -62.79351435682375],
        [-8.929523391096096, -63.51831791666765],
        [-8.929523391096096, -64.8404455930351],
        [-8.929523391096096, -66.23498297271576],
        [-8.929523391096096, -66.79378783049634],
    ]
)


def cosine_responses(freqs: np.ndarray, n_bands: int, f_lo: float, f_hi: float) -> np.ndarray:
    """Amplitude responses of a half-cosine ERB filterbank at ``freqs`` [Hz],
    shape ``(len(freqs), n_bands + 2)``: sonore's ``cosine_filterbank`` with its
    lowpass and highpass edges, whose squared responses sum to 1."""
    lo, hi = freq_to_erb(f_lo), freq_to_erb(f_hi)
    spacing = (hi - lo) / (n_bands + 1)
    knots = lo + spacing * np.arange(n_bands + 2)
    position = freq_to_erb(np.asarray(freqs, float))[:, None]
    distance = (position - knots[None, :]) / spacing
    response = np.where(np.abs(distance) < 1, np.cos(np.pi / 2 * np.clip(distance, -1, 1)), 0.0)
    response[:, 0] = np.where(position[:, 0] <= knots[0], 1.0, response[:, 0])
    response[:, -1] = np.where(position[:, 0] >= knots[-1], 1.0, response[:, -1])
    return response


def _interp_weights(freqs: np.ndarray) -> np.ndarray:
    """Matrix ``W`` with ``W @ values_at_fit_freqs = np.interp(freqs, FIT_FREQS_HZ, values)``."""
    eye = np.eye(len(FIT_FREQS_HZ))
    return np.stack([np.interp(freqs, FIT_FREQS_HZ, eye[i]) for i in range(len(FIT_FREQS_HZ))], axis=1)


def band_rt60s(rt60: torch.Tensor | float, freqs: np.ndarray) -> torch.Tensor:
    """RT60 [s] at ``freqs`` for a median RT60 [s]: the ecological profile."""
    rt60 = torch.as_tensor(rt60, dtype=torch.float64)
    weights = torch.from_numpy(_interp_weights(freqs))
    a, b = torch.from_numpy(FIT_RT60[:, 0]), torch.from_numpy(FIT_RT60[:, 1])
    return weights @ 10 ** (a * torch.log10(rt60) + b)


def onset_levels_db(rt60: torch.Tensor | float, freqs: np.ndarray) -> torch.Tensor:
    """Onset level [dB] of each band's decay at ``freqs``, relative to the
    median over the fit frequencies: the ecological profile."""
    rt60 = torch.as_tensor(rt60, dtype=torch.float64)
    weights = torch.from_numpy(_interp_weights(freqs))
    a, b = torch.from_numpy(FIT_ONSET_DB[:, 0]), torch.from_numpy(FIT_ONSET_DB[:, 1])
    levels = a * torch.log10(rt60) + b
    return weights @ (levels - levels.median())


@dataclass(frozen=True)
class RoomGain:
    """Expected block-by-block gain of a ``synth_ir`` room in each analysis band.

    ``drr_db=None`` is the tail alone, scaled as sonore scales it (RMS 1 over
    its length). The room's own filterbank and decay length are ``synth_ir``'s
    defaults; the analysis bank is the likelihood's (design R4), by default
    the source's bank (design R1).
    """

    fs: float = 20000.0
    block: float = 0.010
    n_blocks: int = 120
    drr_db: float | None = 10.0
    n_bands: int = 39
    f_lo: float = 20.0
    f_hi: float = 4000.0
    room_n_bands: int = 32
    room_f_lo: float = 20.0
    room_f_hi: float = 16000.0
    decay_db: float = 60.0
    n_freqs: int = 1 << 14

    @cached_property
    def _room_freqs(self) -> np.ndarray:
        """Centers and edge knots of the room's bands, where sonore evaluates its fits."""
        f_hi = min(self.room_f_hi, self.fs / 2)
        lo, hi = freq_to_erb(self.room_f_lo), freq_to_erb(f_hi)
        knots = lo + (hi - lo) / (self.room_n_bands + 1) * np.arange(self.room_n_bands + 2)
        return 24.7 * 9.265 * np.expm1(knots / 9.265)

    @cached_property
    def _weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Frequency averages over 0..fs/2: ``W[b, k]`` of |G_b|^2 |H_k|^4 / mean |G_b|^2
        (room band k's share of analysis band b, the noise filtered twice) and
        ``a[k]`` of |H_k|^4 (room band k's share of the whole tail)."""
        freqs = np.linspace(0, self.fs / 2, self.n_freqs)
        room_f_hi = min(self.room_f_hi, self.fs / 2)
        room = cosine_responses(freqs, self.room_n_bands, self.room_f_lo, room_f_hi) ** 4
        analysis = cosine_responses(freqs, self.n_bands, self.f_lo, self.f_hi)[:, 1:-1] ** 2
        shares = analysis.T @ room / len(freqs) / analysis.mean(0)[:, None]
        return torch.from_numpy(shares), torch.from_numpy(room.mean(0))

    def _block_transfer(self, rate: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        """``(len(rate), n_blocks)``: for energy spread evenly over one block,
        the share that an energy decay ``exp(-rate t)``, ``0 <= t < duration``,
        moves ``d`` blocks later. That is the decay weighted by a triangle of
        half-width one block centred on ``d`` blocks, not the decay's energy
        within block ``d``: a sound's energy in a block arrives across the
        block, so its echoes do too."""
        width = self.block
        centers = torch.arange(self.n_blocks, dtype=torch.float64) * width
        rate = rate[:, None]

        def weighted(start, stop, at_start, slope):
            # integral over [start, stop] (clipped to the decay) of
            # (at_start + slope (t - start)) exp(-rate t)
            lo = start.clamp(0, None).clamp(None, duration)
            hi = stop.clamp(0, None).clamp(None, duration)
            span = (hi - lo).clamp_min(0)
            level = at_start + slope * (lo - start)
            x = rate * span
            first = -torch.expm1(-x) / rate
            second = (-torch.expm1(-x) - x * torch.exp(-x)) / rate**2
            return torch.exp(-rate * lo) * (level * first + slope * second)

        rising = weighted(centers - width, centers, 0.0, 1 / width)
        falling = weighted(centers, centers + width, 1.0, -1 / width)
        return rising + falling

    def gain(self, rt60: torch.Tensor | float) -> torch.Tensor:
        """Expected gain, shape ``(n_bands, n_blocks)``, for median RT60 ``rt60`` [s]."""
        rt60 = torch.as_tensor(rt60, dtype=torch.float64)
        taus = band_rt60s(rt60, self._room_freqs)
        onset_power = 10 ** (onset_levels_db(rt60, self._room_freqs) / 10)
        duration = self.decay_db / 60 * taus.max()
        rate = 6 * math.log(10) / taus
        block_energy = onset_power[:, None] * self._block_transfer(rate, duration)
        total_energy = (onset_power / rate) * (1 - torch.exp(-rate * duration))
        shares, room_share = self._weights
        tail = shares @ block_energy / (room_share @ total_energy)
        if self.drr_db is None:
            return tail * duration * self.fs
        tail = tail * 10 ** (-self.drr_db / 10)
        direct = torch.zeros_like(tail)
        direct[:, 0] = 1.0
        return tail + direct
