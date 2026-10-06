"""One source or two for the mistuned-harmonic stimuli (App. C.6), by marginal likelihood.

For each mistuning, as the paper's enumerative inference does [App. C.6]:

- H1, one harmonic source, from the stimulus's in-tune parameters;
- H2, a harmonic source plus a whistle at the mistuned frequency, from the
  paper's two initializations, keeping the better.

Each hypothesis is fitted to its posterior mode under the priors in
``sonore_inference.priors``, then its log marginal likelihood
``log p(sound | H)`` is estimated by the Laplace approximation and by
importance sampling (``sonore_inference.evidence``). Adding the log prior of
each hypothesis's structure (number and types of sources) gives the log
posterior odds of two sources over one. Positive means two sources.

Everything runs in float64. The number of harmonics is given. By default
the sources' onsets and durations are given and their frequencies and levels
constant; ``--whistle-timing inferred``, ``--trajectories`` and
``--harmonic-timing inferred`` bring in the corresponding parts of BASS's
sources, and ``--cochleagram fft-bass-gain`` BASS's uncalibrated levels.

    python tools/compare_mistuned_harmonic.py --f0 200 --harmonic 3
"""

import argparse
import math
import time

import torch

from sonore_inference.cochleagram import Cochleagram, FFTCochleagram, gaussian_log_likelihood
from sonore_inference.evidence import log_evidence
from sonore_inference.fit import fit
from sonore_inference.priors import (
    F0_TRAJECTORY,
    GRID_STEP,
    HARMONIC_LEVEL_TRAJECTORY,
    WHISTLE_LEVEL_TRAJECTORY,
    log_prior_event_timing,
    log_prior_level,
    log_prior_log_frequency,
    log_prior_spectrum,
    log_prior_structure,
)
from sonore_inference.sources import harmonic_tone, trajectory_event, whistle, whistle_event
from sonore_inference.stimuli import mistuned_harmonic as mh

TOTAL = mh.DURATION + 2 * mh.PADDING
TIMING = dict(fs=mh.FS, onset=mh.PADDING, duration=mh.DURATION, total_duration=TOTAL, ramp=mh.RAMP)
RATES = {
    "log_f0": 1e-3,
    "amp_db": 0.5,
    "spectrum_db": 0.5,
    "log_whistle_freq": 1e-3,
    "whistle_db": 0.5,
    "whistle_onset": 2e-4,
    "whistle_log_duration": 1e-3,
    # --trajectories: means and deviations in ERB number and dB
    "f0_mean": 5e-3,
    "f0_deviation": 2e-3,
    "level_mean": 0.5,
    "level_deviation": 0.1,
    "harmonic_onset": 2e-4,
    "harmonic_log_duration": 1e-3,
    "whistle_freq_mean": 5e-3,
    "whistle_freq_deviation": 2e-3,
    "whistle_level_mean": 0.5,
    "whistle_level_deviation": 0.05,
}
N_GRID = round(TOTAL / GRID_STEP) + 1


def erb_to_hz(erb):
    return 24.7 * 9.265 * torch.expm1(erb / 9.265)


def hz_to_erb(freq):
    return 9.265 * math.log1p(freq / (24.7 * 9.265))


def one_source(params):
    return harmonic_tone(params["log_f0"].exp(), params["amp_db"] + params["spectrum_db"], **TIMING)


def two_sources(params):
    return one_source(params) + whistle(params["log_whistle_freq"].exp(), params["whistle_db"], **TIMING)


def prior_one(params):
    return (
        log_prior_log_frequency(params["log_f0"])
        + log_prior_level(params["amp_db"])
        + log_prior_spectrum(params["spectrum_db"], params["log_f0"].exp())
    )


def prior_two(params):
    return (
        prior_one(params)
        + log_prior_log_frequency(params["log_whistle_freq"])
        + log_prior_level(params["whistle_db"])
    )


def two_sources_timed(params):
    """H2 with the whistle's onset and duration inferred (BASS infers every event's timing)."""
    return one_source(params) + whistle_event(
        params["log_whistle_freq"].exp(),
        params["whistle_db"],
        params["whistle_onset"],
        params["whistle_log_duration"].exp(),
        fs=mh.FS,
        total_duration=TOTAL,
        ramp=mh.RAMP,
    )


def prior_two_timed(params):
    return prior_two(params) + log_prior_event_timing(
        params["whistle_onset"], params["whistle_log_duration"], TOTAL
    )


def harmonic_with_trajectories(params):
    """A harmonic source whose f0 and level follow Gaussian-process trajectories (BASS's harmonic source)."""
    f0 = erb_to_hz(F0_TRAJECTORY.trajectory(params["f0_mean"], params["f0_deviation"]))
    level = HARMONIC_LEVEL_TRAJECTORY.trajectory(params["level_mean"], params["level_deviation"])
    numbers = torch.arange(1, params["spectrum_db"].shape[-1] + 1, dtype=f0.dtype)
    if "harmonic_onset" in params:
        onset, duration = params["harmonic_onset"], params["harmonic_log_duration"].exp()
    else:
        onset, duration = mh.PADDING, mh.DURATION
    return trajectory_event(
        numbers[:, None] * f0,
        level + params["spectrum_db"][:, None],
        onset,
        duration,
        grid_step=GRID_STEP,
        fs=mh.FS,
        total_duration=TOTAL,
        ramp=mh.RAMP,
    )


def prior_harmonic_with_trajectories(params):
    log_prior = (
        F0_TRAJECTORY.log_prior(params["f0_mean"], params["f0_deviation"])
        + HARMONIC_LEVEL_TRAJECTORY.log_prior(params["level_mean"], params["level_deviation"])
        # at the f0 the source actually has, not the mean of its prior, which
        # trades off against the deviations and is not seen by the likelihood
        + log_prior_spectrum(
            params["spectrum_db"],
            erb_to_hz(F0_TRAJECTORY.trajectory(params["f0_mean"], params["f0_deviation"]).mean()),
        )
    )
    if "harmonic_onset" in params:
        log_prior = log_prior + log_prior_event_timing(
            params["harmonic_onset"], params["harmonic_log_duration"], TOTAL
        )
    return log_prior


def two_sources_with_trajectories(params):
    """H2 with every trajectory: the whistle's frequency and level follow Gaussian processes too."""
    freq = erb_to_hz(F0_TRAJECTORY.trajectory(params["whistle_freq_mean"], params["whistle_freq_deviation"]))
    level = WHISTLE_LEVEL_TRAJECTORY.trajectory(
        params["whistle_level_mean"], params["whistle_level_deviation"]
    )
    if "whistle_onset" in params:
        onset, duration = params["whistle_onset"], params["whistle_log_duration"].exp()
    else:
        onset, duration = mh.PADDING, mh.DURATION
    return harmonic_with_trajectories(params) + trajectory_event(
        freq[None],
        level[None],
        onset,
        duration,
        grid_step=GRID_STEP,
        fs=mh.FS,
        total_duration=TOTAL,
        ramp=mh.RAMP,
    )


def prior_two_with_trajectories(params):
    log_prior = (
        prior_harmonic_with_trajectories(params)
        + F0_TRAJECTORY.log_prior(params["whistle_freq_mean"], params["whistle_freq_deviation"])
        + WHISTLE_LEVEL_TRAJECTORY.log_prior(params["whistle_level_mean"], params["whistle_level_deviation"])
    )
    if "whistle_onset" in params:
        log_prior = log_prior + log_prior_event_timing(
            params["whistle_onset"], params["whistle_log_duration"], TOTAL
        )
    return log_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--f0", type=float, default=200.0)
    parser.add_argument("--harmonic", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--cochleagram",
        choices=["gammatone", "fft", "fft-bass-gain"],
        default="fft",
        help="BASS's FFT approximation (default), with BASS's uncalibrated gain, or gammatone filtering",
    )
    parser.add_argument(
        "--whistle-timing",
        choices=["given", "inferred"],
        default="given",
        help="whether H2's whistle has the stimulus's timing or infers its onset and duration",
    )
    parser.add_argument(
        "--trajectories",
        action="store_true",
        help="give f0, whistle frequency and both levels Gaussian-process trajectories, as BASS does",
    )
    parser.add_argument(
        "--harmonic-timing",
        choices=["given", "inferred"],
        default="given",
        help="with --trajectories, whether the harmonic source infers its onset and duration",
    )
    parser.add_argument("--phases", choices=["cosine", "sine"], default="cosine", help="component phases")
    parser.add_argument("--percents", type=float, nargs="+", default=list(mh.MISTUNING_PERCENTS))
    args = parser.parse_args()
    dtype = torch.float64
    timed = args.whistle_timing == "inferred"
    render_one, prior_of_one = one_source, prior_one
    render_two, prior_of_two = (two_sources_timed, prior_two_timed) if timed else (two_sources, prior_two)
    if args.trajectories:
        render_one, prior_of_one = harmonic_with_trajectories, prior_harmonic_with_trajectories
        render_two, prior_of_two = two_sources_with_trajectories, prior_two_with_trajectories
    cochleagram = {
        "gammatone": Cochleagram(),
        "fft": FFTCochleagram(),
        "fft-bass-gain": FFTCochleagram(bass_gain=True),
    }[args.cochleagram]
    n_harmonics = len(mh.harmonic_numbers(args.f0))
    structure_one = log_prior_structure(["harmonic"], TOTAL)
    structure_two = log_prior_structure(["harmonic", "whistle"], TOTAL)
    print(
        f"f0 {args.f0:g} Hz, harmonic {args.harmonic} mistuned; "
        f"{args.steps} Adam steps, {args.samples} samples, "
        f"{args.cochleagram} cochleagram, {args.phases} phases, whistle timing {args.whistle_timing}"
        + (f", trajectories, harmonic timing {args.harmonic_timing}" if args.trajectories else "")
    )
    print(f"log prior of structure: one source {structure_one:.2f}, two sources {structure_two:.2f}")
    print(
        "percent  logZ1_laplace  logZ1_is  ess1  logZ2_laplace  logZ2_is  ess2"
        "  log_odds_laplace  log_odds_is  seconds"
    )
    for percent in args.percents:
        start = time.perf_counter()
        sound = mh.mistuned_harmonic_stimulus(args.f0, args.harmonic, percent, phases=args.phases)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=dtype))

        def joint(render, prior, observed=observed):
            return lambda params: (
                gaussian_log_likelihood(observed, cochleagram(render(params))) + prior(params)
            )

        if args.trajectories:
            base = {
                "f0_mean": torch.tensor(hz_to_erb(args.f0), dtype=dtype),
                "f0_deviation": torch.zeros(N_GRID, dtype=dtype),
                "level_mean": torch.tensor(mh.COMPONENT_LEVEL_DB, dtype=dtype),
                "level_deviation": torch.zeros(N_GRID, dtype=dtype),
                "spectrum_db": torch.zeros(n_harmonics, dtype=dtype),
            }
            if args.harmonic_timing == "inferred":
                base |= {
                    "harmonic_onset": torch.tensor(mh.PADDING, dtype=dtype),
                    "harmonic_log_duration": torch.tensor(math.log(mh.DURATION), dtype=dtype),
                }
        else:
            base = {
                "log_f0": torch.tensor(math.log(args.f0), dtype=dtype),
                "amp_db": torch.tensor(mh.COMPONENT_LEVEL_DB, dtype=dtype),
                "spectrum_db": torch.zeros(n_harmonics, dtype=dtype),
            }
        h1 = fit(
            render_one,
            base,
            observed,
            cochleagram,
            learning_rates=RATES,
            steps=args.steps,
            log_prior=prior_of_one,
        )
        mistuned_freq = args.harmonic * args.f0 + args.f0 * percent / 100
        h2 = None
        for drop_db, whistle_db in [(30.0, 60.0), (6.0, 50.0)]:
            spectrum = base["spectrum_db"].clone()
            spectrum[args.harmonic - 1] -= drop_db
            if args.trajectories:
                init = base | {
                    "spectrum_db": spectrum,
                    "whistle_freq_mean": torch.tensor(hz_to_erb(mistuned_freq), dtype=dtype),
                    "whistle_freq_deviation": torch.zeros(N_GRID, dtype=dtype),
                    "whistle_level_mean": torch.tensor(whistle_db, dtype=dtype),
                    "whistle_level_deviation": torch.zeros(N_GRID, dtype=dtype),
                }
            else:
                init = base | {
                    "spectrum_db": spectrum,
                    "log_whistle_freq": torch.tensor(math.log(mistuned_freq), dtype=dtype),
                    "whistle_db": torch.tensor(whistle_db, dtype=dtype),
                }
            if timed:
                init |= {
                    "whistle_onset": torch.tensor(mh.PADDING, dtype=dtype),
                    "whistle_log_duration": torch.tensor(math.log(mh.DURATION), dtype=dtype),
                }
            result = fit(
                render_two,
                init,
                observed,
                cochleagram,
                learning_rates=RATES,
                steps=args.steps,
                log_prior=prior_of_two,
            )
            if h2 is None or result.log_likelihood > h2.log_likelihood:
                h2 = result
        generator = torch.Generator().manual_seed(0)
        e1 = log_evidence(
            joint(render_one, prior_of_one), h1.params, n_samples=args.samples, generator=generator
        )
        e2 = log_evidence(
            joint(render_two, prior_of_two), h2.params, n_samples=args.samples, generator=generator
        )
        odds_laplace = e2.laplace + structure_two - e1.laplace - structure_one
        odds_is = e2.importance + structure_two - e1.importance - structure_one
        print(
            f"{percent:7g}  {e1.laplace:13.1f}  {e1.importance:8.1f}  {e1.effective_sample_size:4.0f}"
            f"  {e2.laplace:13.1f}  {e2.importance:8.1f}  {e2.effective_sample_size:4.0f}"
            f"  {odds_laplace:16.1f}  {odds_is:11.1f}  {time.perf_counter() - start:7.1f}",
            flush=True,
        )
        if e1.floored_eigenvalues or e2.floored_eigenvalues:
            print(
                "         note: floored Hessian eigenvalues"
                f" H1 {e1.floored_eigenvalues}, H2 {e2.floored_eigenvalues}"
            )


if __name__ == "__main__":
    main()
