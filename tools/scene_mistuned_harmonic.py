"""Milestone (a)'s full model re-run through ``sonore_inference.scene``, as its test.

Does what ``tools/compare_mistuned_harmonic.py --trajectories
--harmonic-timing inferred --whistle-timing inferred`` does (same
initializations, step sizes, parameter order and random draws) and prints
the same table, so the two outputs can be compared line by line.

    python tools/scene_mistuned_harmonic.py --f0 200 --harmonic 3
"""

import argparse
import time

import torch

from sonore_inference.cochleagram import FFTCochleagram
from sonore_inference.scene import Harmonic, Scene, SceneTiming, Whistle, evaluate
from sonore_inference.stimuli import mistuned_harmonic as mh

TIMING = SceneTiming(fs=mh.FS, total_duration=mh.DURATION + 2 * mh.PADDING, ramp=mh.RAMP)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--f0", type=float, default=200.0)
    parser.add_argument("--harmonic", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--percents", type=float, nargs="+", default=list(mh.MISTUNING_PERCENTS))
    args = parser.parse_args()
    dtype = torch.float64
    cochleagram = FFTCochleagram()
    harmonic = Harmonic("h0", len(mh.harmonic_numbers(args.f0)))
    whistle = Whistle("w0")
    one = Scene((harmonic,), TIMING)
    two = Scene((harmonic, whistle), TIMING)
    timing = dict(onset=mh.PADDING, duration=mh.DURATION, dtype=dtype)
    print(
        f"f0 {args.f0:g} Hz, harmonic {args.harmonic} mistuned; "
        f"{args.steps} Adam steps, {args.samples} samples, fft-bass-gain cochleagram, cosine phases, "
        "whistle timing inferred, trajectories, harmonic timing inferred (through sonore_inference.scene)"
    )
    print(
        f"log prior of structure: one source {one.structure_log_prior():.2f}, "
        f"two sources {two.structure_log_prior():.2f}"
    )
    print(
        "percent  logZ1_laplace  logZ1_is  ess1  logZ2_laplace  logZ2_is  ess2"
        "  log_odds_laplace  log_odds_is  seconds"
    )
    for percent in args.percents:
        start = time.perf_counter()
        sound = mh.mistuned_harmonic_stimulus(args.f0, args.harmonic, percent)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=dtype))
        base = harmonic.initial(TIMING, f0=args.f0, level_db=mh.COMPONENT_LEVEL_DB, **timing)
        mistuned_freq = args.harmonic * args.f0 + args.f0 * percent / 100
        inits = []
        for drop_db, whistle_db in [(30.0, 60.0), (6.0, 50.0)]:
            spectrum = base["h0.spectrum_db"].clone()
            spectrum[args.harmonic - 1] -= drop_db
            inits.append(
                base
                | {"h0.spectrum_db": spectrum}
                | whistle.initial(TIMING, freq=mistuned_freq, level_db=whistle_db, **timing)
            )
        generator = torch.Generator().manual_seed(0)
        r1 = evaluate(
            one, [base], observed, cochleagram, steps=args.steps, n_samples=args.samples, generator=generator
        )
        r2 = evaluate(
            two, inits, observed, cochleagram, steps=args.steps, n_samples=args.samples, generator=generator
        )
        e1, e2 = r1.evidence, r2.evidence
        odds_laplace = r2.log_posterior_laplace - r1.log_posterior_laplace
        odds_is = r2.log_posterior_importance - r1.log_posterior_importance
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
