"""Milestone (a)'s full model re-run through ``sonore_inference.scene``, as its test.

Does what ``tools/compare_mistuned_harmonic.py --trajectories
--harmonic-timing inferred --whistle-timing inferred`` does (same
initializations, step sizes, parameter order and random draws) and prints
the same table, so the two outputs can be compared line by line.

    python tools/scene_mistuned_harmonic.py --f0 200 --harmonic 3
"""

import argparse
import sys
import time

import torch

from sonore_inference.cochleagram import FFTCochleagram
from sonore_inference.scene import Harmonic, Scene, SceneTiming, Whistle, evaluate
from sonore_inference.stimuli import mistuned_harmonic as mh

TIMING = SceneTiming(fs=mh.FS, total_duration=mh.DURATION + 2 * mh.PADDING, ramp=mh.RAMP)


def status(prefix):
    """A ``report`` for ``evaluate`` that writes progress to stderr (read by tools/run_seeds.py)."""
    return lambda message: print(f"status: {prefix}: {message}", file=sys.stderr, flush=True)


def variational_columns(r1, r2):
    v1, v2 = r1.variational, r2.variational
    return (
        f"  {v1.elbo:10.1f}  {v1.importance:11.1f}  {v1.effective_sample_size:7.0f}"
        f"  {v2.elbo:10.1f}  {v2.importance:11.1f}  {v2.effective_sample_size:7.0f}"
        f"  {r2.log_posterior_elbo - r1.log_posterior_elbo:13.1f}"
        f"  {r2.log_posterior_variational - r1.log_posterior_variational:14.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--f0", type=float, default=200.0)
    parser.add_argument("--harmonic", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--variational-steps",
        type=int,
        default=0,
        help="also estimate the evidence by variational inference with this many steps (extra columns)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the evidence's sampling (fits are deterministic)"
    )
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
        f"{args.steps} Adam steps, {args.samples} samples, seed {args.seed}, "
        "fft-bass-gain cochleagram, cosine phases, whistle timing inferred, trajectories, "
        "harmonic timing inferred (through sonore_inference.scene)"
    )
    print(
        f"log prior of structure: one source {one.structure_log_prior():.2f}, "
        f"two sources {two.structure_log_prior():.2f}"
    )
    print(
        "percent  logZ1_laplace  logZ1_is  ess1  logZ2_laplace  logZ2_is  ess2"
        "  log_odds_laplace  log_odds_is  seconds"
        + (
            "  logZ1_elbo  logZ1_vi_is  ess1_vi  logZ2_elbo  logZ2_vi_is  ess2_vi"
            "  log_odds_elbo  log_odds_vi_is"
            if args.variational_steps
            else ""
        )
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
        generator = torch.Generator().manual_seed(args.seed)
        r1 = evaluate(
            one,
            [base],
            observed,
            cochleagram,
            steps=args.steps,
            n_samples=args.samples,
            generator=generator,
            variational_steps=args.variational_steps,
            report=status(f"{percent:g}%, one source"),
        )
        r2 = evaluate(
            two,
            inits,
            observed,
            cochleagram,
            steps=args.steps,
            n_samples=args.samples,
            generator=generator,
            variational_steps=args.variational_steps,
            report=status(f"{percent:g}%, harmonic plus whistle"),
        )
        e1, e2 = r1.evidence, r2.evidence
        odds_laplace = r2.log_posterior_laplace - r1.log_posterior_laplace
        odds_is = r2.log_posterior_importance - r1.log_posterior_importance
        print(
            f"{percent:7g}  {e1.laplace:13.1f}  {e1.importance:8.1f}  {e1.effective_sample_size:4.0f}"
            f"  {e2.laplace:13.1f}  {e2.importance:8.1f}  {e2.effective_sample_size:4.0f}"
            f"  {odds_laplace:16.1f}  {odds_is:11.1f}  {time.perf_counter() - start:7.1f}"
            + (variational_columns(r1, r2) if args.variational_steps else ""),
            flush=True,
        )
        if e1.floored_eigenvalues or e2.floored_eigenvalues:
            print(
                "         note: floored Hessian eigenvalues"
                f" H1 {e1.floored_eigenvalues}, H2 {e2.floored_eigenvalues}"
            )


if __name__ == "__main__":
    main()
