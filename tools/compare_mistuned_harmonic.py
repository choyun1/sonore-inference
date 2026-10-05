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

Everything runs in float64. The number of harmonics, the onset and the
duration are given, not inferred.

    python tools/compare_mistuned_harmonic.py --f0 200 --harmonic 3
"""

import argparse
import math
import time

import torch

from sonore_inference.cochleagram import Cochleagram, gaussian_log_likelihood
from sonore_inference.evidence import log_evidence
from sonore_inference.fit import fit
from sonore_inference.priors import (
    log_prior_level,
    log_prior_log_frequency,
    log_prior_spectrum,
    log_prior_structure,
)
from sonore_inference.sources import harmonic_tone, whistle
from sonore_inference.stimuli import mistuned_harmonic as mh

TOTAL = mh.DURATION + 2 * mh.PADDING
TIMING = dict(fs=mh.FS, onset=mh.PADDING, duration=mh.DURATION, total_duration=TOTAL, ramp=mh.RAMP)
RATES = {"log_f0": 1e-3, "amp_db": 0.5, "spectrum_db": 0.5, "log_whistle_freq": 1e-3, "whistle_db": 0.5}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--f0", type=float, default=200.0)
    parser.add_argument("--harmonic", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--percents", type=float, nargs="+", default=list(mh.MISTUNING_PERCENTS))
    args = parser.parse_args()
    dtype = torch.float64
    cochleagram = Cochleagram()
    n_harmonics = len(mh.harmonic_numbers(args.f0))
    structure_one = log_prior_structure(["harmonic"], TOTAL)
    structure_two = log_prior_structure(["harmonic", "whistle"], TOTAL)
    print(
        f"f0 {args.f0:g} Hz, harmonic {args.harmonic} mistuned; "
        f"{args.steps} Adam steps, {args.samples} samples"
    )
    print(f"log prior of structure: one source {structure_one:.2f}, two sources {structure_two:.2f}")
    print(
        "percent  logZ1_laplace  logZ1_is  ess1  logZ2_laplace  logZ2_is  ess2"
        "  log_odds_laplace  log_odds_is  seconds"
    )
    for percent in args.percents:
        start = time.perf_counter()
        sound = mh.mistuned_harmonic_stimulus(args.f0, args.harmonic, percent)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=dtype))

        def joint(render, prior, observed=observed):
            return lambda params: (
                gaussian_log_likelihood(observed, cochleagram(render(params))) + prior(params)
            )

        base = {
            "log_f0": torch.tensor(math.log(args.f0), dtype=dtype),
            "amp_db": torch.tensor(mh.COMPONENT_LEVEL_DB, dtype=dtype),
            "spectrum_db": torch.zeros(n_harmonics, dtype=dtype),
        }
        h1 = fit(
            one_source,
            base,
            observed,
            cochleagram,
            learning_rates=RATES,
            steps=args.steps,
            log_prior=prior_one,
        )
        mistuned_freq = args.harmonic * args.f0 + args.f0 * percent / 100
        h2 = None
        for drop_db, whistle_db in [(30.0, 60.0), (6.0, 50.0)]:
            spectrum = base["spectrum_db"].clone()
            spectrum[args.harmonic - 1] -= drop_db
            init = base | {
                "spectrum_db": spectrum,
                "log_whistle_freq": torch.tensor(math.log(mistuned_freq), dtype=dtype),
                "whistle_db": torch.tensor(whistle_db, dtype=dtype),
            }
            result = fit(
                two_sources,
                init,
                observed,
                cochleagram,
                learning_rates=RATES,
                steps=args.steps,
                log_prior=prior_two,
            )
            if h2 is None or result.log_likelihood > h2.log_likelihood:
                h2 = result
        generator = torch.Generator().manual_seed(0)
        e1 = log_evidence(
            joint(one_source, prior_one), h1.params, n_samples=args.samples, generator=generator
        )
        e2 = log_evidence(
            joint(two_sources, prior_two), h2.params, n_samples=args.samples, generator=generator
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
