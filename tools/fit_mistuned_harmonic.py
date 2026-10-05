"""Fit one- and two-source hypotheses to the mistuned-harmonic stimuli (App. C.6).

For each mistuning of one harmonic, as the paper's enumerative inference
does [App. C.6]:

- H1, one harmonic source, initialized at the stimulus's in-tune parameters;
- H2, a harmonic source plus a whistle at the mistuned frequency, from the
  paper's two initializations (the matching harmonic 30 dB down and the
  whistle at 60 dB; or 6 dB down and 50 dB), keeping the better fit.

Prints each hypothesis's best log likelihood relative to a perfect fit
(``-0.5 * sum(residual**2) / sigma**2``, so 0 is the best possible), their
difference, and H2's fitted whistle. These are maximum-likelihood fits without priors: H2 has two
more parameters than H1, so the difference is not evidence for two sources
on its own (see ``sonore_inference.fit``). The number of harmonics, the
onset and the duration are given, not inferred.

    python tools/fit_mistuned_harmonic.py --f0 200 --harmonic 3
"""

import argparse
import math
import time

import torch

from sonore_inference.cochleagram import Cochleagram, gaussian_log_likelihood
from sonore_inference.fit import fit
from sonore_inference.sources import harmonic_tone, whistle
from sonore_inference.stimuli import mistuned_harmonic as mh

TIMING = dict(
    fs=mh.FS,
    onset=mh.PADDING,
    duration=mh.DURATION,
    total_duration=mh.DURATION + 2 * mh.PADDING,
    ramp=mh.RAMP,
)
RATES = {"log_f0": 1e-3, "levels_db": 0.5, "log_whistle_freq": 1e-3, "whistle_db": 0.5}


def one_source(params):
    return harmonic_tone(params["log_f0"].exp(), params["levels_db"], **TIMING)


def two_sources(params):
    return one_source(params) + whistle(params["log_whistle_freq"].exp(), params["whistle_db"], **TIMING)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--f0", type=float, default=200.0)
    parser.add_argument("--harmonic", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--percents", type=float, nargs="+", default=list(mh.MISTUNING_PERCENTS))
    args = parser.parse_args()
    dtype = torch.float32
    cochleagram = Cochleagram()
    n_harmonics = len(mh.harmonic_numbers(args.f0))
    log_f0 = torch.tensor(math.log(args.f0), dtype=dtype)
    print(f"f0 {args.f0:g} Hz, harmonic {args.harmonic} mistuned, {args.steps} Adam steps per fit, {dtype}")
    print("percent  logL_H1  logL_H2  H2-H1  whistle_Hz  whistle_dB  seconds")
    for percent in args.percents:
        start = time.perf_counter()
        sound = mh.mistuned_harmonic_stimulus(args.f0, args.harmonic, percent)
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=dtype))
        perfect = gaussian_log_likelihood(observed, observed).item()
        in_tune = torch.full((n_harmonics,), mh.COMPONENT_LEVEL_DB, dtype=dtype)
        h1 = fit(
            one_source,
            {"log_f0": log_f0, "levels_db": in_tune},
            observed,
            cochleagram,
            learning_rates=RATES,
            steps=args.steps,
        )
        mistuned_freq = args.harmonic * args.f0 + args.f0 * percent / 100
        h2_fits = []
        for drop_db, whistle_db in [(30.0, 60.0), (6.0, 50.0)]:
            levels = in_tune.clone()
            levels[args.harmonic - 1] -= drop_db
            init = {
                "log_f0": log_f0,
                "levels_db": levels,
                "log_whistle_freq": torch.tensor(math.log(mistuned_freq), dtype=dtype),
                "whistle_db": torch.tensor(whistle_db, dtype=dtype),
            }
            h2_fits.append(
                fit(two_sources, init, observed, cochleagram, learning_rates=RATES, steps=args.steps)
            )
        h2 = max(h2_fits, key=lambda result: result.log_likelihood)
        h1_relative, h2_relative = h1.log_likelihood - perfect, h2.log_likelihood - perfect
        print(
            f"{percent:7g}  {h1_relative:7.1f}  {h2_relative:7.1f}  {h2_relative - h1_relative:5.1f}"
            f"  {h2.params['log_whistle_freq'].exp().item():10.1f}  {h2.params['whistle_db'].item():10.1f}"
            f"  {time.perf_counter() - start:7.1f}"
        )


if __name__ == "__main__":
    main()
