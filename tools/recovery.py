"""Experiment (a): recover RT60 and the source's statistics from sounds the
model could have made (docs/design/source-reverb.md, R7).

Each run is a 400 ms ``sonore.gaussian_spectrogram`` with given band and
time correlation lengths and level sd, on a Gaussian noise carrier, ramped
by 10 ms, in a ``synth_ir`` room of given RT60 at the typical DRR (9.9 dB),
analysed into 39 bands and 10 ms blocks over 1.2 s. Spectrogram, carrier
and room each get their own seed from ``SeedSequence(run)`` (a shared seed
would make them the same noise). The posterior is
:func:`sonore_inference.sourceroom.infer`, at a floor of 60 dB and at the
measured noise sigma and twice it (design R4).

Runs: 10 seeds x RT60 {0.2, 0.4, 0.8} s x three source settings (sonore's
defaults and two corners inside the prior box, a quarter of each log range
in from opposite edges for the correlation lengths: fine and slow with a
small sd, coarse and fast with a large one).

Each run writes one JSON line per sigma: the truth, 5/50/95% posterior
quantiles of each parameter, how many RT60 points were evaluated, the
seconds taken, and an importance-sampling check of the Laplace estimate
at the best RT60 point (ESS out of 128). ``--summary`` prints, per setting
and sigma, how often the truth falls in the 90% interval and the median
interval width (ratio of the 95% to the 5% quantile).

    python tools/recovery.py --out runs.jsonl [--shard 0 --shards 1]
    python tools/recovery.py --summary runs.jsonl
"""

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import sonore as so
import torch
from scipy.signal import fftconvolve

from sonore_inference.blockpower import SIGMA_DB, BlockPower
from sonore_inference.room import TYPICAL_DRR_DB
from sonore_inference.sourceroom import THETA_DEFAULT, Model, importance_check, infer

FS = 20_000.0
RT60S = (0.2, 0.4, 0.8)
QUARTER = 10**0.25
SETTINGS = {
    "default": THETA_DEFAULT,
    "fine-slow-small": (8.78 / QUARTER, 0.154 * QUARTER, 6.0),
    "coarse-fast-large": (8.78 * QUARTER, 0.154 / QUARTER, 24.0),
}
NAMES = ("rt60", "band_correlation_erb", "time_correlation", "sd_db")


def runs():
    """(run index, setting name, RT60, seed) for every run, in a fixed order."""
    out = []
    for name in SETTINGS:
        for rt60 in RT60S:
            for seed in range(10):
                out.append((len(out), name, rt60, seed))
    return out


def observe(power: BlockPower, run: int, theta, rt60: float) -> torch.Tensor:
    env_seed, carrier_seed, ir_seed = (int(s) for s in np.random.SeedSequence(run).generate_state(3))
    a, b, s = theta
    env = so.gaussian_spectrogram(
        power.duration, FS, band_correlation_erb=a, time_correlation=b, sd_db=s, rng=env_seed
    )
    x = env.to_sound(so.gaussian_noise(power.duration, FS, rng=carrier_seed)).ramp(power.ramp)
    ir = so.synth_ir(rt60, FS, drr_db=TYPICAL_DRR_DB, rng=ir_seed)
    return power.analyze(fftconvolve(x.data.ravel(), ir.data.ravel()))


def run_all(args):
    power = BlockPower()
    todo = [r for r in runs() if r[0] % args.shards == args.shard]
    generator = torch.Generator().manual_seed(args.shard)
    with open(args.out, "a") as out:
        for run, name, rt60, seed in todo:
            theta = SETTINGS[name]
            observed = observe(power, run, theta, rt60)
            for sigma in (SIGMA_DB, 2 * SIGMA_DB):
                start = time.time()
                model = Model(observed, power, sigma_db=sigma)
                posterior = infer(model)
                best = max(posterior.points, key=lambda p: p.log_evidence)
                estimate, ess = importance_check(model, best, generator=generator)
                quantiles = np.vstack([posterior.rt60_quantiles(), posterior.theta_quantiles()])
                record = {
                    "run": run,
                    "setting": name,
                    "seed": seed,
                    "sigma_db": sigma,
                    "truth": [rt60, *theta],
                    "quantiles": quantiles.tolist(),
                    "points": len(posterior.points),
                    "seconds": round(time.time() - start, 1),
                    "laplace_at_best": best.laplace_at_mode,
                    "importance_at_best": estimate,
                    "ess": ess,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"run {run} {name} rt60 {rt60} seed {seed} sigma {sigma}: rt60 "
                    + " ".join(f"{q:.3f}" for q in quantiles[0])
                    + f"  ({record['seconds']} s)",
                    flush=True,
                )


def summary(path):
    records = [json.loads(line) for line in open(path)]
    groups = defaultdict(list)
    for r in records:
        groups[(r["setting"], r["truth"][0], r["sigma_db"])].append(r)
    print(f"{len(records)} records; 90% intervals: share containing the truth, median width (q95/q5)")
    print(f"{'setting':18s} {'rt60':>4s} {'sigma':>5s}  n  " + "  ".join(f"{n:>22s}" for n in NAMES))
    for (name, rt60, sigma), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        cells = []
        for i in range(4):
            q = np.array([r["quantiles"][i] for r in group])
            truth = np.array([r["truth"][i] for r in group])
            inside = np.mean((q[:, 0] <= truth) & (truth <= q[:, 2]))
            width = np.median(q[:, 2] / q[:, 0])
            bias = np.median(np.log(q[:, 1] / truth))
            cells.append(f"{inside:4.0%} x{width:5.3f} {math.exp(bias):5.3f}")
        label = f"{name:18s} {rt60:4.1f} {sigma:5.1f} {len(group):2d}  "
        print(label + "  ".join(f"{c:>22s}" for c in cells))
    print("each cell: coverage, interval width, median of posterior median / truth")
    gap = np.array([r["importance_at_best"] - r["laplace_at_best"] for r in records])
    ess = np.array([r["ess"] for r in records])
    print(
        f"importance sampling minus Laplace at the best RT60 [nats]: median {np.median(gap):.1f}, "
        f"range {gap.min():.1f} to {gap.max():.1f}; ESS median {np.median(ess):.1f} of 128"
    )
    print(f"seconds per inference: median {np.median([r['seconds'] for r in records]):.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="recovery.jsonl")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--summary", default=None, help="summarize this JSON-lines file instead")
    args = parser.parse_args()
    if args.summary:
        summary(args.summary)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    run_all(args)


if __name__ == "__main__":
    main()
