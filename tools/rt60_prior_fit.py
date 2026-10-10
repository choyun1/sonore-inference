"""RT60 prior and typical DRR from Traer & McDermott's IR survey (docs/design/source-reverb.md, R2, R8).

The survey's 271 impulse responses are not redistributed. Download them
from https://mcdermottlab.mit.edu/Reverb/IR_Survey.html, unzip, and pass the
folder; every .wav below it is read.

For each IR:

- the direct sound is the 5 ms around its largest sample (1 ms before,
  4 ms after); DRR is its energy over the energy of everything after it;
- the reverberation is everything after it, measured with
  ``sonore.measure_rt60`` (30 ERB bands, 50 Hz to 8 kHz, T20 x 3); the IR's
  RT60 is the median over bands that decay through the fit range, as
  ``synth_ir``'s RT60 is a median over frequency.

This measurement reads ``synth_ir``'s RT60 5-10% high (its RT60 is a
median over 20 Hz-16 kHz on its own profile, and T20 x 3 is not the
model's decay exactly). So each measured RT60 is also mapped back to the
``synth_ir`` RT60 that measures the same, by interpolating measurements of
``synth_ir`` rooms from 0.1 to 3.2 s; the prior is fitted to those, since it
is a prior on ``synth_ir``'s parameter.

Printed: how many IRs were read and measured, the lognormal fit to the
RT60s (median and sigma of the natural log, the two numbers the prior
needs) against an exponential, raw and mapped, and the DRR's median and
quartiles.
``--synthetic`` runs the same measurement on ``synth_ir`` rooms of known
RT60 and DRR instead, to show what it recovers.

    python tools/rt60_prior_fit.py PATH/TO/IR_FOLDER [--save rt60s.csv]
    python tools/rt60_prior_fit.py --synthetic
"""

import argparse
import math
from pathlib import Path

import numpy as np
import sonore as so

BEFORE, AFTER = 0.001, 0.004  # direct-sound window around the peak [s]


def measure(ir: so.Sound) -> tuple[float, float, int]:
    """(RT60 [s], DRR [dB], bands measured) of one impulse response."""
    x = np.asarray(ir.mono().data).ravel()
    peak = int(np.argmax(np.abs(x)))
    lo, hi = max(0, peak - round(BEFORE * ir.fs)), peak + round(AFTER * ir.fs)
    direct, tail = np.sum(x[lo:hi] ** 2), np.sum(x[hi:] ** 2)
    _, rt60s = so.measure_rt60(so.Sound(x[hi:], ir.fs))
    finite = rt60s[np.isfinite(rt60s)]
    rt60 = float(np.median(finite)) if finite.size else math.nan
    return rt60, 10 * math.log10(direct / tail), int(finite.size)


def fit(rt60s: np.ndarray) -> None:
    logs = np.log(rt60s)
    mu, sigma = logs.mean(), logs.std()
    n = len(rt60s)
    normal = -np.log(sigma * math.sqrt(2 * math.pi)) - (logs - mu) ** 2 / (2 * sigma**2)
    loglik_lognormal = np.sum(normal - logs)
    mean = rt60s.mean()
    loglik_exponential = np.sum(-np.log(mean) - rt60s / mean)
    print(f"RT60 [s]: n {n}, min {rt60s.min():.3f}, median {np.median(rt60s):.3f}, max {rt60s.max():.3f}")
    print(f"lognormal: median exp(mu) = {math.exp(mu):.4f} s, sigma = {sigma:.4f} (natural log)")
    print(f"AIC lognormal {4 - 2 * loglik_lognormal:.1f}, exponential {2 - 2 * loglik_exponential:.1f}")


def calibration(fs: float = 32000.0, draws: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(measured, true) RT60s of synth_ir rooms, for mapping measured values back."""
    true = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    measured = np.array(
        [np.mean([measure(so.synth_ir(r, fs, drr_db=10.0, rng=s))[0] for s in range(draws)]) for r in true]
    )
    return measured, true


def synthetic() -> None:
    print("synth_ir rooms (fs 32 kHz), true vs measured")
    print("rt60   drr    measured rt60 (mean, sd over 8)   measured drr")
    for rt60 in (0.2, 0.4, 0.8, 1.6):
        for drr in (0.0, 10.0):
            rows = np.array([measure(so.synth_ir(rt60, 32000.0, drr_db=drr, rng=s))[:2] for s in range(8)])
            print(
                f"{rt60:4.1f} {drr:5.1f}    {rows[:, 0].mean():.3f} {rows[:, 0].std():.3f}"
                f"                     {rows[:, 1].mean():5.1f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path)
    parser.add_argument("--save", type=Path, help="write file, RT60, DRR per IR as CSV (stays local)")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        synthetic()
        return
    if args.folder is None:
        parser.error("give the folder of survey IRs, or --synthetic")
    files = sorted(args.folder.rglob("*.wav"))
    rows, failed = [], []
    for path in files:
        try:
            rt60, drr, n_bands = measure(so.load(path))
        except Exception as error:  # noqa: BLE001 - report and keep going
            failed.append(f"{path.name}: {error}")
            continue
        rows.append((path.name, rt60, drr, n_bands))
    print(f"sonore {so.__version__}; {len(files)} .wav files read, {len(failed)} failed")
    for line in failed[:10]:
        print("  failed", line)
    rt60s = np.array([r[1] for r in rows])
    ok = np.isfinite(rt60s)
    print(f"{ok.sum()} with an RT60 (median over a mean of {np.mean([r[3] for r in rows]):.1f} of 30 bands)")
    print("\nas measured:")
    fit(rt60s[ok])
    measured, true = calibration()
    mapped = np.exp(np.interp(np.log(rt60s[ok]), np.log(measured), np.log(true)))
    print("\nmapped to synth_ir's RT60 (measured " + ", ".join(f"{m:.3f}" for m in measured)
          + " for " + ", ".join(f"{t:g}" for t in true) + " s):")
    fit(mapped)
    drrs = np.array([r[2] for r in rows])
    q1, med, q3 = np.percentile(drrs, [25, 50, 75])
    print(f"DRR [dB]: median {med:.1f}, quartiles {q1:.1f} and {q3:.1f}")
    if args.save:
        with args.save.open("w") as f:
            f.write("file,rt60_s,drr_db,bands\n")
            for name, rt60, drr, n_bands in rows:
                f.write(f"{name},{rt60:.4f},{drr:.2f},{n_bands}\n")


if __name__ == "__main__":
    main()
