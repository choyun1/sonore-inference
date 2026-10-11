"""One note or two for milestone (b)'s stimuli, by marginal likelihood (docs/design/milestone-b.md).

For each stimulus, three hypotheses (B2), each fitted from known starting
points (B3) and the best posterior mode kept:

- ``one``: one harmonic source, started at 200 Hz, at the upper note's f0,
  and at 100 Hz (every component of the just fifth is a harmonic of 100 Hz);
  each start has every harmonic up to 2400 Hz, so they differ in size and
  each is its own hypothesis; the best is reported;
- ``two``: two harmonic sources, started at the two notes
  (on a control, one note split into two sources an octave apart that sum
  to it);
- ``whistles``: a harmonic source at 200 Hz plus one whistle for each upper
  component that is not a harmonic of 200 Hz (not fitted at the octave,
  where there are none, nor on the controls).

Every start takes its levels and timing from the stimulus: each harmonic's
level is that of the stimulus component at its frequency (coinciding
components add in phase), or 30 dB below the note's level where the
stimulus has none. Prints one line per fit and, per stimulus, the log
posterior odds of ``two`` and ``whistles`` over ``one`` (positive favours the
alternative), by the Laplace and the importance-sampling estimates.

    python tools/compare_two_notes.py --interval octave --asynchronies 0 0.02
    python tools/compare_two_notes.py --controls
"""

import argparse
import math
import time

import torch

from sonore_inference.cochleagram import FFTCochleagram
from sonore_inference.scene import Harmonic, Scene, SceneTiming, Whistle, evaluate
from sonore_inference.stimuli import two_notes as tn

TIMING = SceneTiming(fs=tn.FS, total_duration=tn.DURATION + 2 * tn.PADDING, ramp=tn.RAMP)
ABSENT_DB = 30.0
DTYPE = torch.float64


def notes(condition):
    """The stimulus's notes as (f0, falling, onset, duration)."""
    if condition["interval"] == "control":
        return [(condition["f0"], False, tn.PADDING, tn.DURATION)]
    interval, asynchrony = condition["interval"], condition["asynchrony"]
    upper_duration = round((tn.DURATION - asynchrony) * tn.FS) / tn.FS
    return [
        (tn.LOWER_F0, False, tn.PADDING, tn.DURATION),
        (tn.UPPER_F0S[interval], interval in tn.DIFFERENT_SPECTRA, tn.PADDING + asynchrony, upper_duration),
    ]


def components(condition):
    """Steady-state level [dB] of every component, by frequency rounded to 0.01 Hz."""
    amplitudes = {}
    for f0, falling, _, _ in notes(condition):
        for number, level in zip(tn.harmonic_numbers(f0), tn.harmonic_levels_db(f0, falling), strict=True):
            key = round(number * f0, 2)
            amplitudes[key] = amplitudes.get(key, 0.0) + 10 ** (level / 20)
    return {freq: 20 * math.log10(amplitude) for freq, amplitude in amplitudes.items()}


def harmonic_start(source, f0, onset, duration, levels):
    """A harmonic source at ``f0`` whose harmonics take the stimulus's levels."""
    found = [levels.get(round(n * f0, 2)) for n in tn.harmonic_numbers(f0)]
    present = [level for level in found if level is not None]
    level_mean = tn.COMPONENT_LEVEL_DB if not present else max(present)
    spectrum = torch.tensor(
        [(level - level_mean) if level is not None else -ABSENT_DB for level in found], dtype=DTYPE
    )
    return source.initial(
        TIMING, f0=f0, level_db=level_mean, onset=onset, duration=duration, spectrum_db=spectrum, dtype=DTYPE
    )


def hypotheses(condition, infer_kernel=False):
    """(label, scene, starts) for every hypothesis fitted to this stimulus; with
    ``infer_kernel``, the sources infer their GPs' sigma and lengthscale."""
    levels = components(condition)
    note_list = notes(condition)
    lower_f0, _, lower_onset, lower_duration = note_list[0]
    out = []
    one_f0s = sorted({f0 for f0, *_ in note_list} | {100.0}, reverse=True)
    for f0 in one_f0s:
        source = Harmonic("h0", len(tn.harmonic_numbers(f0)), infer_kernel=infer_kernel)
        start = harmonic_start(source, f0, lower_onset, lower_duration, levels)
        out.append((f"one@{f0:g}", Scene((source,), TIMING), [start]))
    if len(note_list) == 1:
        # the note split into two sources an octave apart, which sum to it: the
        # lower with its even harmonics 6 dB down, the upper at those harmonics, 6 dB down
        half = {freq: level - 20 * math.log10(2) for freq, level in levels.items()}
        split = {
            freq: (half[freq] if round(freq / lower_f0) % 2 == 0 else level) for freq, level in levels.items()
        }
        lower = Harmonic("h0", len(tn.harmonic_numbers(lower_f0)), infer_kernel=infer_kernel)
        upper = Harmonic("h1", len(tn.harmonic_numbers(2 * lower_f0)), infer_kernel=infer_kernel)
        start = harmonic_start(lower, lower_f0, lower_onset, lower_duration, split) | harmonic_start(
            upper, 2 * lower_f0, lower_onset, lower_duration, half
        )
        out.append(("two", Scene((lower, upper), TIMING), [start]))
        return out
    upper_f0, falling, upper_onset, upper_duration = note_list[1]
    lower = Harmonic("h0", len(tn.harmonic_numbers(lower_f0)), infer_kernel=infer_kernel)
    upper = Harmonic("h1", len(tn.harmonic_numbers(upper_f0)), infer_kernel=infer_kernel)
    upper_levels = dict(
        zip(
            [round(n * upper_f0, 2) for n in tn.harmonic_numbers(upper_f0)],
            tn.harmonic_levels_db(upper_f0, falling),
            strict=True,
        )
    )
    lower_levels = {round(n * lower_f0, 2): tn.COMPONENT_LEVEL_DB for n in tn.harmonic_numbers(lower_f0)}
    start = harmonic_start(lower, lower_f0, lower_onset, lower_duration, lower_levels) | harmonic_start(
        upper, upper_f0, upper_onset, upper_duration, upper_levels
    )
    out.append(("two", Scene((lower, upper), TIMING), [start]))
    unshared = tn.unshared_upper_harmonics(condition["interval"])
    if len(unshared):
        whistles = [Whistle(f"w{index}", infer_kernel=infer_kernel) for index in range(len(unshared))]
        start = harmonic_start(lower, lower_f0, lower_onset, lower_duration, levels)
        for whistle, number in zip(whistles, unshared, strict=True):
            freq = number * upper_f0
            start |= whistle.initial(
                TIMING,
                freq=freq,
                level_db=levels[round(freq, 2)],
                onset=upper_onset,
                duration=upper_duration,
                dtype=DTYPE,
            )
        out.append(("whistles", Scene((lower, *whistles), TIMING), [start]))
    return out


def label(condition):
    if condition["interval"] == "control":
        return f"control {condition['f0']:.1f} Hz"
    return f"{condition['interval']} {1000 * condition['asynchrony']:g} ms"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", choices=list(tn.UPPER_F0S))
    parser.add_argument(
        "--asynchronies", type=float, nargs="+", default=list(tn.ASYNCHRONIES), help="seconds"
    )
    parser.add_argument(
        "--controls",
        type=float,
        nargs="*",
        help="fit one-note controls instead (these f0s [Hz], default all four)",
    )
    parser.add_argument("--only", nargs="+", help="fit only these hypotheses (e.g. one@200 two)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--variational-steps",
        type=int,
        default=0,
        help="also estimate the evidence by variational inference with this many steps (extra columns)",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the evidence's sampling (fits are deterministic)"
    )
    parser.add_argument(
        "--sigma", type=float, default=10.0, help="likelihood SD [dB] (BASS's value by default)"
    )
    parser.add_argument(
        "--infer-kernel",
        action="store_true",
        help="infer each GP's sigma and lengthscale instead of fixing them at the medians (design C3 (B))",
    )
    args = parser.parse_args()
    if args.controls is not None:
        conditions = [{"interval": "control", "f0": float(f0)} for f0 in (args.controls or tn.CONTROL_F0S)]
    elif args.interval:
        conditions = [{"interval": args.interval, "asynchrony": a} for a in args.asynchronies]
    else:
        parser.error("give --interval or --controls")
    cochleagram = FFTCochleagram()
    print(
        f"{args.steps} Adam steps, {args.samples} samples, sigma {args.sigma:g} dB, seed {args.seed}, "
        "fft-bass-gain cochleagram, trajectories, all timing inferred"
        + (", GP sigma and lengthscale inferred" if args.infer_kernel else "")
    )
    print(
        "stimulus  hypothesis  n_params  structure  laplace  importance  ess  floored"
        "  log_joint_at_mode  seconds" + ("  elbo  vi_importance  vi_ess" if args.variational_steps else "")
    )
    for condition in conditions:
        name = label(condition)
        if condition["interval"] == "control":
            sound = tn.single_note_control(condition["f0"])
        else:
            sound = tn.two_notes_stimulus(condition["interval"], condition["asynchrony"])
        observed = cochleagram(torch.tensor(sound.data[:, 0], dtype=DTYPE))
        results = {}
        for hypothesis, scene, starts in hypotheses(condition, args.infer_kernel):
            if args.only and hypothesis not in args.only:
                continue
            begin = time.perf_counter()
            generator = torch.Generator().manual_seed(args.seed)
            result = evaluate(
                scene,
                starts,
                observed,
                cochleagram,
                steps=args.steps,
                n_samples=args.samples,
                generator=generator,
                variational_steps=args.variational_steps,
                sigma=args.sigma,
            )
            results[hypothesis] = result
            e = result.evidence
            n_params = sum(value.numel() for value in starts[0].values())
            print(
                f"{name:34s}  {hypothesis:9s}  {n_params:5d}  {result.structure_log_prior:7.2f}"
                f"  {e.laplace:9.1f}  {e.importance:9.1f}  {e.effective_sample_size:4.0f}"
                f"  {e.floored_eigenvalues:3d}"
                f"  {e.log_joint_at_mode:9.1f}  {time.perf_counter() - begin:7.1f}"
                + (
                    f"  {result.variational.elbo:9.1f}  {result.variational.importance:9.1f}"
                    f"  {result.variational.effective_sample_size:4.0f}"
                    if result.variational
                    else ""
                ),
                flush=True,
            )
        ones = [r for h, r in results.items() if h.startswith("one@")]
        if not ones:
            continue
        estimates = ["laplace", "importance"] + (["elbo", "variational"] if args.variational_steps else [])
        for estimate in estimates:
            best_one = max(getattr(r, f"log_posterior_{estimate}") for r in ones)
            odds = {
                h: getattr(r, f"log_posterior_{estimate}") - best_one
                for h, r in results.items()
                if not h.startswith("one@")
            }
            if odds:
                text = ", ".join(f"{h} {value:.1f}" for h, value in odds.items())
                print(f"{name:34s}  log odds vs best one ({estimate}): {text}", flush=True)


if __name__ == "__main__":
    main()
