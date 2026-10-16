"""Buildup of streaming online: one pass over the longest sequence (docs/design/online.md, step 2).

The stimulus is App. C.11's: ``--triplets`` ABA triplets, A at 500 Hz, B
``--semitones`` higher, after 50 ms of silence (``LEAD``, as
``tools/grammar/compare_aba.py``). Both hypotheses of the paper are run online, each
in one pass with :func:`sonore_infer.online.run_online`:

- ``one``: one whistle source with every tone as its events;
- ``two``: one whistle source with the A tones and one with the B tones
  (born with the first B tone, design O7).

Prints one line per window and hypothesis, then, at the end of each triplet,
the log odds of two streams over one from the online run and, with
``--batch-steps``, from the batch method on the same prefix (O5 (A)): Adam
from the tones (as ``tools/grammar/compare_aba.py``), then Laplace. ``batch@online``
is the batch Laplace estimate at the online run's mode, so the online
total minus it is the error of the window terms alone (zero with an infinite
lag), and it minus ``batch`` is the difference the fits make.

The evidence is taken every ``--evidence-every`` seconds (0.5 by default,
speed-up S2 of docs/design/online-speed.md, the default since 2026-10-16)
and at each triplet's end; the fit still runs every window. A window line is
printed only at the evidence times, with the seconds and steps since the
last. ``--evidence-every 0`` takes it at every window, as the O1 runs did.

    python tools/online/online_buildup.py --semitones 8 --triplets 6 --lag inf 1 --batch-steps 300
"""

import argparse
import math
import time

import torch

from sonore_infer.cochleagram import FFTCochleagram
from sonore_infer.fit import fit
from sonore_infer.online import Tone, WhistleSpec, batch_laplace, prefix_scene, run_online
from sonore_infer.stimuli import buildup as bu

DTYPE = torch.float64
LEAD = 0.05


def hypotheses(semitones: float, triplets: int) -> dict:
    freqs = bu.tone_freqs(semitones)
    onsets = bu.tone_onsets(triplets, LEAD)
    tones = [(Tone(onset, bu.TONE_DURATION, freqs[kind], bu.TONE_LEVEL_DB), kind) for onset, kind in onsets]
    return {
        "one": (WhistleSpec("w0", tuple(tone for tone, _ in tones)),),
        "two": (
            WhistleSpec("w0", tuple(tone for tone, kind in tones if kind == "A")),
            WhistleSpec("w1", tuple(tone for tone, kind in tones if kind == "B")),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semitones", type=float, default=8)
    parser.add_argument("--triplets", type=int, default=6)
    parser.add_argument("--window", type=float, default=0.05, help="s between updates")
    parser.add_argument(
        "--lag", type=float, nargs="*", default=[1.0], help="s; inf for no freezing; none for batch only"
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Adam steps per window (most, with --window-patience)"
    )
    parser.add_argument(
        "--window-patience",
        type=int,
        default=None,
        help="each window stops once it gains less than --window-tolerance over this many steps (O9)",
    )
    parser.add_argument(
        "--window-tolerance", type=float, default=1.0, help="log joint gain, with --window-patience"
    )
    parser.add_argument(
        "--heard", type=float, default=0.02, help="s of a tone heard before its event enters the scene"
    )
    parser.add_argument(
        "--evidence-every",
        type=float,
        default=0.5,
        help="s between evidence times, plus each triplet's end (online-speed S2); 0 for every window",
    )
    parser.add_argument("--batch-steps", type=int, default=0, help="batch reference at each triplet's end")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="batch fits stop once they gain less than 1 over this many steps (--batch-steps is then a cap)",
    )
    args = parser.parse_args()
    cochleagram = FFTCochleagram()
    sound = bu.buildup_stimulus(args.triplets, args.semitones, lead=LEAD)
    waveform = torch.tensor(sound.data[:, 0], dtype=DTYPE)
    specs = hypotheses(args.semitones, args.triplets)
    checkpoints = [round(bu.duration(n, LEAD), 9) for n in range(1, args.triplets + 1)]
    # window ends: multiples of the window, plus each triplet's end, so the odds can be read there
    total = bu.duration(args.triplets, LEAD)
    grid = [round((k + 1) * args.window, 9) for k in range(int(total / args.window + 1e-9))]
    ends = sorted(set(grid) | set(checkpoints))
    evidence_ends = None
    if args.evidence_every:
        n_evidence = int(total / args.evidence_every + 1e-9)
        every = [round((k + 1) * args.evidence_every, 9) for k in range(n_evidence)]
        evidence_ends = sorted(set(every) | set(checkpoints))
    print(
        f"{args.semitones:g} st, {args.triplets} triplets, window {1000 * args.window:g} ms, "
        f"{args.steps} Adam steps per window"
        + (f", evidence every {args.evidence_every:g} s and at each triplet's end" if evidence_ends else "")
        + ", fft-bass-gain cochleagram, all timing inferred"
    )

    batch = {}
    if args.batch_steps:
        print("batch: triplets  hypothesis  laplace  structure  log_joint  seconds  steps")
        for n, end in enumerate(checkpoints, start=1):
            observed = cochleagram(waveform[: round(end * bu.FS)])
            for name, spec in specs.items():
                begin = time.perf_counter()
                scene, init = prefix_scene(spec, end, bu.FS, bu.RAMP)
                result = fit(
                    scene.render,
                    init,
                    observed,
                    cochleagram,
                    learning_rates=scene.learning_rates(),
                    steps=args.batch_steps,
                    log_prior=scene.log_prior,
                    patience=args.patience,
                )
                state = batch_laplace(scene, result.params, observed, cochleagram)
                batch[n, name] = state.value + scene.structure_log_prior()
                print(
                    f"batch: {n}  {name}  {state.value:10.2f}  {scene.structure_log_prior():6.2f}"
                    f"  {state.peak:10.2f}  {time.perf_counter() - begin:6.1f}  {len(result.history) - 1}",
                    flush=True,
                )

    for lag in args.lag:
        print(f"lag {lag:g} s")
        print(
            "window: lag  hypothesis  end  n_free  term  total  structure  floored  log_joint  seconds  steps"
        )
        runs = {}
        for name, spec in specs.items():

            def report(step, name=name, lag=lag):
                w = step.window
                print(
                    f"window: {lag:g}  {name}  {w.end:.3f}  {w.n_free:4d}  {w.term:9.2f}"
                    f"  {w.total:10.2f}  {step.structure_log_prior:6.2f}  {w.floored:2d}"
                    f"  {w.log_joint:10.2f}  {w.seconds:6.2f}  {w.adam_steps}",
                    flush=True,
                )

            runs[name] = {
                round(step.window.end, 9): step
                for step in run_online(
                    waveform,
                    spec,
                    cochleagram,
                    fs=bu.FS,
                    ramp=bu.RAMP,
                    window=args.window,
                    lag=lag,
                    steps=args.steps,
                    patience=args.window_patience,
                    tolerance=args.window_tolerance,
                    heard=args.heard,
                    ends=ends,
                    evidence_ends=evidence_ends,
                    report=report,
                )
            }
        print("odds: lag  triplets  online  batch@online  batch  (log odds two vs one)")
        for n, end in enumerate(checkpoints, start=1):
            observed = cochleagram(waveform[: round(end * bu.FS)])
            at_online = {}
            for name, spec in specs.items():
                scene, _ = prefix_scene(spec, end, bu.FS, bu.RAMP)
                state = batch_laplace(scene, runs[name][end].params, observed, cochleagram)
                at_online[name] = state.value + scene.structure_log_prior()
            online = runs["two"][end].log_posterior - runs["one"][end].log_posterior
            batch_odds = batch[n, "two"] - batch[n, "one"] if batch else math.nan
            print(
                f"odds: {lag:g}  {n}  {online:9.2f}  {at_online['two'] - at_online['one']:9.2f}"
                f"  {batch_odds:9.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
