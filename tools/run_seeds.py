"""Re-run milestones (a) and (b) with more seeds for the evidence estimates, resumably.

Each job is one stimulus at one seed: (a) one mistuning of one condition
(``scene_mistuned_harmonic.py``), (b) one two-note stimulus or control
(``compare_two_notes.py``), both with the variational evidence. Jobs run in
parallel, one CPU thread each. A job's output is written to a temporary file
and renamed when the job succeeds, so finished jobs are the checkpoint: run the
same command again to continue where it stopped, and stop at any time with
Ctrl+C (running jobs are killed and redone next time). Jobs go seed by seed, so
whole seeds finish first.

The fits are deterministic (Adam from fixed starting points); the seed changes
only the sampling in the evidence estimates. Seed 0 is the runs already made.

    python tools/run_seeds.py --out seeds_out             # seeds 1-9, half the CPU's threads
    python tools/run_seeds.py --out seeds_out --list      # print the jobs and stop
    python tools/run_seeds.py --out seeds_out --jobs 6 --only a
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
VARIATIONAL_STEPS = "2000"

# milestone (a): conditions of variational_a, the paper's 7 mistunings
A_CONDITIONS = [(100, 1), (200, 2), (200, 3), (400, 3)]
A_PERCENTS = [0, 5, 10, 15, 20, 25, 30]
# milestone (b): the intervals at ASYNCHRONIES, the octave also at 120-240 ms, and the four controls
B_INTERVALS = ["tritone_different_spectra", "tritone", "just_fifth", "octave"]
B_ASYNCHRONIES = [0, 0.01, 0.02, 0.04, 0.08]
B_OCTAVE_EXTRA = [0.12, 0.16, 0.24]
B_CONTROLS = [200.0, 200.0 * 2**0.5, 300.0, 400.0]


def jobs(seeds, only):
    """(name, argv) for every job, seed by seed."""
    out = []
    for seed in seeds:
        common = ["--variational-steps", VARIATIONAL_STEPS, "--seed", str(seed)]
        if only in (None, "a"):
            for f0, harmonic in A_CONDITIONS:
                for percent in A_PERCENTS:
                    out.append(
                        (
                            f"a/f0-{f0}_h{harmonic}_p{percent}_seed{seed}",
                            ["scene_mistuned_harmonic.py", "--f0", str(f0), "--harmonic", str(harmonic)]
                            + ["--percents", str(percent)]
                            + common,
                        )
                    )
        if only in (None, "b"):
            for interval in B_INTERVALS:
                extra = B_OCTAVE_EXTRA if interval == "octave" else []
                for asynchrony in B_ASYNCHRONIES + extra:
                    out.append(
                        (
                            f"b/{interval}_{round(1000 * asynchrony)}ms_seed{seed}",
                            [
                                "compare_two_notes.py",
                                "--interval",
                                interval,
                                "--asynchronies",
                                str(asynchrony),
                            ]
                            + common,
                        )
                    )
            for f0 in B_CONTROLS:
                out.append(
                    (
                        f"b/control_{f0:.1f}Hz_seed{seed}",
                        ["compare_two_notes.py", "--controls", repr(f0)] + common,
                    )
                )
    return out


def run(name, argv, out_dir, running):
    target = out_dir / f"{name}.txt"
    partial = target.with_suffix(".partial")
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    begin = time.perf_counter()
    with open(partial, "w") as handle:
        process = subprocess.Popen(
            [sys.executable, str(TOOLS / argv[0]), *argv[1:]],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        running.add(process)
        code = process.wait()
        running.discard(process)
    if code == 0:
        partial.replace(target)
    return name, code, time.perf_counter() - begin


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output folder (created; reused to resume)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel jobs")
    parser.add_argument("--only", choices=["a", "b"], help="run only milestone (a) or (b)")
    parser.add_argument("--list", action="store_true", help="print the jobs and their state, then stop")
    args = parser.parse_args()
    for sub in ("a", "b"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*/*.partial"):
        stale.unlink()
    todo = [
        (name, argv) for name, argv in jobs(args.seeds, args.only) if not (args.out / f"{name}.txt").exists()
    ]
    total = len(jobs(args.seeds, args.only))
    print(
        f"{total - len(todo)} of {total} jobs already done, {len(todo)} to run with {args.jobs} in parallel"
    )
    if args.list:
        for name, argv in todo:
            print(name, " ".join(argv))
        return
    running = set()
    finished, elapsed, failed = 0, 0.0, []
    start = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    futures = [executor.submit(run, name, argv, args.out, running) for name, argv in todo]
    try:
        for future in as_completed(futures):
            name, code, seconds = future.result()
            finished += 1
            elapsed = time.perf_counter() - start
            if code:
                failed.append(name)
            remaining = elapsed / finished * (len(todo) - finished)
            print(
                f"[{finished}/{len(todo)}] {name} {'FAILED' if code else 'done'} in {seconds / 60:.1f} min;"
                f" about {remaining / 3600:.1f} h left",
                flush=True,
            )
    except KeyboardInterrupt:
        print("stopping: killing running jobs; finished ones are kept, run the same command to resume")
        for future in futures:
            future.cancel()
        for process in list(running):
            process.kill()
        executor.shutdown(wait=True, cancel_futures=True)
        for stale in args.out.glob("*/*.partial"):
            stale.unlink()
        sys.exit(1)
    executor.shutdown()
    print(
        f"all done in {elapsed / 3600:.1f} h"
        + (f"; failed (see their .partial files): {failed}" if failed else "")
    )


if __name__ == "__main__":
    main()
