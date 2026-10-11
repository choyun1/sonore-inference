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
import collections
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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


class Progress:
    """Prints when each seed and job starts and finishes, a progress bar, and a periodic heartbeat."""

    def __init__(self, todo, done_before, total):
        self.lock = threading.Lock()
        self.todo, self.done_before, self.total = len(todo), done_before, total
        self.per_seed = collections.Counter(seed_of(name) for name, _ in todo)
        self.seed_done = collections.Counter()
        self.started_seeds = set()
        self.running = {}  # name -> start time
        self.stage = {}  # name -> latest status message
        self.finished, self.failed = 0, []
        self.start = time.perf_counter()

    def bar(self):
        done = self.done_before + self.finished
        width = 30
        filled = round(width * done / self.total) if self.total else width
        line = f"[{'#' * filled}{'-' * (width - filled)}] {done}/{self.total} jobs"
        line += f" ({100 * done / self.total:.0f}%)"
        line += f", {len(self.running)} running"
        if self.finished:
            # wall time per finished job so far, times the jobs left
            left = (time.perf_counter() - self.start) / self.finished * (self.todo - self.finished)
            line += (
                f", about {left / 3600:.1f} h left" if left >= 3600 else f", about {left / 60:.0f} min left"
            )
        if self.failed:
            line += f", {len(self.failed)} failed"
        return line

    def job_started(self, name):
        with self.lock:
            seed = seed_of(name)
            if seed not in self.started_seeds:
                self.started_seeds.add(seed)
                print(
                    f"\n========== seed {seed}: starting ({self.per_seed[seed]} jobs) ==========", flush=True
                )
            self.running[name] = time.perf_counter()
            print(f"{clock()}  START  {name}", flush=True)

    def job_status(self, name, message):
        with self.lock:
            self.stage[name] = message
            print(f"{clock()}    {name}: {message}", flush=True)

    def job_finished(self, name, code):
        with self.lock:
            self.stage.pop(name, None)
            seconds = time.perf_counter() - self.running.pop(name)
            self.finished += 1
            if code:
                self.failed.append(name)
            status = "FAILED (see its .partial file)" if code else "done"
            print(f"{clock()}  {status:6s} {name} in {seconds / 60:.1f} min", flush=True)
            print("          " + self.bar(), flush=True)
            seed = seed_of(name)
            self.seed_done[seed] += 1
            if self.seed_done[seed] == self.per_seed[seed]:
                print(f"========== seed {seed}: finished ==========\n", flush=True)

    def heartbeat(self):
        with self.lock:
            now = time.perf_counter()
            print(f"{clock()}  still running:", flush=True)
            for name, began in sorted(self.running.items()):
                stage = self.stage.get(name, "starting")
                print(f"            {name} ({(now - began) / 60:.0f} min): {stage}", flush=True)
            print("          " + self.bar(), flush=True)


def clock():
    return time.strftime("%H:%M")


def seed_of(name):
    return int(name.rsplit("_seed", 1)[1])


def run(name, argv, out_dir, running, progress):
    target = out_dir / f"{name}.txt"
    partial = target.with_suffix(".partial")
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    progress.job_started(name)
    errors = []
    with open(partial, "w") as handle:
        process = subprocess.Popen(
            [sys.executable, str(TOOLS / argv[0]), *argv[1:]],
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        running.add(process)
        # the tools write "status: ..." progress lines to stderr; anything else there is kept for the file
        for line in process.stderr:
            if line.startswith("status: "):
                progress.job_status(name, line[len("status: ") :].strip())
            else:
                errors.append(line)
        code = process.wait()
        running.discard(process)
    if errors:
        with open(partial, "a") as handle:
            handle.writelines(errors)
    if code == 0:
        partial.replace(target)
    progress.job_finished(name, code)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output folder (created; reused to resume)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel jobs")
    parser.add_argument("--only", choices=["a", "b"], help="run only milestone (a) or (b)")
    parser.add_argument("--heartbeat", type=float, default=5.0, help="minutes between 'still running' lines")
    parser.add_argument("--list", action="store_true", help="print the jobs and their state, then stop")
    args = parser.parse_args()
    for sub in ("a", "b"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*/*.partial"):
        stale.unlink()
    all_jobs = jobs(args.seeds, args.only)
    todo = [(name, argv) for name, argv in all_jobs if not (args.out / f"{name}.txt").exists()]
    done_before = len(all_jobs) - len(todo)
    print(
        f"{done_before} of {len(all_jobs)} jobs already done, {len(todo)} to run, {args.jobs} at a time;"
        f" results go to {args.out.resolve()}"
    )
    if args.list:
        for name, argv in todo:
            print(name, " ".join(argv))
        return
    print("Ctrl+C stops; run the same command again to resume.")
    progress = Progress(todo, done_before, len(all_jobs))
    running = set()
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    pending = {executor.submit(run, name, argv, args.out, running, progress) for name, argv in todo}
    try:
        while pending:
            done, pending = wait(pending, timeout=60 * args.heartbeat, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
            if not done:
                progress.heartbeat()
    except KeyboardInterrupt:
        print("\nstopping: killing running jobs; finished ones are kept, run the same command to resume")
        for future in pending:
            future.cancel()
        for process in list(running):
            process.kill()
        executor.shutdown(wait=True, cancel_futures=True)
        for stale in args.out.glob("*/*.partial"):
            stale.unlink()
        sys.exit(1)
    executor.shutdown()
    elapsed = time.perf_counter() - progress.start
    failures = f"; failed (see their .partial files): {progress.failed}" if progress.failed else ""
    print(f"all done in {elapsed / 3600:.1f} h{failures}")


if __name__ == "__main__":
    main()
