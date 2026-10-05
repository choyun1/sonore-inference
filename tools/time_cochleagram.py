"""Seconds per cochleagram forward and backward pass, for sizing inference (design D6).

Times the default cochleagram (BASS's 64 half-ERB channels, 25 ms frames
every 10 ms) on white noise of each duration, with a backward pass through
the Gaussian likelihood, as one optimization step would need. Prints the
median of ``--repeats`` runs after one warm-up run.

    python tools/time_cochleagram.py                 # CPU
    python tools/time_cochleagram.py --device cuda   # an NVIDIA GPU
"""

import argparse
import platform
import time

import torch

from sonore_inference.cochleagram import Cochleagram, gaussian_log_likelihood


def time_step(cochleagram, n_samples, dtype, device, repeats):
    generator = torch.Generator().manual_seed(0)
    target = cochleagram(1e-3 * torch.randn(n_samples, generator=generator, dtype=dtype).to(device))
    waveform = (1e-3 * torch.randn(n_samples, generator=generator, dtype=dtype)).to(device).requires_grad_()
    times = []
    for _ in range(repeats + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss = -gaussian_log_likelihood(target, cochleagram(waveform))
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        waveform.grad = None
    return sorted(times[1:])[repeats // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--durations", type=float, nargs="+", default=[0.5, 1.0, 3.1])
    args = parser.parse_args()
    device = torch.device(args.device)
    cochleagram = Cochleagram()
    print(f"torch {torch.__version__}, {platform.processor() or platform.machine()}, device {device}")
    print(f"{torch.get_num_threads()} CPU threads; padding {cochleagram.pad_samples} samples at each end")
    print("duration_s  float32_s  float64_s")
    for duration in args.durations:
        n_samples = round(duration * cochleagram.fs)
        seconds = [
            time_step(cochleagram, n_samples, dtype, device, args.repeats)
            for dtype in (torch.float32, torch.float64)
        ]
        print(f"{duration:10.2f}  {seconds[0]:9.3f}  {seconds[1]:9.3f}")


if __name__ == "__main__":
    main()
