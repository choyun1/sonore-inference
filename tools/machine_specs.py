"""Print the specs of this machine that matter for timing inference (design D6).

Reports the operating system, CPU model and logical cores, RAM, Python, and,
if they are installed, torch and any NVIDIA GPU with its memory. Uses only
the standard library, so it also runs before the package is installed; torch
and nvidia-smi are reported when present.

    python tools/machine_specs.py
"""

import os
import platform
import re
import shutil
import subprocess
import sys


def run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model():
    system = platform.system()
    if system == "Windows":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if system == "Darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as file:
            match = re.search(r"^model name\s*:\s*(.+)$", file.read(), re.MULTILINE)
        if match:
            return match.group(1).strip()
    return platform.processor() or platform.machine()


def ram_gb():
    system = platform.system()
    if system == "Windows":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total_physical", "available_physical", "total_page_file",
                             "available_page_file", "total_virtual", "available_virtual",
                             "available_extended_virtual")
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total_physical / 1e9
    if system == "Darwin":
        return int(run(["sysctl", "-n", "hw.memsize"]) or 0) / 1e9
    with open("/proc/meminfo") as file:
        return int(re.search(r"MemTotal:\s*(\d+) kB", file.read()).group(1)) * 1e3 / 1e9


def main():
    print(f"OS:      {platform.platform()}")
    print(f"CPU:     {cpu_model()}, {os.cpu_count()} logical cores")
    print(f"RAM:     {ram_gb():.1f} GB")
    print(f"Python:  {sys.version.split()[0]} ({platform.architecture()[0]})")

    try:
        import torch
    except ImportError:
        print("torch:   not installed")
    else:
        print(f"torch:   {torch.__version__}, {torch.get_num_threads()} CPU threads, "
              f"built for CUDA {torch.version.cuda or 'none (CPU-only build)'}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                print(f"GPU {index}:   {properties.name}, {properties.total_memory / 1e9:.1f} GB, "
                      f"compute capability {properties.major}.{properties.minor}")
        else:
            print("GPU:     torch sees no CUDA device")

    if shutil.which("nvidia-smi"):
        query = "--query-gpu=name,memory.total,driver_version"
        for line in run(["nvidia-smi", query, "--format=csv,noheader"]).splitlines():
            print(f"nvidia-smi: {line}")


if __name__ == "__main__":
    main()
