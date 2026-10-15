"""Shared helpers: paths, timing, CUDA kernel build/run, result storage."""
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
KERNELS = ROOT / "src" / "kernels"
BIN = KERNELS / "bin"
for p in (DATA, RESULTS, FIGS, BIN):
    p.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda")
VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"


def sync():
    torch.cuda.synchronize()


def timeit(fn, reps=5, warmup=1, gpu=False):
    """Run fn several times, return (mean_seconds, std_seconds, last_result)."""
    out = None
    for _ in range(warmup):
        out = fn()
        if gpu:
            sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if gpu:
            sync()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts)), out


def build_kernel(name):
    """Compile src/kernels/<name>.cu with nvcc (needs MSVC env on Windows)."""
    src, exe = KERNELS / f"{name}.cu", BIN / f"{name}.exe"
    if exe.exists() and exe.stat().st_mtime > src.stat().st_mtime:
        return exe
    cmd = f'call "{VCVARS}" >nul && nvcc -O3 -arch=sm_120 -o "{exe}" "{src}"'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not exe.exists():
        raise RuntimeError(f"nvcc failed for {name}:\n{r.stdout}\n{r.stderr}")
    return exe


def run_kernel(name, *args):
    exe = build_kernel(name)
    r = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


def psnr(ref, img):
    ref, img = np.asarray(ref, np.float64), np.asarray(img, np.float64)
    mse = np.mean((ref - img) ** 2)
    rng = ref.max() - ref.min()
    return float(10 * np.log10(rng ** 2 / mse)) if mse > 0 else float("inf")


def save_results(stage, data):
    path = RESULTS / "results.json"
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[stage] = data
    path.write_text(json.dumps(allr, indent=2))
    print(f"[{stage}] saved")


def hardware_info():
    p = torch.cuda.get_device_properties(0)
    cpu = platform.processor()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
        cpu = out.stdout.strip() or cpu
    except Exception:
        pass
    return {
        "gpu": p.name,
        "gpu_mem_gb": round(p.total_memory / 2**30, 1),
        "sm_count": p.multi_processor_count,
        "compute_capability": f"{p.major}.{p.minor}",
        "cpu": cpu,
        "cpu_threads": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
