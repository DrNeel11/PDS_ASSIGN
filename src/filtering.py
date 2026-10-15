"""Stage 3 - 3D denoising with a 5x5x5 Gaussian convolution.

Same 125-tap filter computed four ways:
  CPU   : scipy.ndimage.convolve (direct 3D convolution)
  GPU 1 : our naive CUDA kernel (global-memory reads)
  GPU 2 : our tiled CUDA kernel (shared-memory halo tile)
  GPU 3 : PyTorch conv3d (cuDNN library)
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from common import DATA, DEV, FIGS, psnr, run_kernel, save_results, timeit
from phantoms import phantom3d


def gauss_kernel(r=2, sigma=1.0):
    ax = np.arange(-r, r + 1)
    g = np.exp(-ax ** 2 / (2 * sigma ** 2))
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return (k / k.sum()).astype(np.float32)


def main():
    k = gauss_kernel()
    kfile = DATA / "gauss5.bin"
    k.tofile(kfile)
    kt = torch.from_numpy(k).to(DEV)[None, None]
    rng = np.random.default_rng(0)
    res = {"scaling": []}

    for n in (64, 128, 256):
        clean = phantom3d(n)
        noisy = (clean + rng.normal(0, 0.08, clean.shape)).astype(np.float32)

        t_cpu, _, out_cpu = timeit(lambda: ndimage.convolve(noisy, k, mode="nearest"),
                                   reps=3 if n < 256 else 1)

        fin, fout = DATA / "conv_in.bin", DATA / "conv_out.bin"
        noisy.tofile(fin)
        kr = run_kernel("conv3d", fin, fout, kfile, n, n, n, 10)
        out_gpu = np.fromfile(fout, np.float32).reshape(n, n, n)

        x = torch.from_numpy(noisy).to(DEV)[None, None]
        t_cudnn, _, _ = timeit(lambda: F.conv3d(F.pad(x, (2,) * 6, mode="replicate"), kt),
                               reps=10, gpu=True)

        row = {
            "n": n, "voxels": n ** 3,
            "cpu_scipy_ms": t_cpu * 1e3,
            "gpu_naive_ms": kr["naive_ms"], "gpu_tiled_ms": kr["tiled_ms"], "gpu_cudnn_ms": t_cudnn * 1e3,
            "h2d_ms": kr["h2d_ms"], "d2h_ms": kr["d2h_ms"],
            "speedup_naive": t_cpu * 1e3 / kr["naive_ms"],
            "speedup_tiled": t_cpu * 1e3 / kr["tiled_ms"],
            "speedup_tiled_with_transfer": t_cpu * 1e3 / (kr["tiled_ms"] + kr["h2d_ms"] + kr["d2h_ms"]),
            "tiled_vs_naive": kr["naive_ms"] / kr["tiled_ms"],
            "gvoxels_per_s_tiled": n ** 3 / (kr["tiled_ms"] * 1e-3) / 1e9,
            "psnr_noisy": psnr(clean, noisy), "psnr_denoised": psnr(clean, out_gpu),
            "max_abs_diff_cpu_gpu": float(np.abs(out_cpu - out_gpu).max()),
        }
        res["scaling"].append(row)
        print(row)
        if n == 256:
            demo = (clean[n // 2], noisy[n // 2], out_gpu[n // 2])

    save_results("filter3d", res)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, im, t in zip(ax, demo, ["Clean CT slice", "Noisy (simulated low-dose)", "Denoised on GPU (tiled kernel)"]):
        a.imshow(im, cmap="gray", vmin=-0.1, vmax=1.1)
        a.set_title(t, fontsize=13)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "denoise.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
