"""Stage 2 - MRI reconstruction from k-space (FFT based).

Real T1 brain MRI slices (Stanford MRbrain volume) are turned into simulated
k-space, 75% of the phase-encoding lines are dropped (4x faster scan) and the
image is recovered with an iterative compressed-sensing style reconstruction
(TV regularisation + k-space data consistency). Every iteration does an FFT
and an inverse FFT, so this is FFT-bound -> cuFFT on GPU.

Also benchmarks raw 3D FFT size scaling: NumPy vs PyTorch-CPU vs cuFFT.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from common import DATA, DEV, FIGS, psnr, save_results, timeit


def load_slices():
    # Stanford MRbrain: 109 sagittal T1 slices, 256x256, 16-bit (see data/README)
    v = np.load(DATA / "MRbrain.npy")[45:55]                       # 10 mid-sagittal slices
    lo, hi = np.percentile(v, (1, 99.5))
    return np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float32)  # (10, 256, 256)


def make_mask(n, accel=4, centre_frac=0.08, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.zeros(n, bool)
    c = int(n * centre_frac)
    keep[n // 2 - c // 2: n // 2 + c // 2] = True                  # low frequencies always kept
    extra = n // accel - keep.sum()
    keep[rng.choice(np.where(~keep)[0], extra, replace=False)] = True
    return np.repeat(keep[:, None], n, axis=1).astype(np.float32)  # rows = phase encodes


def fft2c(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2c(k):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def tv_grad(x, eps=1e-3):
    dx = torch.diff(x, dim=-1, append=x[..., -1:])
    dy = torch.diff(x, dim=-2, append=x[..., -1:, :])
    mag = torch.sqrt(dx ** 2 + dy ** 2 + eps)
    nx, ny = dx / mag, dy / mag
    div = (nx - torch.roll(nx, 1, dims=-1)) + (ny - torch.roll(ny, 1, dims=-2))
    return -div


def cs_recon(y, mask, iters=200, lam=0.005):
    """y: undersampled k-space (complex), mask: sampling mask. Same code on CPU or GPU."""
    x = ifft2c(y).real
    for _ in range(iters):
        x = x - lam * tv_grad(x)
        k = fft2c(x.to(torch.complex64))
        k = k * (1 - mask) + y * mask                              # keep measured samples exactly
        x = ifft2c(k).real
    return x


def main():
    res = {}
    img = load_slices()
    S, N, _ = img.shape
    mask = make_mask(N)

    def run(dev):
        x = torch.from_numpy(img).to(dev)
        m = torch.from_numpy(mask).to(dev)
        y = fft2c(x.to(torch.complex64)) * m
        zf = ifft2c(y).real
        rec = cs_recon(y, m)
        return zf.cpu().numpy(), rec.cpu().numpy()

    torch.set_num_threads(torch.get_num_threads())
    t_cpu, _, (zf, rec_cpu) = timeit(lambda: run("cpu"), reps=2)
    t_gpu, _, (_, rec_gpu) = timeit(lambda: run(DEV), reps=5, gpu=True)
    full = img
    res["cs_recon"] = {
        "slices": S, "size": N, "acceleration": 4, "iterations": 200,
        "ffts_per_slice": 400,
        "cpu_torch_ms": t_cpu * 1e3, "gpu_ms": t_gpu * 1e3, "speedup": t_cpu / t_gpu,
        "cpu_threads": torch.get_num_threads(),
        "psnr_zero_filled": float(np.mean([psnr(full[i], zf[i]) for i in range(S)])),
        "psnr_cs_cpu": float(np.mean([psnr(full[i], rec_cpu[i]) for i in range(S)])),
        "psnr_cs_gpu": float(np.mean([psnr(full[i], rec_gpu[i]) for i in range(S)])),
        "max_abs_diff_cpu_gpu": float(np.abs(rec_cpu - rec_gpu).max()),
    }
    print(res["cs_recon"])

    # --- raw 3D FFT scaling -------------------------------------------------------------------
    res["fft3d"] = []
    for n in (64, 128, 256, 512):
        a = (np.random.rand(n, n, n).astype(np.float32) + 0j).astype(np.complex64)
        r = 1 if n == 512 else 3
        t_np, _, _ = timeit(lambda: np.fft.fftn(a), reps=r, warmup=0 if n == 512 else 1)
        ta = torch.from_numpy(a)
        t_tc, _, _ = timeit(lambda: torch.fft.fftn(ta), reps=r)
        tg = ta.to(DEV)
        t_tg, _, _ = timeit(lambda: torch.fft.fftn(tg), reps=10, gpu=True)
        row = {"n": n, "numpy_ms": t_np * 1e3, "torch_cpu_ms": t_tc * 1e3, "cufft_ms": t_tg * 1e3,
               "speedup_vs_numpy": t_np / t_tg, "speedup_vs_torch_cpu": t_tc / t_tg}
        res["fft3d"].append(row)
        print(row)
        del a, ta, tg
        torch.cuda.empty_cache()
    save_results("mri", res)

    # --- figure -------------------------------------------------------------------------------
    i = 5
    k = np.fft.fftshift(np.fft.fft2(img[i]))
    panels = [(img[i], "Fully sampled MRI"), (np.log1p(np.abs(k)), "k-space (log magnitude)"),
              (mask, "4x undersampling mask"), (zf[i], "Zero-filled (blurry/aliased)"),
              (rec_gpu[i], "Iterative recon on GPU")]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im, cmap="gray")
        a.set_title(t, fontsize=13)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "mri_recon.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
