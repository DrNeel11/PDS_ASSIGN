"""Stage 1 - CT reconstruction with Filtered Back Projection (FBP).

CPU : NumPy ramp filter + NumPy back-projection (one angle at a time)
GPU : PyTorch cuFFT ramp filter + custom CUDA back-projection kernel
      (one thread per output pixel, 3D grid over all slices at once)
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.transform import iradon, radon

from common import DATA, DEV, FIGS, psnr, run_kernel, save_results, sync, timeit
from phantoms import phantom2d, phantom3d


def sinogram(img, n_angles):
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    return radon(img, theta=theta, circle=True).T.astype(np.float32)      # [A, D]


def ramp(D):
    size = max(64, int(2 ** np.ceil(np.log2(2 * D))))
    return size, (2 * np.abs(np.fft.fftfreq(size))).astype(np.float32)


def filter_cpu(sino):
    A, D = sino.shape[-2:]
    size, h = ramp(D)
    F = np.fft.fft(sino, n=size, axis=-1) * h
    return np.real(np.fft.ifft(F, axis=-1))[..., :D].astype(np.float32)


def filter_gpu(sino_t):
    D = sino_t.shape[-1]
    size, h = ramp(D)
    F = torch.fft.fft(sino_t, n=size, dim=-1) * torch.from_numpy(h).to(DEV)
    return torch.fft.ifft(F, dim=-1).real[..., :D].contiguous()


def backproject_cpu(fsino, N):
    A, D = fsino.shape
    rows, cols = np.mgrid[:N, :N] - N // 2
    det = np.arange(D) - D // 2
    img = np.zeros((N, N), np.float64)
    for a in range(A):
        th = np.pi * a / A
        t = cols * np.cos(th) - rows * np.sin(th)
        img += np.interp(t, det, fsino[a], left=0, right=0)
    return (img * np.pi / (2 * A)).astype(np.float32)


def fbp_cpu(sinos, N):
    f = filter_cpu(sinos)
    return np.stack([backproject_cpu(s, N) for s in f])


def fbp_gpu(sinos, N, reps=5):
    """Returns (volume, timing dict). Filtering in cuFFT, back-projection in our CUDA kernel."""
    S, A, D = sinos.shape
    t0 = time.perf_counter()
    st = torch.from_numpy(sinos).to(DEV)
    sync()
    t_up = time.perf_counter() - t0
    t_filter, _, f = timeit(lambda: filter_gpu(st), reps=reps, gpu=True)
    fin, fout = DATA / "fbp_in.bin", DATA / "fbp_out.bin"
    f.cpu().numpy().tofile(fin)
    k = run_kernel("fbp", fin, fout, S, A, D, N, reps)
    vol = np.fromfile(fout, np.float32).reshape(S, N, N)
    timing = {
        "filter_ms": t_filter * 1e3,
        "h2d_ms": t_up * 1e3 + k["h2d_ms"],
        "kernel_ms": k["kernel_ms"],
        "d2h_ms": k["d2h_ms"],
    }
    timing["total_ms"] = timing["filter_ms"] + timing["h2d_ms"] + timing["kernel_ms"] + timing["d2h_ms"]
    return vol, timing


def main():
    res = {"scaling": [], "volume": {}}
    fbp_gpu(sinogram(phantom2d(64), 64)[None], 64)            # warm-up: CUDA context + cuFFT plans

    # --- 1. accuracy + single-slice scaling -------------------------------------------------
    for N in (128, 256, 512):
        img = phantom2d(N)
        sino = sinogram(img, N)[None]                         # [1, A, D], A = N angles
        t_cpu, _, rec_cpu = timeit(lambda: fbp_cpu(sino, N), reps=3 if N < 512 else 1)
        t_sk, _, _ = timeit(lambda: iradon(sino[0].T, theta=np.linspace(0, 180, N, endpoint=False),
                                           filter_name="ramp", circle=True), reps=1)
        rec_gpu, tg = fbp_gpu(sino, N)
        row = {
            "N": N, "angles": N,
            "cpu_numpy_ms": t_cpu * 1e3, "cpu_skimage_ms": t_sk * 1e3,
            "gpu_total_ms": tg["total_ms"], "gpu_kernel_ms": tg["kernel_ms"],
            "speedup_total": t_cpu * 1e3 / tg["total_ms"],
            "speedup_kernel": t_cpu * 1e3 / (tg["kernel_ms"] + tg["filter_ms"]),
            "psnr_cpu": psnr(img, rec_cpu[0]), "psnr_gpu": psnr(img, rec_gpu[0]),
            "max_abs_diff_cpu_gpu": float(np.abs(rec_cpu - rec_gpu).max()),
        }
        res["scaling"].append(row)
        print(row)
        if N == 256:
            demo = (img, sino[0], rec_cpu[0], rec_gpu[0])

    # --- 2. full CT volume: 128 slices of 256x256, 256 angles each ------------------------------
    S, N = 128, 256
    vol = phantom3d(N, S)
    sinos_path = DATA / f"ct_sinos_{S}x{N}.npy"
    if sinos_path.exists():
        sinos = np.load(sinos_path)
    else:
        sinos = np.stack([sinogram(vol[s], N) for s in range(S)])
        np.save(sinos_path, sinos)
    t0 = time.perf_counter()
    rec_cpu = fbp_cpu(sinos, N)
    t_cpu = time.perf_counter() - t0
    rec_gpu, tg = fbp_gpu(sinos, N)
    res["volume"] = {
        "slices": S, "N": N, "angles": N, "cpu_ms": t_cpu * 1e3, **{f"gpu_{k}": v for k, v in tg.items()},
        "speedup_total": t_cpu * 1e3 / tg["total_ms"],
        "speedup_compute": t_cpu * 1e3 / (tg["kernel_ms"] + tg["filter_ms"]),
        "psnr_gpu": psnr(vol, rec_gpu), "max_abs_diff_cpu_gpu": float(np.abs(rec_cpu - rec_gpu).max()),
        "gpu_threads": S * N * N,
    }
    print(res["volume"])
    np.save(DATA / "ct_volume_recon.npy", rec_gpu)
    save_results("ct_fbp", res)

    # --- figure -------------------------------------------------------------------------------
    img, sino, rc, rg = demo
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, im, t in zip(ax, [img, sino, rc, rg],
                        ["Ground-truth phantom", "Sinogram (256 angles)", "FBP on CPU (NumPy)",
                         "FBP on GPU (CUDA kernel)"]):
        a.imshow(im, cmap="gray", aspect="auto" if im is sino else "equal")
        a.set_title(t, fontsize=13)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "ct_fbp.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
