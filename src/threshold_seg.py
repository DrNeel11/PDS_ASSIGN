"""Stage 4a - Classical threshold segmentation (Otsu) on the real head CT.

GPU: custom CUDA histogram (global-atomic vs shared-memory version) + label kernel.
CPU: NumPy histogram + the same Otsu search + NumPy labelling.
Two thresholds from ONE histogram: Otsu on all voxels separates air from the head,
Otsu again on the bins above it separates soft tissue from bone.
Also measures how well the best possible single threshold could segment the
synthetic lesions, to show why the U-Net (stage 4b) is needed.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, FIGS, run_kernel, save_results, timeit
from segmentation import make_dataset

BINS = 256


def otsu(hist, lo=0, hi=BINS):
    """Threshold bin maximising between-class variance, searched inside bins [lo, hi)."""
    h = hist[lo:hi].astype(np.float64)
    p = h / h.sum()
    centres = (np.arange(lo, hi) + 0.5) / BINS
    w0 = np.cumsum(p)
    mu0 = np.cumsum(p * centres)
    mu_t = mu0[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * w0 - mu0) ** 2 / (w0 * w1)
    k = int(np.nanargmax(sigma_b[:-1]))
    return (lo + k + 1) / BINS                                       # threshold value in [0,1]


def thresholds_from_hist(hist):
    t1 = otsu(hist)
    t2 = otsu(hist, lo=int(t1 * BINS))
    return t1, t2


def load_ct(scale=1):
    ct = np.load(DATA / "CThead.npy")
    ct = np.clip(ct / np.percentile(ct, 99.9), 0, 1).astype(np.float32)
    if scale > 1:
        t = torch.from_numpy(ct)[None, None]
        ct = F.interpolate(t, scale_factor=scale, mode="trilinear", align_corners=False)[0, 0].numpy()
    return np.ascontiguousarray(ct)


def cpu_pipeline(v):
    hist = np.histogram(v, bins=BINS, range=(0, 1))[0]
    t1, t2 = thresholds_from_hist(hist)
    lab = np.where(v > t2, 2, np.where(v > t1, 1, 0)).astype(np.uint8)
    return hist, (t1, t2), lab


def gpu_pipeline(v, reps=10):
    fin, flab, fh = DATA / "thr_in.bin", DATA / "thr_lab.bin", DATA / "thr_hist.bin"
    v.tofile(fin)
    k1 = run_kernel("threshold", fin, flab, fh, v.size, -1, -1, reps)   # histogram only
    hist = np.fromfile(fh, np.uint32)
    t0 = time.perf_counter()
    t1, t2 = thresholds_from_hist(hist)
    t_otsu = (time.perf_counter() - t0) * 1e3
    k2 = run_kernel("threshold", fin, flab, fh, v.size, t1, t2, reps)
    lab = np.fromfile(flab, np.uint8).reshape(v.shape)
    timing = {"h2d_ms": k2["h2d_ms"], "hist_global_ms": k2["hist_global_ms"], "hist_shared_ms": k2["hist_shared_ms"],
              "otsu_host_ms": t_otsu, "label_ms": k2["label_ms"], "d2h_ms": k2["d2h_ms"],
              "hist_match": k2["hist_match"], "hist_blocks": k2["hist_blocks"], "label_blocks": k2["label_blocks"]}
    timing["gpu_compute_ms"] = timing["hist_shared_ms"] + timing["otsu_host_ms"] + timing["label_ms"]
    timing["gpu_total_ms"] = timing["gpu_compute_ms"] + timing["h2d_ms"] + timing["d2h_ms"]
    return hist, (t1, t2), lab, timing


def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def main():
    res = {"scaling": []}
    for scale in (1, 2):
        v = load_ct(scale)
        t_cpu, _, (h_cpu, (c1, c2), lab_cpu) = timeit(lambda: cpu_pipeline(v), reps=3)
        h_gpu, (g1, g2), lab_gpu, tg = gpu_pipeline(v)
        row = {"shape": list(v.shape), "voxels": int(v.size), "cpu_ms": t_cpu * 1e3, **tg,
               "speedup_compute": t_cpu * 1e3 / tg["gpu_compute_ms"], "speedup_total": t_cpu * 1e3 / tg["gpu_total_ms"],
               "shared_vs_global": tg["hist_global_ms"] / tg["hist_shared_ms"],
               "t1": g1, "t2": g2, "hist_equal_cpu_gpu": bool(np.array_equal(h_cpu, h_gpu)),
               "label_mismatch_voxels": int((lab_cpu != lab_gpu).sum()),
               "frac_background": float((lab_gpu == 0).mean()), "frac_soft": float((lab_gpu == 1).mean()),
               "frac_bone": float((lab_gpu == 2).mean())}
        res["scaling"].append(row)
        print(row)
        if scale == 1:
            demo = (v, h_gpu, g1, g2, lab_gpu)

    # best possible single global threshold on the lesion test set (uses the ground truth!)
    xte, yte = make_dataset(200, 2)
    x, y = xte[:, 0], yte[:, 0].astype(bool)
    best = (0, None, None)
    for t in np.linspace(0, 0.8, 161):
        for above in (True, False):
            d = dice(x > t if above else x < t, y)
            if d > best[0]:
                best = (d, float(t), above)
    res["lesion_best_threshold"] = {"dice": best[0], "threshold": best[1], "above": best[2]}
    print(res["lesion_best_threshold"])
    save_results("threshold", res)

    v, h, t1, t2, lab = demo
    z = 60
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.5, 1]})
    ax[0].imshow(v[z], cmap="gray"); ax[0].set_title("Real head CT slice", fontsize=13); ax[0].axis("off")
    ax[1].bar(np.arange(BINS) / BINS, h, width=1 / BINS, color="0.35")
    ax[1].set_yscale("log"); ax[1].axvline(t1, color="k", ls="--", lw=2); ax[1].axvline(t2, color="k", ls=":", lw=2)
    ax[1].set_title(f"GPU histogram, 256 bins: Otsu t1 = {t1:.2f} (air|tissue), t2 = {t2:.2f} (tissue|bone)", fontsize=12)
    ax[1].set_xlabel("normalised intensity"); ax[1].set_ylabel("voxels (log)")
    ax[2].imshow(lab[z], cmap="gray", vmin=0, vmax=2); ax[2].set_title("Labels: black air / gray tissue / white bone", fontsize=13); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "threshold.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
