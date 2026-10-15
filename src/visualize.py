"""Stage 5 - 3D visualisation: rotating MIP and direct volume rendering (DVR).

Uses a real head CT (Stanford CThead, 113 x 256 x 256). For every view angle the
volume is resampled (rotated) with trilinear interpolation, then rays are cast
front-to-back: MIP keeps the maximum, DVR does emission-absorption compositing.
Each ray is independent -> parallel. Same PyTorch code on CPU and GPU.
"""
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, DEV, FIGS, save_results, sync


def rotate(vol, deg):
    """vol: [1,1,D,H,W]; rotate about the vertical (H) axis."""
    a = math.radians(deg)
    theta = torch.tensor([[math.cos(a), 0, math.sin(a), 0],
                          [0, 1, 0, 0],
                          [-math.sin(a), 0, math.cos(a), 0]], dtype=vol.dtype, device=vol.device)[None]
    grid = F.affine_grid(theta, vol.shape, align_corners=False)
    return F.grid_sample(vol, grid, mode="bilinear", align_corners=False)


def mip(v):
    return v.amax(dim=2)[0, 0]


def dvr(v, opacity=0.25):
    # emission-absorption along depth: C = sum_i c_i * a_i * prod_{j<i} (1 - a_j)
    # transfer function: soft tissue nearly transparent, bone opaque
    d = v.clamp(0, 1)
    alpha = (opacity * ((d - 0.25) / 0.35).clamp(0, 1)) + 0.004 * (d > 0.08)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha[:, :, :-1]], 2), 2)
    return (d * alpha * trans).sum(2)[0, 0]


def load_volume():
    ct = np.load(DATA / "CThead.npy")                               # [113 axial slices, 256, 256]
    ct = np.clip(ct / np.percentile(ct, 99.9), 0, 1)
    # axial slices are 2 mm apart vs ~1 mm pixels -> stretch z by 2 so the skull is not squashed
    t = torch.from_numpy(np.ascontiguousarray(ct))[None, None]
    t = F.interpolate(t, size=(226, 256, 256), mode="trilinear", align_corners=False)[0, 0]
    # [D = depth along view, H = vertical (z), W = horizontal]
    return np.ascontiguousarray(t.permute(1, 0, 2).numpy())


def render_all(vol, n_views, fn):
    return [fn(rotate(vol, 360 * i / n_views)) for i in range(n_views)]


def main():
    vol = load_volume()
    res = {"volume_shape": list(vol.shape), "views": 36}
    out = {}
    for name, dev in (("cpu", "cpu"), ("gpu", DEV)):
        v = torch.from_numpy(vol).to(dev)[None, None]
        for kind, fn in (("mip", mip), ("dvr", dvr)):
            render_all(v, 2, fn)                                     # warm-up
            if dev != "cpu":
                sync()
            t0 = time.perf_counter()
            imgs = render_all(v, 36, fn)
            if dev != "cpu":
                sync()
            t = time.perf_counter() - t0
            res[f"{kind}_{name}_s"] = t
            res[f"{kind}_{name}_fps"] = 36 / t
            out[(kind, name)] = [im.cpu().numpy() for im in imgs]
            print(kind, name, f"{t:.3f}s", f"{36 / t:.1f} fps")
    res["mip_speedup"] = res["mip_cpu_s"] / res["mip_gpu_s"]
    res["dvr_speedup"] = res["dvr_cpu_s"] / res["dvr_gpu_s"]
    res["max_abs_diff_dvr"] = float(np.max(np.abs(np.array(out[("dvr", "cpu")]) - np.array(out[("dvr", "gpu")]))))
    save_results("render3d", res)

    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for c, i in enumerate((0, 4, 9, 13)):
        ax[0, c].imshow(out[("mip", "gpu")][i], cmap="gray", vmin=0, vmax=np.percentile(out[("mip", "gpu")][i], 99.5) * 1.15)
        ax[1, c].imshow(out[("dvr", "gpu")][i], cmap="gray")
        ax[0, c].set_title(f"MIP  {i * 10} deg", fontsize=12)
        ax[1, c].set_title(f"Volume render  {i * 10} deg", fontsize=12)
        ax[0, c].axis("off"); ax[1, c].axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "render3d.png", dpi=140, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
