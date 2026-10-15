"""Figure for the preprocessing slide: what happens to the raw data before any stage runs."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import DATA, FIGS
from phantoms import phantom3d


def main():
    ct = np.load(DATA / "CThead.npy")                       # raw 16-bit values
    mr = np.load(DATA / "MRbrain.npy")
    ctn = np.clip(ct / np.percentile(ct, 99.9), 0, 1)
    sag_before = ctn[:, :, 128]                             # 113 x 256, 2 mm slices -> squashed
    t = torch.from_numpy(ctn.astype(np.float32))[None, None]
    sag_after = F.interpolate(t, size=(226, 256, 256), mode="trilinear", align_corners=False)[0, 0, :, :, 128].numpy()
    lo, hi = np.percentile(mr[45:55], (1, 99.5))
    mrn = np.clip((mr[50] - lo) / (hi - lo), 0, 1)
    clean = phantom3d(128)[64]
    noisy = clean + np.random.default_rng(0).normal(0, 0.08, clean.shape)

    panels = [
        (ct[60], f"1. Raw CT slice: 16-bit, {int(ct.min())}–{int(ct.max())}", dict(cmap="gray")),
        (sag_before, "2. Side view before resampling (113 × 2 mm)", dict(cmap="gray", aspect=0.5)),
        (sag_after, "3. After trilinear resampling to 1 mm", dict(cmap="gray", aspect=1)),
        (mr[50], f"4. Raw MRI {int(mr.min())}–{int(mr.max())} on full 16-bit scale", dict(cmap="gray", vmin=0, vmax=65535)),
        (mrn, "5. Percentile normalised to [0, 1]", dict(cmap="gray")),
        (noisy, "6. Phantom + Gaussian noise σ = 0.08", dict(cmap="gray", vmin=-0.1, vmax=1.1)),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    for a, (im, title, kw) in zip(ax.ravel(), panels):
        a.imshow(im, **kw)
        a.set_title(title, fontsize=13)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "preprocess.png", dpi=130, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
