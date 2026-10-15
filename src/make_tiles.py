"""Square, label-free image tiles used on the slides (pipeline, datasets, title)."""
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DATA, FIGS, ROOT
from mri_recon import load_slices
from phantoms import phantom2d, phantom3d
from segmentation import UNet, make_dataset
from skimage.transform import radon

TILES = FIGS / "tiles"
TILES.mkdir(exist_ok=True)


def save(name, img, cmap="gray", overlay=None, vmin=None, vmax=None):
    fig = plt.figure(figsize=(4, 4), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if overlay is not None:
        for m, c, ls, lw in overlay:
            ax.contour(m, [0.5], colors=c, linestyles=ls, linewidths=lw)
    ax.axis("off")
    fig.savefig(TILES / f"{name}.png", facecolor="black")
    plt.close(fig)


def main():
    p = phantom2d(256)
    save("phantom", p)
    save("sinogram", radon(p, np.linspace(0, 180, 256, endpoint=False), circle=True))
    ct = np.load(DATA / "ct_volume_recon.npy")
    save("ct_recon", ct[64], vmin=0, vmax=1)
    mri = load_slices()
    save("mri", mri[5])
    save("kspace", np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(mri[5])))))
    clean = phantom3d(128)[64]
    rng = np.random.default_rng(0)
    save("noisy", clean + rng.normal(0, 0.08, clean.shape), vmin=-0.1, vmax=1.1)
    from scipy import ndimage
    save("denoised", ndimage.gaussian_filter(clean + rng.normal(0, 0.08, clean.shape), 1.0), vmin=-0.1, vmax=1.1)

    x, y = make_dataset(8, 2)
    m = UNet()
    m.load_state_dict(torch.load(ROOT / "results" / "unet.pt", map_location="cpu"))
    m.eval()
    with torch.no_grad():
        pr = (m(torch.from_numpy(x)) > 0).float().numpy()
    save("lesion_input", x[0, 0])
    save("seg_overlay", x[0, 0], overlay=[(y[0, 0], "white", "solid", 2.5), (pr[0, 0], "black", "dashed", 1.5)])

    # volume render tile from stage 5 code
    from visualize import dvr, load_volume, rotate
    vol = torch.from_numpy(load_volume())[None, None]
    save("render", dvr(rotate(vol, 30)).numpy()[10:236, 15:241])
    save("ct_head", np.load(DATA / "CThead.npy")[60], vmin=0, vmax=2200)
    import json
    thr = json.loads((ROOT / "results" / "results.json").read_text())["threshold"]["scaling"][0]
    from threshold_seg import load_ct
    v = load_ct()[60]
    save("ct_labels", np.where(v > thr["t2"], 2, np.where(v > thr["t1"], 1, 0)), vmin=0, vmax=2)
    save("ct_norm", v)
    # black-and-white version of the segmentation figure (same model, same test set)
    from segmentation import save_figure
    xte, yte = make_dataset(200, 2)
    with torch.no_grad():
        save_figure(xte, yte, (m(torch.from_numpy(xte)) > 0).float().numpy())
    print("tiles written to", TILES)


if __name__ == "__main__":
    main()
