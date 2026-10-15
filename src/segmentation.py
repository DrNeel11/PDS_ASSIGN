"""Stage 4 - Lesion segmentation with a 2D U-Net (deep learning on GPU).

Data: synthetic CT-like slices (body + organs + 1-3 lesions + noise) with exact
ground-truth lesion masks, so Dice can be measured precisely.
Measures: training speed CPU vs GPU, inference throughput CPU fp32 vs GPU fp32
vs GPU mixed precision (fp16 Tensor Cores), and segmentation accuracy (Dice).
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from common import DEV, FIGS, ROOT, save_results, sync

SZ = 128


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SZ, :SZ] / (SZ - 1) * 2 - 1
    imgs = np.zeros((n, SZ, SZ), np.float32)
    masks = np.zeros((n, SZ, SZ), np.float32)

    def ell(cx, cy, a, b, th):
        c, s = np.cos(th), np.sin(th)
        xr, yr = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c
        return (xr / a) ** 2 + (yr / b) ** 2 <= 1

    for i in range(n):
        body = ell(rng.uniform(-.05, .05), rng.uniform(-.05, .05), rng.uniform(.75, .9), rng.uniform(.6, .8), rng.uniform(-.2, .2))
        im = 0.25 * body
        for _ in range(rng.integers(2, 4)):                       # organs
            o = ell(rng.uniform(-.4, .4), rng.uniform(-.3, .3), rng.uniform(.15, .3), rng.uniform(.1, .25), rng.uniform(0, np.pi)) & body
            im = np.where(o, rng.uniform(.4, .6), im)
        m = np.zeros_like(body)
        for _ in range(rng.integers(1, 4)):                       # lesions
            l = ell(rng.uniform(-.5, .5), rng.uniform(-.4, .4), rng.uniform(.04, .12), rng.uniform(.04, .12), rng.uniform(0, np.pi)) & body
            im = np.where(l, im + rng.choice([-1, 1]) * rng.uniform(.12, .2), im)
            m |= l
        im = ndimage.gaussian_filter(im, 1.0) + rng.normal(0, 0.04, im.shape)
        imgs[i], masks[i] = im, m
    return imgs[:, None], masks[:, None]


class Block(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        ch = 1
        for o in c:
            self.downs.append(Block(ch, o))
            ch = o
        self.bott = Block(c[-1], c[-1] * 2)
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        ch = c[-1] * 2
        for o in reversed(c):
            self.ups.append(nn.ConvTranspose2d(ch, o, 2, stride=2))
            self.dec.append(Block(o * 2, o))
            ch = o
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bott(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


def dice_score(pred, y):
    inter = (pred * y).sum((1, 2, 3))
    return ((2 * inter + 1e-6) / (pred.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1e-6)).mean().item()


def train_step(model, opt, scaler, x, y, amp):
    opt.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.float16 if x.device.type == "cuda" else torch.bfloat16, enabled=amp):
        out = model(x)
        loss = F.binary_cross_entropy_with_logits(out, y) + dice_loss(out.float(), y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


def main():
    torch.manual_seed(0)
    xtr, ytr = make_dataset(2000, 1)
    xte, yte = make_dataset(200, 2)
    res = {"dataset": {"train": 2000, "test": 200, "size": SZ}}
    BS, EPOCHS = 32, 25

    # --- CPU training speed (measured on 10 iterations, then per-epoch extrapolated) -----------
    m_cpu = UNet()
    res["params_millions"] = sum(p.numel() for p in m_cpu.parameters()) / 1e6
    opt = torch.optim.Adam(m_cpu.parameters(), 1e-3)
    sc = torch.amp.GradScaler("cpu", enabled=False)
    xb, yb = torch.from_numpy(xtr[:BS]), torch.from_numpy(ytr[:BS])
    train_step(m_cpu, opt, sc, xb, yb, False)
    t0 = time.perf_counter()
    for _ in range(10):
        train_step(m_cpu, opt, sc, xb, yb, False)
    cpu_iter = (time.perf_counter() - t0) / 10

    # --- GPU training (full run, mixed precision) ---------------------------------------------
    model = UNet().to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    Xtr, Ytr = torch.from_numpy(xtr).to(DEV), torch.from_numpy(ytr).to(DEV)
    Xte, Yte = torch.from_numpy(xte).to(DEV), torch.from_numpy(yte).to(DEV)
    iters = len(xtr) // BS
    history = []
    sync()
    t_train = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr), device=DEV)
        tot = 0.0
        for i in range(iters):
            idx = perm[i * BS:(i + 1) * BS]
            tot += train_step(model, opt, scaler, Xtr[idx], Ytr[idx], True)
        sched.step()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            d = dice_score((model(Xte) > 0).float(), Yte)
        history.append({"epoch": ep + 1, "loss": tot / iters, "test_dice": d})
        print(history[-1])
    sync()
    t_train = time.perf_counter() - t_train
    gpu_iter = t_train / (EPOCHS * iters)
    torch.save(model.state_dict(), ROOT / "results" / "unet.pt")

    res["training"] = {
        "epochs": EPOCHS, "batch": BS, "iters_per_epoch": iters,
        "cpu_s_per_iter": cpu_iter, "gpu_amp_s_per_iter": gpu_iter,
        "cpu_epoch_s": cpu_iter * iters, "gpu_epoch_s": gpu_iter * iters,
        "gpu_total_train_s": t_train, "cpu_total_train_s_est": cpu_iter * iters * EPOCHS,
        "speedup": cpu_iter / gpu_iter, "history": history,
    }

    # --- inference throughput -----------------------------------------------------------------
    m_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    m_cpu.eval()
    model.eval()
    xc = torch.from_numpy(xte)

    def bench(fn, n_img, gpu):
        fn()
        if gpu:
            sync()
        t0 = time.perf_counter()
        r = 3
        for _ in range(r):
            fn()
        if gpu:
            sync()
        return n_img * r / (time.perf_counter() - t0)

    with torch.no_grad():
        ips_cpu = bench(lambda: m_cpu(xc), len(xc), False)
        m32 = UNet().to(DEV)
        m32.load_state_dict(model.state_dict())
        m32.eval()
        ips_gpu32 = bench(lambda: m32(Xte), len(xc), True)
        with torch.autocast("cuda", dtype=torch.float16):
            ips_gpu16 = bench(lambda: model(Xte), len(xc), True)
            pred = (model(Xte) > 0).float()
    res["inference"] = {
        "cpu_fp32_img_per_s": ips_cpu, "gpu_fp32_img_per_s": ips_gpu32, "gpu_amp_fp16_img_per_s": ips_gpu16,
        "speedup_fp32": ips_gpu32 / ips_cpu, "speedup_amp": ips_gpu16 / ips_cpu,
        "ms_per_slice_cpu": 1e3 / ips_cpu, "ms_per_slice_gpu_amp": 1e3 / ips_gpu16,
        "test_dice": dice_score(pred, Yte),
    }
    print(res["inference"])
    save_results("segmentation", res)

    # --- figures ------------------------------------------------------------------------------
    save_figure(xte, yte, pred.cpu().numpy())


def save_figure(xte, yte, p):
    fig, ax = plt.subplots(3, 4, figsize=(13, 10))
    for r, i in enumerate((0, 7, 15)):
        ax[r, 0].imshow(xte[i, 0], cmap="gray"); ax[r, 0].set_title("Input slice" if r == 0 else "")
        ax[r, 1].imshow(yte[i, 0], cmap="gray"); ax[r, 1].set_title("Ground truth" if r == 0 else "")
        ax[r, 2].imshow(p[i, 0], cmap="gray"); ax[r, 2].set_title("U-Net prediction" if r == 0 else "")
        ax[r, 3].imshow(xte[i, 0], cmap="gray")
        ax[r, 3].contour(yte[i, 0], [0.5], colors="white", linewidths=2.5)
        ax[r, 3].contour(p[i, 0], [0.5], colors="black", linewidths=1.3, linestyles="--")
        if r == 0:
            ax[r, 3].set_title("Truth (white) / prediction (black dashed)")
        for a in ax[r]:
            a.axis("off")
    plt.tight_layout()
    plt.savefig(FIGS / "segmentation.png", dpi=140, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
