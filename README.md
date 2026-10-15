# GPU-Accelerated Medical Image Analysis (CT/MRI)

Case study for Parallel & Distributed Systems: a five-stage CT/MRI processing
pipeline implemented twice (CPU and GPU), benchmarked on one machine, with
correctness checks between the two versions.

| Stage | CPU version | GPU version | Data |
|---|---|---|---|
| 1. CT reconstruction (FBP) | NumPy | custom CUDA kernel `src/kernels/fbp.cu` + cuFFT | Shepp-Logan phantom (simulated) |
| 2. MRI reconstruction (k-space, iterative CS) | PyTorch CPU | PyTorch CUDA (cuFFT) | Stanford MRbrain (real T1 MRI) |
| 3. 3D denoising (5x5x5 Gaussian) | SciPy | custom CUDA kernels `src/kernels/conv3d.cu` (naive + shared-memory tiled), cuDNN | 3D phantom + noise |
| 4a. Threshold segmentation (Otsu) | NumPy | custom CUDA kernels `src/kernels/threshold.cu` (shared-memory histogram + label) | Stanford CThead (real CT) |
| 4b. Lesion segmentation (U-Net) | PyTorch CPU | PyTorch CUDA, FP16 mixed precision | synthetic lesion slices |
| 5. 3D rendering (MIP + DVR) | PyTorch CPU | PyTorch CUDA | Stanford CThead (real CT) |

## Results (RTX 5060, Ryzen 5 7600)

| Stage | CPU | GPU | Speedup |
|---|---|---|---|
| CT reconstruction, 128 x 256² | 11.8 s | 45 ms | 265x |
| MRI iterative recon, 10 x 256² | 670 ms | 96 ms | 7.0x |
| 3D denoise 256³ (incl. transfers) | 1.2 s | 16 ms | 78x |
| U-Net inference, 128 slices | 2.2 s | 49 ms | 46x |
| 3D rendering, 36 views | 16.2 s | 523 ms | 31x |
| **End-to-end** | **32.2 s** | **728 ms** | **44x** |

All numbers are in `results/results.json` and regenerate the deck. GPU and CPU
outputs agree to floating-point rounding (see slide 13). U-Net test Dice: 0.909.

## Layout

```
src/kernels/        fbp.cu, conv3d.cu (compiled automatically with nvcc)
src/*.py            one script per stage, benchmark.py runs all
data/               CThead.npy, MRbrain.npy (+ raw Stanford files), cached sinograms
results/            results.json, figures/, unet.pt
presentation/       build_deck.js -> GPU_Medical_Imaging_Detailed.pptx (32 slides by member, black and white, speaker notes)
                    GPU_Medical_Imaging.pptx = earlier 20-slide summary version
```

## Reproduce

1. CUDA toolkit 13.x and Visual Studio 2022 Build Tools (for `nvcc` on Windows).
2. `pip install -r requirements.txt`
3. Data: download `MRbrain.tar.gz` and `CThead.tar.gz` from
   https://graphics.stanford.edu/data/voldata/, extract into `data/raw/`, and convert to
   `data/MRbrain.npy` / `data/CThead.npy` (16-bit big-endian, 256x256 per slice).
4. `python src/benchmark.py` (about 15 minutes; U-Net training is the slowest part), then
   `python src/make_tiles.py`
5. Deck: `cd presentation && npm install && node build_deck.js`
   (needs `NODE_PATH=./node_modules` and `PPTX_SKILL` pointing to the pptx skill for the theme step).

Timings vary a few percent between runs, especially host-to-device copies.
