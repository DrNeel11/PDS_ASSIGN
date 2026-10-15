"""Run the whole pipeline and print a summary table.

    python src/benchmark.py            # run all stages, then summarise
    python src/benchmark.py --summary  # only summarise results/results.json
"""
import json
import sys

from common import RESULTS, hardware_info, save_results


def summary():
    r = json.loads((RESULTS / "results.json").read_text())
    ct, mri, flt = r["ct_fbp"]["volume"], r["mri"]["cs_recon"], r["filter3d"]["scaling"][-1]
    seg, vis = r["segmentation"], r["render3d"]
    rows = [
        ("CT reconstruction (128 x 256^2, FBP)", ct["cpu_ms"], ct["gpu_total_ms"]),
        ("MRI CS recon (10 x 256^2, 200 iters)", mri["cpu_torch_ms"], mri["gpu_ms"]),
        ("3D denoise (256^3, 5^3 Gaussian)", flt["cpu_scipy_ms"], flt["gpu_tiled_ms"] + flt["h2d_ms"] + flt["d2h_ms"]),
        ("U-Net inference (128 slices)", 128 * seg["inference"]["ms_per_slice_cpu"],
         128 * seg["inference"]["ms_per_slice_gpu_amp"]),
        ("3D rendering (36 DVR views)", vis["dvr_cpu_s"] * 1e3, vis["dvr_gpu_s"] * 1e3),
    ]
    tot_c = sum(c for _, c, _ in rows)
    tot_g = sum(g for _, _, g in rows)
    print(f"{'Stage':42s} {'CPU ms':>10s} {'GPU ms':>10s} {'Speedup':>8s}")
    for n, c, g in rows:
        print(f"{n:42s} {c:10.1f} {g:10.1f} {c / g:7.1f}x")
    print(f"{'END-TO-END':42s} {tot_c:10.1f} {tot_g:10.1f} {tot_c / tot_g:7.1f}x")
    save_results("pipeline", {
        "stages": [{"stage": n, "cpu_ms": c, "gpu_ms": g, "speedup": c / g} for n, c, g in rows],
        "total_cpu_ms": tot_c, "total_gpu_ms": tot_g, "speedup": tot_c / tot_g,
    })


if __name__ == "__main__":
    save_results("hardware", hardware_info())
    if "--summary" not in sys.argv:
        import ct_recon, mri_recon, filtering, segmentation, threshold_seg, visualize
        for m in (ct_recon, mri_recon, filtering, threshold_seg, segmentation, visualize):
            m.main()
    summary()
