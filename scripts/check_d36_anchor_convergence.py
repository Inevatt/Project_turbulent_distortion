"""Numerical convergence test for D3 local-PSF anchor spacing.

This does NOT change production code.  It draws one full dense coefficient
realization and reuses the same tilt/high-order field while only changing the
number of local-PSF anchors.  Therefore differences are caused only by the
PSF-anchor discretization.
"""
from pathlib import Path
import time
import yaml
import numpy as np
from scipy.ndimage import map_coordinates

from src.data import split_indices
from src.distortion.d3 import (
    D3Anisoplanatic,
    _exact_tilt_free_psfs,
    _regular_weights,
    _sample_regular,
    _scatter_anchor_blur,
)

CFG = "configs/seed1.yaml"
N_REAL = 3
DR_VALUES = (1.0, 3.0, 5.0)
GRIDS = (5, 9, 17)
DEEP_GRID = 33


def _rmse(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _psnr_from_rmse(e):
    return float("inf") if e == 0.0 else float(20.0 * np.log10(1.0 / e))


def _make_psfs(deg, high_dense, g):
    high_anchor = _sample_regular(high_dense, int(g))
    vectors = high_anchor.reshape(high_anchor.shape[0], -1).T
    psfs = _exact_tilt_free_psfs(deg.wf, vectors)
    k = psfs.shape[-1]
    return psfs.reshape(g, g, k, k)


def _interp_psfs_to_grid(psfs, out_g):
    """Interpolate a g×g PSF field onto an out_g×out_g probe grid."""
    g = psfs.shape[0]
    w = _regular_weights(int(out_g), int(g))
    return np.einsum("ay,bx,abij->yxij", w, w, psfs, optimize=True)


def _psf_l1(approx, ref):
    # Each PSF sums to one, so this is an intuitive per-location kernel error.
    return float(np.mean(np.sum(np.abs(approx - ref), axis=(-2, -1))))


def _render(deg, warped, psfs):
    g = psfs.shape[0]
    w = _regular_weights(warped.shape[0], g)
    out = _scatter_anchor_blur(warped, psfs, w, w)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def main():
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    d = cfg["data"]
    f0 = cfg["degradation"]["diffraction_fwhm_px"]
    n = int(d["crop_px"] + 2 * d["margin_px"])
    if n != 208:
        print(f"NOTE: working size is {n}x{n}, not 208x208")

    _, val_idx, _ = split_indices(
        d["tiles"], d["val_frac"], d["test_frac"], cfg["split_seed"]
    )
    tiles = np.load(Path(d["tiles"]) / "tiles.npy", mmap_mode="r")
    if len(val_idx) < N_REAL:
        raise RuntimeError("validation split is too small for convergence test")

    deg = D3Anisoplanatic(f0)
    rows = []

    print("D3_36 PSF-anchor convergence")
    print(f"working image: {n}x{n}")
    print("grids: 5x5 (~55 px), 9x9 (~27.5 px), 17x17 (~13.8 px)")
    print("same dense a2..a36 realization is reused across all grids")
    print()

    for r in range(N_REAL):
        tile = np.asarray(tiles[val_idx[r]], dtype=np.float32) / 255.0
        y0 = (tile.shape[0] - n) // 2
        x0 = (tile.shape[1] - n) // 2
        img = tile[y0:y0+n, x0:x0+n]
        if img.shape != (n, n):
            raise RuntimeError(f"tile too small: got crop {img.shape}, need {(n,n)}")

        for dr in DR_VALUES:
            rng = np.random.default_rng([20260916, r, int(dr * 1000)])
            t0 = time.perf_counter()

            # Diagnostic full field: needed specifically so every grid sees the
            # SAME underlying high-order realization at all spatial positions.
            a = deg.coefficient_field(n, dr, rng)
            tilt = np.empty((2, n, n), dtype=np.float32)
            tilt[0] = a[1] * np.float32(deg.wf.px_per_rad)
            tilt[1] = a[0] * np.float32(deg.wf.px_per_rad)
            warped = map_coordinates(
                img, deg._grid(n) - tilt, order=3, mode="reflect"
            ).astype(np.float32, copy=False)
            high = a[2:]

            psfs = {g: _make_psfs(deg, high, g) for g in GRIDS}
            outs = {g: _render(deg, warped, psfs[g]) for g in GRIDS}

            e59 = _rmse(outs[5], outs[9])
            e917 = _rmse(outs[9], outs[17])
            e517 = _rmse(outs[5], outs[17])
            blur = _rmse(outs[17], warped)

            # PSF-field error at exact 17x17 probe locations.
            p5 = _interp_psfs_to_grid(psfs[5], 17)
            p9 = _interp_psfs_to_grid(psfs[9], 17)
            p17 = psfs[17]
            l1_5 = _psf_l1(p5, p17)
            l1_9 = _psf_l1(p9, p17)

            ratio = e917 / e59 if e59 > 0 else float("nan")
            rel = e517 / blur if blur > 0 else float("nan")
            dt = time.perf_counter() - t0
            rows.append((dr, e59, e917, e517, ratio, rel, l1_5, l1_9))

            print(
                f"real={r} dr={dr:.0f} | "
                f"RMSE 5->9={e59:.5f} 9->17={e917:.5f} "
                f"5->17={e517:.5f} ({_psnr_from_rmse(e517):.1f} dB) | "
                f"ratio={ratio:.2f} | grid/blurring={rel:.2%} | "
                f"PSF-L1 5={l1_5:.3f} 9={l1_9:.3f} | {dt:.1f}s"
            )

    print("\nAverages by D/r0:")
    for dr in DR_VALUES:
        a = np.asarray([x[1:] for x in rows if x[0] == dr], dtype=np.float64)
        m = a.mean(axis=0)
        print(
            f"dr={dr:.0f}: RMSE 5->9={m[0]:.5f}, 9->17={m[1]:.5f}, "
            f"5->17={m[2]:.5f}, ratio={m[3]:.2f}, "
            f"grid/blurring={m[4]:.2%}, PSF-L1 5={m[5]:.3f}, 9={m[6]:.3f}"
        )

    # One deeper worst-case comparison.  33x33 is expensive, so do it once.
    print("\nDeep worst-case check: first tile, D/r0=5, 17x17 vs 33x33")
    tile = np.asarray(tiles[val_idx[0]], dtype=np.float32) / 255.0
    y0 = (tile.shape[0] - n) // 2
    x0 = (tile.shape[1] - n) // 2
    img = tile[y0:y0+n, x0:x0+n]
    rng = np.random.default_rng([20260916, 999, 5000])
    t0 = time.perf_counter()
    a = deg.coefficient_field(n, 5.0, rng)
    tilt = np.empty((2, n, n), dtype=np.float32)
    tilt[0] = a[1] * np.float32(deg.wf.px_per_rad)
    tilt[1] = a[0] * np.float32(deg.wf.px_per_rad)
    warped = map_coordinates(img, deg._grid(n) - tilt, order=3, mode="reflect").astype(np.float32)
    high = a[2:]

    ps5 = _make_psfs(deg, high, 5)
    ps17 = _make_psfs(deg, high, 17)
    ps33 = _make_psfs(deg, high, DEEP_GRID)
    out5 = _render(deg, warped, ps5)
    out17 = _render(deg, warped, ps17)
    out33 = _render(deg, warped, ps33)
    e517 = _rmse(out5, out17)
    e1733 = _rmse(out17, out33)
    e533 = _rmse(out5, out33)
    print(
        f"RMSE 5->17={e517:.5f}, 17->33={e1733:.5f}, 5->33={e533:.5f}; "
        f"17->33 / 5->17 = {e1733/e517:.2f}; elapsed={time.perf_counter()-t0:.1f}s"
    )

    print("\nInterpretation:")
    print("  - convergence is good if the refinement increments shrink clearly:")
    print("      RMSE(9,17) << RMSE(5,9), and RMSE(17,33) << RMSE(5,17).")
    print("  - grid/blurring reports the 5x5 discretization change relative to the")
    print("    magnitude of the blur itself; smaller is better.")
    print("  - there is intentionally no hard PASS threshold: the project has not")
    print("    established a source-backed numerical tolerance for this approximation.")


if __name__ == "__main__":
    main()
