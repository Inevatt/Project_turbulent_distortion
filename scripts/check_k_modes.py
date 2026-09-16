"""Convergence study for the number of Zernike modes in D2/D3.

Purpose
-------
Choose K from optical convergence, not from convention.  For every complete
radial order n, K=(n+1)(n+2)/2, this script draws the *same* 231+/max-K
Kolmogorov realization and zeros the tail above K.  Therefore comparisons
between K are paired: the low-order coefficients are exactly identical.

Two independent references are reported:
  1. Fried long-exposure OTF (infinite Kolmogorov spectrum):
       H_LE = H_diff * exp[-3.44 (nu D/r0)^(5/3)].
  2. The largest tested K, to measure finite-modal convergence directly.

The test uses the full 128x128 optical PSF, before PSF_RADIUS_PX cropping, so
that K is judged as a property of the phase model rather than by the later
finite-support approximation.  With --deployed it additionally repeats the
K-vs-Kmax comparison after the exact crop+renormalize operation used by D2.

References
----------
R. J. Noll, "Zernike polynomials and atmospheric turbulence",
JOSA 66(3), 207-211 (1976), doi:10.1364/JOSA.66.000207.
D. L. Fried, "Optical Resolution Through a Randomly Inhomogeneous Medium
for Very Long and Very Short Exposures", JOSA 56(10), 1372-1379 (1966),
doi:10.1364/JOSA.56.001372.
N. Chimitt et al., "Real-Time Dense Field Phase-to-Space Simulation of
Imaging through Atmospheric Turbulence", arXiv:2210.06713 / IEEE TCI.
The latter uses N=36 and explicitly states that N can be increased at the
cost of speed; this script determines what N/K our optical discretization
actually needs.
"""

import argparse
import math
import time

import numpy as np

from src.distortion.wavefront import (
    Kolmogorov,
    PSF_RADIUS_PX,
    noll_cov,
    noll_nm,
)
from src.optics import lambda_over_d_px

NOLL_TOTAL_VAR = 1.0299  # Noll 1976, residual Delta_1, piston excluded.


def k_end(n: int) -> int:
    """Last Noll index in complete radial order n."""
    return (n + 1) * (n + 2) // 2


def full_psf_from_phase(wf: Kolmogorov, phase_on_pupil: np.ndarray) -> np.ndarray:
    """Same Fourier optics as Kolmogorov.psf, but without spatial cropping."""
    u = np.zeros((wf.n_grid, wf.n_grid), dtype=np.complex128)
    u[wf.mask] = np.exp(1j * phase_on_pupil)
    h = np.abs(np.fft.fft2(u)) ** 2
    h = np.fft.fftshift(h)
    return h / h.sum()


def deployed_psf(full_psf: np.ndarray, radius: int) -> np.ndarray:
    """Exact crop+renormalize used by D2, re-embedded on the full FFT grid."""
    n = len(full_psf)
    c = n // 2
    crop = full_psf[c - radius:c + radius + 1,
                    c - radius:c + radius + 1].copy()
    crop /= crop.sum()
    out = np.zeros_like(full_psf)
    out[c - radius:c + radius + 1,
        c - radius:c + radius + 1] = crop
    return out


def otf(psf: np.ndarray) -> np.ndarray:
    """Centered real OTF, normalized to H(0)=1."""
    h = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf))).real
    c = len(h) // 2
    return h / h[c, c]


def rel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    num = np.linalg.norm((a - b)[mask])
    den = np.linalg.norm(b[mask])
    return float(num / den)


def radial_profile(a: np.ndarray, nu: np.ndarray, bins: int = 64) -> np.ndarray:
    """Azimuthal average on 0<=nu<=1, used only for Fried comparison."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (nu >= lo) & (nu < hi)
        if np.any(m):
            out.append(float(a[m].mean()))
    return np.asarray(out, dtype=np.float64)


def block_stats(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a) <= 1:
        return float(a.mean()), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mc", type=int, default=2000,
                   help="Monte-Carlo realizations per D/r0 (default: 2000)")
    p.add_argument("--blocks", type=int, default=10,
                   help="paired blocks for Monte-Carlo standard errors")
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--fwhm", type=float, default=2.0)
    p.add_argument("--n-min", type=int, default=7,
                   help="lowest complete radial order (n=7 => K=36)")
    p.add_argument("--n-max", type=int, default=24,
                   help="largest reference radial order (n=24 => K=325)")
    p.add_argument("--dr", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--deployed", action="store_true",
                   help="also compare after D2's finite-radius crop")
    args = p.parse_args()

    if args.mc < args.blocks or args.mc % args.blocks:
        raise SystemExit("--mc must be divisible by --blocks and >= --blocks")
    if args.n_min < 1 or args.n_max < args.n_min:
        raise SystemExit("invalid radial-order range")

    orders = list(range(args.n_min, args.n_max + 1))
    ks = [k_end(n) for n in orders]
    kmax = ks[-1]

    print("=== K-mode convergence study ===")
    print(f"radial orders: {args.n_min}..{args.n_max}")
    print(f"K range: {ks[0]}..{kmax}")
    print(f"MC: {args.mc}, blocks: {args.blocks}, D/r0: {args.dr}")
    print("reference K is deliberately >231 by default: 231 must be tested, not assumed.")

    t0 = time.perf_counter()
    wf = Kolmogorov(args.fwhm, k_modes=kmax)
    print(f"built K={kmax} basis/root in {time.perf_counter()-t0:.2f}s")

    # Direct sanity check against two tabulated Noll residuals.
    # Delta_2=0.582 and Delta_3=0.134 are printed in Noll's Table 1.
    c3 = noll_cov(3)
    d2 = NOLL_TOTAL_VAR - float(c3[1, 1])
    d3 = NOLL_TOTAL_VAR - float(np.trace(c3[1:3, 1:3]))
    print(f"Noll sanity: Delta_2={d2:.6f} (paper 0.582), "
          f"Delta_3={d3:.6f} (paper 0.134)")
    if abs(d2 - 0.582) > 0.003 or abs(d3 - 0.134) > 0.003:
        raise RuntimeError("noll_cov does not reproduce Noll's tabulated residuals")

    # Noll phase-variance convergence.  Top-left marginals of the max-K
    # covariance are exactly the lower-K Noll distributions.
    cov = noll_cov(kmax)[1:, 1:]
    print("\nNoll phase-variance coverage (D/r0 factor cancels):")
    print("  n     K      captured / 1.0299     residual")
    for n, k in zip(orders, ks):
        captured = float(np.trace(cov[:k - 1, :k - 1]))
        residual = NOLL_TOTAL_VAR - captured
        print(f" {n:2d}  {k:4d}       {captured/NOLL_TOTAL_VAR:9.5%}      {residual:.6f}")

    # Numerical diffraction OTF on exactly the same pupil/grid.  This avoids
    # mixing modal-truncation error with the small discretization difference
    # between an analytic circular pupil and our sampled pupil.
    phase0 = np.zeros(int(wf.mask.sum()), dtype=np.float64)
    ideal_psf = full_psf_from_phase(wf, phase0)
    ideal_otf = np.maximum(otf(ideal_psf), 0.0)

    freq = np.fft.fftshift(np.fft.fftfreq(wf.n_grid))
    fx, fy = np.meshgrid(freq, freq)
    nu = np.hypot(fx, fy) * lambda_over_d_px(args.fwhm)
    optical_band = nu <= 1.0

    block_size = args.mc // args.blocks

    for dr in args.dr:
        # Each D/r0 gets a deterministic independent stream.  Within one
        # realization every K shares exactly the same coefficients 2..K.
        dr_key = int(round(dr * 1_000_000))
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, dr_key]))

        full_sum = np.zeros((args.blocks, len(ks), wf.n_grid, wf.n_grid),
                            dtype=np.float64)
        dep_sum = (np.zeros_like(full_sum) if args.deployed else None)

        start_time = time.perf_counter()
        for sample in range(args.mc):
            b = sample // block_size
            a = wf.coeffs(dr, rng)  # max-K draw; low-K marginals stay exact.
            phase = np.zeros(int(wf.mask.sum()), dtype=np.float64)
            prev = 0

            if args.deployed:
                dy, dx = wf.shift_px(a)
                radius = PSF_RADIUS_PX + int(np.ceil(max(abs(dy), abs(dx))))

            for q, k in enumerate(ks):
                end = k - 1  # a[0] is Noll j=2.
                if end > prev:
                    phase += a[prev:end] @ wf.basis[prev:end]
                h = full_psf_from_phase(wf, phase)
                full_sum[b, q] += h
                if args.deployed:
                    dep_sum[b, q] += deployed_psf(h, radius)
                prev = end

        elapsed = time.perf_counter() - start_time

        # Fried long-exposure OTF, including the same numerical diffraction OTF.
        fried = ideal_otf * np.exp(-3.44 * (nu * dr) ** (5.0 / 3.0))

        # Total means and block means.
        full_block_otf = np.empty_like(full_sum)
        for b in range(args.blocks):
            for q in range(len(ks)):
                full_block_otf[b, q] = otf(full_sum[b, q] / block_size)
        full_total_otf = np.array([
            otf(full_sum[:, q].sum(axis=0) / args.mc) for q in range(len(ks))
        ])

        if args.deployed:
            dep_block_otf = np.empty_like(dep_sum)
            for b in range(args.blocks):
                for q in range(len(ks)):
                    dep_block_otf[b, q] = otf(dep_sum[b, q] / block_size)
            dep_total_otf = np.array([
                otf(dep_sum[:, q].sum(axis=0) / args.mc) for q in range(len(ks))
            ])

        ref = full_total_otf[-1]
        print(f"\nD/r0={dr:g}   ({elapsed:.1f}s)")
        print("  n    K   radialL2->Fried  relL2->Kmax (paired)      max|dH|->Kmax")

        for q, (n, k) in enumerate(zip(orders, ks)):
            fried_err = float(np.linalg.norm(radial_profile(full_total_otf[q], nu) - radial_profile(fried, nu)) / np.linalg.norm(radial_profile(fried, nu)))
            ref_err = rel_l2(full_total_otf[q], ref, optical_band)
            max_abs = float(np.max(np.abs((full_total_otf[q] - ref)[optical_band])))

            block_ref = [
                rel_l2(full_block_otf[b, q], full_block_otf[b, -1], optical_band)
                for b in range(args.blocks)
            ]
            _, ref_se = block_stats(block_ref)
            print(f" {n:2d}  {k:4d}      {fried_err:9.5f}       "
                  f"{ref_err:9.5f} +/- {ref_se:7.5f}       {max_abs:9.5f}")

        # Incremental convergence is often more informative than an arbitrary
        # acceptance threshold: if K->next K still moves the OTF, K is not at
        # a plateau yet.
        print("  incremental relL2 K -> next complete radial order:")
        for q in range(len(ks) - 1):
            inc = rel_l2(full_total_otf[q], full_total_otf[q + 1], optical_band)
            print(f"    {ks[q]:4d} -> {ks[q+1]:4d}: {inc:.6f}")

        if args.deployed:
            dep_ref = dep_total_otf[-1]
            print("  deployed crop+renormalize: relL2 -> deployed Kmax")
            for q, (n, k) in enumerate(zip(orders, ks)):
                e = rel_l2(dep_total_otf[q], dep_ref, optical_band)
                print(f"    n={n:2d} K={k:4d}: {e:.6f}")

    print("\nInterpretation:")
    print("  * Do NOT choose K from phase-variance coverage alone; high modes can")
    print("    have a visible OTF effect despite small total variance.")
    print("  * Fried is the infinite-spectrum physical reference; Kmax is only a")
    print("    numerical convergence reference.")
    print("  * Prefer the smallest complete radial order for which K->next and")
    print("    K->Kmax changes are negligible relative to the accuracy you need.")
    print("  * If K=231 still moves appreciably toward Kmax/Fried, 231 is not a")
    print("    mathematically converged endpoint; it is merely a cost/accuracy choice.")


if __name__ == "__main__":
    main()
