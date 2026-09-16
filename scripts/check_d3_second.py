"""Acceptance checks for the final project D3.

Run from the project root with this file in ``scripts``::

    python -m scripts.check_d3_second

No asset build or extra calibration step is required.
"""

import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from src.distortion.d3 import (
    D3Anisoplanatic,
    PSF_ANCHOR_SPACING_PX,
    _regular_weights,
    _sample_regular,
    _scatter_anchor_blur,
)
from src.distortion.dense_zernike import (
    ANCHOR_SPACING_PX,
    DenseZernikeSampler,
    TILT_CORR_HALF_PX,
)
from src.distortion.wavefront import K_MODES, PSF_RADIUS_PX, noll_cov
from src.optics import TILT_CLIP


def check_fast_path_equivalence():
    print("[0] optimized coefficient path == full reference path")
    sampler = DenseZernikeSampler()
    n, dr, g = 208, 3.0, 5
    full = sampler.sample(n, dr, np.random.default_rng(24680))
    tilt, high = sampler.sample_for_d3(n, dr, np.random.default_rng(24680), g)
    tilt_err = float(np.max(np.abs(tilt - full[:2])))
    high_ref = _sample_regular(full[2:], g)
    high_err = float(np.max(np.abs(high - high_ref)))
    print(f"  dense tilt max error = {tilt_err:.3e}")
    print(f"  high-order PSF-anchor max error = {high_err:.3e}")
    assert tilt_err < 1e-7
    assert high_err < 2e-6


def check_covariance():
    print("[1] Noll point covariance")
    assert K_MODES == 231
    assert TILT_CORR_HALF_PX == 55.0
    sampler = DenseZernikeSampler()

    represented = sampler.modal_covariance_from_roots()
    err = float(np.max(np.abs(represented - sampler.cov)))
    print(f"  algebraic max |LL^T-C| = {err:.3e}")
    assert err < 1e-6

    # Empirical check goes through FFT generation + interpolation correction.
    rng = np.random.default_rng(1234)
    take = np.array([0, 1, 2, 5, 20, 100, 229])
    vals = []
    for _ in range(350):
        a = sampler.sample(17, 1.0, rng)
        vals.append(a[take, 8, 8])
    vals = np.asarray(vals)
    ref = noll_cov(K_MODES)[1:, 1:][np.ix_(take, take)]
    emp = np.cov(vals, rowvar=False, bias=True)
    diag_ratio = np.diag(emp) / np.diag(ref)
    print("  empirical variance ratios:", " ".join(f"{x:.3f}" for x in diag_ratio))
    assert np.all((diag_ratio > 0.78) & (diag_ratio < 1.22))

    # Known non-zero Noll correlation a3-a7 (indices 1 and 5 after piston).
    rng = np.random.default_rng(5678)
    pair = np.empty((500, 2), dtype=np.float32)
    for i in range(len(pair)):
        a = sampler.sample(9, 1.0, rng)
        pair[i] = a[[1, 5], 4, 4]
    full = noll_cov(K_MODES)[1:, 1:]
    ref_corr = full[1, 5] / np.sqrt(full[1, 1] * full[5, 5])
    emp_corr = np.corrcoef(pair.T)[0, 1]
    print(f"  corr(a3,a7): empirical={emp_corr:.3f}, Noll={ref_corr:.3f}")
    assert abs(emp_corr - ref_corr) < 0.10


def _interpolated_corr_x(sampler, n, y, x0, x1):
    """Exact correlation after the sampler's bilinear interpolation/correction."""
    g, _ = sampler._geometry(n)
    m = sampler._embed_size(n)
    root = sampler._spectrum_root(n).astype(np.float64)
    import scipy.fft as _sfft
    c = _sfft.irfft2(root * root, s=(m, m)).real
    c /= c[0, 0]

    def weights(py, px):
        yy = py * (g - 1) / (n - 1)
        xx = px * (g - 1) / (n - 1)
        y0, x0i = int(np.floor(yy)), int(np.floor(xx))
        y1, x1i = min(y0 + 1, g - 1), min(x0i + 1, g - 1)
        fy, fx = yy - y0, xx - x0i
        return [
            ((y0, x0i), (1-fy)*(1-fx)),
            ((y0, x1i), (1-fy)*fx),
            ((y1, x0i), fy*(1-fx)),
            ((y1, x1i), fy*fx),
        ]

    def cov(p, q):
        total = 0.0
        for (ip, wp) in weights(*p):
            for (iq, wq) in weights(*q):
                total += wp*wq*c[(iq[0]-ip[0]) % m, (iq[1]-ip[1]) % m]
        return total

    p, q = (y, x0), (y, x1)
    return cov(p, q) / np.sqrt(cov(p, p) * cov(q, q))


def check_spatial_geometry():
    print("[2] fixed spatial geometry")
    sampler = DenseZernikeSampler()
    got = float(_interpolated_corr_x(sampler, 208, 100, 70, 125))
    print(f"  actual interpolated C(55 px) = {got:.4f}")
    assert abs(got - 0.5) < 0.03

    a1 = DenseZernikeSampler().sample(9, 1.0, np.random.default_rng(99))
    a5 = DenseZernikeSampler().sample(9, 5.0, np.random.default_rng(99))
    scale = 5.0 ** (5.0 / 6.0)
    err = float(np.max(np.abs(a5 - a1 * scale)))
    print(f"  exact D/r0^(5/6) scaling max error = {err:.3e}")
    assert err < 5e-6


def check_scattering():
    print("[3] source-indexed/scattering blur")
    rng = np.random.default_rng(7)
    src = rng.random((13, 13), dtype=np.float32)
    g = 3
    side = 5
    r = side // 2
    psfs = rng.random((g, g, side, side), dtype=np.float32)
    psfs /= psfs.sum(axis=(-2, -1), keepdims=True)
    w = _regular_weights(len(src), g)

    got = _scatter_anchor_blur(src, psfs, w, w, chunk=3)

    # Algebraically equivalent reference: sum_a conv(pad(src*w_a), h_a).
    ref = np.zeros_like(src)
    for ay in range(g):
        for ax in range(g):
            weight = w[ay, :, None] * w[ax, None, :]
            weighted = np.pad(src * weight, r, mode="symmetric")
            ref += fftconvolve(weighted, psfs[ay, ax], mode="valid")
    err = float(np.max(np.abs(got - ref)))
    print(f"  scattering identity max error = {err:.3e}")
    assert err < 3e-6

    # If every local PSF is identical, spatially varying scattering must
    # collapse exactly to the ordinary D2-style convolution with that kernel.
    kernel = psfs[0, 0].copy()
    const = np.broadcast_to(kernel, psfs.shape).copy()
    got = _scatter_anchor_blur(src, const, w, w, chunk=4)
    ref = fftconvolve(np.pad(src, r, mode="symmetric"), kernel, mode="valid")
    err = float(np.max(np.abs(got - ref)))
    print(f"  constant-PSF limit max error = {err:.3e}")
    assert err < 3e-6


def check_order_and_registration(fwhm=2.0):
    print("[4] T -> B and registration")
    d3 = D3Anisoplanatic(fwhm, psf_anchor_spacing_px=16.0)
    n = 48
    yy, xx = np.mgrid[0:n, 0:n]
    img = np.clip(0.45 + 0.25*np.sin(xx/3.0) + 0.20*np.cos(yy/4.0), 0, 1).astype(np.float32)

    # Replace only the random draw by a deterministic field.  All real D3
    # warp/scattering code below stays untouched.
    a = np.zeros((K_MODES - 1, n, n), dtype=np.float32)
    a[0] = 0.10 + 0.03*np.sin(yy/7.0)  # j=2 -> x
    a[1] = -0.08 + 0.02*np.cos(xx/8.0) # j=3 -> y
    a[2] = 0.05*np.sin((xx+yy)/9.0)    # one high-order mode varies

    g, _ = d3._psf_geometry(n)
    high_anchor = _sample_regular(a[2:], g)

    def fixed_components(_n, dr, _rng):
        tilt_coeff = a[:2].copy()
        lim = np.float32(TILT_CLIP*d3.wf.sigma_tilt_rad*dr**(5.0/6.0))
        np.clip(tilt_coeff, -lim, lim, out=tilt_coeff)
        return tilt_coeff, high_anchor.copy()

    d3.coefficient_components = fixed_components
    out, shift = d3(img, 3.0, np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0

    tilt = np.empty((2, n, n), dtype=np.float32)
    tilt[0] = a[1] * np.float32(d3.wf.px_per_rad)
    tilt[1] = a[0] * np.float32(d3.wf.px_per_rad)
    # Independently reconstruct the intended order from the fixed field.
    grid = np.mgrid[0:n, 0:n].astype(np.float32)
    warped = map_coordinates(img, grid - tilt, order=3, mode="reflect").astype(np.float32)
    psfs, w = d3._local_psfs(a[2:])
    correct = _scatter_anchor_blur(warped, psfs, w, w)
    correct = np.clip(correct, 0.0, 1.0)
    impl_err = float(np.max(np.abs(out - correct)))
    print(f"  T->B reconstruction max error={impl_err:.3e}")
    assert impl_err < 3e-6

    # Wrong order B->T must not accidentally be equivalent for this dense field.
    blurred_first = _scatter_anchor_blur(img, psfs, w, w)
    wrong = map_coordinates(blurred_first, grid - tilt, order=3, mode="reflect")
    order_gap = float(np.sqrt(np.mean((correct - wrong) ** 2)))
    print(f"  RMS(T->B minus B->T)={order_gap:.3e}")
    assert order_gap > 1e-5

    expected = (float(tilt[0].mean()), float(tilt[1].mean()))
    err = max(abs(shift[0]-expected[0]), abs(shift[1]-expected[1]))
    print(f"  returned shift={shift}, mean-tilt error={err:.3e}")
    assert err < 1e-7

    # In particular, no PSF centroid is added to the returned registration.
    assert shift == expected

    support = D3Anisoplanatic(fwhm).support_radius_px(5.0)
    print(f"  support radius at D/r0=5: {support} px (project margin=40)")
    assert support <= 40


def check_real_call_and_d3n(fwhm=2.0):
    print("[5] real D3 call + D3n pairing")
    from src.distortion.d3n import D3Noise, NOISE_SIGMA, SIGMA_SPREAD

    rng_img = np.random.default_rng(11)
    img = rng_img.random((48, 48), dtype=np.float32)
    seed = 13579

    d3 = D3Anisoplanatic(fwhm, psf_anchor_spacing_px=24.0)
    t0 = time.time()
    r1 = np.random.default_rng(seed)
    base, shift_base = d3(img, 2.0, r1)
    sec = time.time() - t0
    print(f"  real 48x48 call: {sec:.2f} s")
    assert base.shape == img.shape and base.dtype == np.float32
    assert np.isfinite(base).all()

    # Production geometry: crop 128 + 2*margin 40 = 208.  This is a timing
    # report, not a machine-dependent acceptance threshold.
    big = rng_img.random((208, 208), dtype=np.float32)
    prod = D3Anisoplanatic(fwhm)
    prod(big, 3.0, np.random.default_rng(1))  # warm caches
    t1 = time.perf_counter()
    prod(big, 3.0, np.random.default_rng(2))
    prod_sec = time.perf_counter() - t1
    print(f"  production 208x208 call: {prod_sec:.3f} s")

    # Reproduce exactly what D3Noise must do after the identical super() call.
    sigma = float(r1.uniform(
        (1.0-SIGMA_SPREAD)*NOISE_SIGMA,
        (1.0+SIGMA_SPREAD)*NOISE_SIGMA,
    ))
    expected = base + r1.standard_normal(base.shape, dtype=np.float32)*sigma
    expected = np.clip(expected, 0.0, 1.0)

    d3n = D3Noise(fwhm, psf_anchor_spacing_px=24.0)
    noisy, shift_noisy = d3n(img, 2.0, np.random.default_rng(seed))
    pix_err = float(np.max(np.abs(noisy-expected)))
    shift_err = max(abs(shift_base[i]-shift_noisy[i]) for i in (0, 1))
    print(f"  D3n pairing pixel error={pix_err:.3e}, shift error={shift_err:.3e}")
    assert pix_err < 1e-7
    assert shift_err < 1e-12


def main():
    t0 = time.time()
    check_fast_path_equivalence()
    check_covariance()
    check_spatial_geometry()
    check_scattering()
    check_order_and_registration()
    check_real_call_and_d3n()
    print(f"ALL D3 CHECKS PASSED in {time.time()-t0:.1f} s")
    print(f"PSF_ANCHOR_SPACING_PX={PSF_ANCHOR_SPACING_PX:.1f} is numerical only")
    print(f"FFT coefficient spacing={ANCHOR_SPACING_PX:.1f} px is numerical only")


if __name__ == "__main__":
    main()
