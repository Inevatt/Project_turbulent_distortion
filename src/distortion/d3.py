"""D3: dense anisoplanatic Zernike turbulence.

Frozen experiment semantics:
  * K=231 and the same Noll covariance/Zernike normalization as D2;
  * one joint spatial coefficient field a2..a231;
  * a2,a3 -> dense tilt field;
  * a4..a231 -> spatially varying tilt-free PSF;
  * T first, then B;
  * B is source-indexed/scattering;
  * only the global mean tilt is returned for target registration;
  * blur centroid is NOT removed.

The only runtime approximation is numerical: exact tilt-free PSFs are evaluated
on a regular anchor grid and bilinearly interpolated across source pixels.
Scattering is then exact for that interpolated PSF field.  No external
model or extra physical parameter is required.
"""

import numpy as np
import scipy.fft as sfft
from scipy.fft import next_fast_len
from scipy.ndimage import map_coordinates

from .base import Degradation
from .dense_zernike import DenseZernikeSampler, TILT_CORR_HALF_PX
from .wavefront import K_MODES, Kolmogorov, PSF_RADIUS_PX
from ..optics import TILT_CLIP

SPLINE_ORDER = 3
SPLINE_REACH_PX = 8
# Numerical grid for local PSFs.  With the fixed 55 px spatial geometry this
# uses one PSF anchor per half-correlation length; this is only a
# numerical approximation of the smoothly varying local PSF field.
PSF_ANCHOR_SPACING_PX = TILT_CORR_HALF_PX
SCATTER_CHUNK = 2


def _regular_weights(n, g):
    """1-D linear interpolation weights from g anchors to n pixels.

    Returns W with shape (g,n), non-negative and sum_a W[a,x] == 1.
    """
    n, g = int(n), int(g)
    if g < 2 or n < 2:
        raise ValueError("n and g must be >= 2")
    q = np.linspace(0.0, g - 1.0, n, dtype=np.float32)
    i0 = np.floor(q).astype(np.intp)
    i1 = np.minimum(i0 + 1, g - 1)
    t = q - i0
    w = np.zeros((g, n), dtype=np.float32)
    x = np.arange(n)
    w[i0, x] += 1.0 - t
    w[i1, x] += t
    return w


def _sample_regular(field, g):
    """Bilinearly sample (C,n,n) on a regular g x g anchor grid."""
    field = np.asarray(field, dtype=np.float32)
    c, n, n2 = field.shape
    if n != n2:
        raise ValueError("field must be square")
    # First y, then x.  This is equivalent to evaluating the same bilinear
    # interpolation used by _regular_weights at the anchor coordinates.
    # Anchors are exactly the end points of the dense field's coordinate span.
    q = np.linspace(0.0, n - 1.0, g, dtype=np.float32)
    i0 = np.floor(q).astype(np.intp)
    i1 = np.minimum(i0 + 1, n - 1)
    t = q - i0
    y = field[:, i0, :] * (1.0 - t)[None, :, None] + field[:, i1, :] * t[None, :, None]
    return (
        y[:, :, i0] * (1.0 - t)[None, None, :]
        + y[:, :, i1] * t[None, None, :]
    ).astype(np.float32, copy=False)


def _exact_tilt_free_psfs(wf, high, batch=64):
    """Exact local PSFs from a4..a231 using the same pupil model as D2."""
    high = np.asarray(high, dtype=np.float32)
    if high.ndim != 2 or high.shape[1] != K_MODES - 3:
        raise ValueError(f"expected (N,{K_MODES - 3}) high-order coefficients")

    r = PSF_RADIUS_PX
    c = wf.n_grid // 2
    basis_hi = wf.basis[2:].astype(np.float32, copy=False)
    out = []
    for start in range(0, len(high), int(batch)):
        hcoeff = high[start:start + int(batch)]
        phase = hcoeff @ basis_hi
        u = np.zeros((len(hcoeff), wf.n_grid, wf.n_grid), dtype=np.complex64)
        u[:, wf.mask] = np.exp(1j * phase).astype(np.complex64, copy=False)
        h = np.abs(sfft.fft2(u, axes=(-2, -1))) ** 2
        h = sfft.fftshift(h, axes=(-2, -1))
        h = h[:, c-r:c+r+1, c-r:c+r+1]
        h /= h.sum(axis=(-2, -1), keepdims=True)
        out.append(h.astype(np.float32, copy=False))
    return np.concatenate(out, axis=0)


def _scatter_anchor_blur(src, psfs, wy, wx, chunk=SCATTER_CHUNK):
    """Source-indexed blur for a bilinearly interpolated anchor PSF field.

    Local PSF at source pixel (y,x):
        h_(y,x) = sum_ay,ax wy[ay,y] * wx[ax,x] * psfs[ay,ax]

    By linearity, scattering is therefore a sum of ordinary convolutions of
    ``src * anchor_weight`` with the corresponding anchor PSF.  The weight is
    attached to the SOURCE before convolution; this is not gathering.
    """
    src = np.asarray(src, dtype=np.float32)
    psfs = np.asarray(psfs, dtype=np.float32)
    gy, gx, kh, kw = psfs.shape
    if gy != wy.shape[0] or gx != wx.shape[0]:
        raise ValueError("anchor PSF/weight shapes disagree")
    if kh != kw or kh % 2 != 1:
        raise ValueError("PSF kernels must be odd square arrays")
    if wy.shape[1] != src.shape[0] or wx.shape[1] != src.shape[1]:
        raise ValueError("weight maps do not match source shape")

    r = kh // 2
    h, w = src.shape
    # Same boundary convention as D2: symmetric == scipy ndimage reflect.
    src_pad = np.pad(src, r, mode="symmetric")
    hp, wp = src_pad.shape
    fy = next_fast_len(hp + kh - 1)
    fx = next_fast_len(wp + kw - 1)
    y0 = x0 = kh - 1
    out = np.zeros((h, w), dtype=np.float32)

    pairs = [(ay, ax) for ay in range(gy) for ax in range(gx)]
    for start in range(0, len(pairs), int(chunk)):
        sub = pairs[start:start + int(chunk)]
        weights = np.stack(
            [wy[ay, :, None] * wx[ax, None, :] for ay, ax in sub], axis=0
        ).astype(np.float32, copy=False)
        # pad(src * weight) is the correct symmetric extension of the weighted
        # source.  Computing the product before padding avoids any ambiguity.
        weighted = np.pad(
            src[None, :, :] * weights,
            ((0, 0), (r, r), (r, r)),
            mode="symmetric",
        )
        kernels = np.stack([psfs[ay, ax] for ay, ax in sub], axis=0)
        fs = sfft.rfft2(weighted, s=(fy, fx), axes=(-2, -1))
        fk = sfft.rfft2(kernels, s=(fy, fx), axes=(-2, -1))
        conv = sfft.irfft2(fs * fk, s=(fy, fx), axes=(-2, -1))
        out += conv[:, y0:y0+h, x0:x0+w].sum(axis=0).astype(np.float32)
    return out


class D3Anisoplanatic(Degradation):
    name = "d3"

    def __init__(
        self,
        diffraction_fwhm_px,
        corr_half_px=TILT_CORR_HALF_PX,
        psf_anchor_spacing_px=PSF_ANCHOR_SPACING_PX,
    ):
        super().__init__(diffraction_fwhm_px)
        self.wf = Kolmogorov(diffraction_fwhm_px)
        self.sampler = DenseZernikeSampler(K_MODES, corr_half_px)
        self.psf_anchor_spacing_px = float(psf_anchor_spacing_px)
        if self.psf_anchor_spacing_px <= 0.0:
            raise ValueError("psf_anchor_spacing_px must be positive")
        self._grid_cache = {}
        self._weight_cache = {}

    def support_radius_px(self, d_over_r0):
        return (
            PSF_RADIUS_PX
            + SPLINE_REACH_PX
            + int(np.ceil(TILT_CLIP * self.wf.tilt_sigma_px(d_over_r0)))
        )

    def _grid(self, n):
        n = int(n)
        if n not in self._grid_cache:
            self._grid_cache[n] = np.mgrid[0:n, 0:n].astype(np.float32)
        return self._grid_cache[n]

    def _psf_geometry(self, n):
        n = int(n)
        g = int(np.ceil((n - 1) / self.psf_anchor_spacing_px)) + 1
        g = max(2, min(g, n))
        key = (n, g)
        if key not in self._weight_cache:
            w = _regular_weights(n, g)
            self._weight_cache[key] = w
        return g, self._weight_cache[key]

    def _tilt_limit(self, d_over_r0):
        return np.float32(
            TILT_CLIP * self.wf.sigma_tilt_rad * float(d_over_r0) ** (5.0 / 6.0)
        )

    def coefficient_field(self, n, d_over_r0, rng):
        """Full dense a2..a231 field, retained for diagnostics/tests.

        Production ``__call__`` uses ``coefficient_components`` below so the
        228 high-order planes are not materialized at unused pixels.
        """
        a = self.sampler.sample(n, d_over_r0, rng)
        np.clip(a[:2], -self._tilt_limit(d_over_r0), self._tilt_limit(d_over_r0), out=a[:2])
        return a

    def coefficient_components(self, n, d_over_r0, rng):
        """Coefficient data actually consumed by D3.

        Returns dense j=2,3 and j=4..231 only at the local-PSF anchor grid.
        Both come from one joint FFT/Noll realization.
        """
        g, _ = self._psf_geometry(n)
        tilt_coeff, high_anchor = self.sampler.sample_for_d3(
            n, d_over_r0, rng, high_grid=g
        )
        lim = self._tilt_limit(d_over_r0)
        np.clip(tilt_coeff, -lim, lim, out=tilt_coeff)
        return tilt_coeff, high_anchor

    def _local_psfs_from_anchors(self, high_anchor):
        """Exact tilt-free PSFs from already sampled high-order anchors."""
        high_anchor = np.asarray(high_anchor, dtype=np.float32)
        if high_anchor.ndim != 3 or high_anchor.shape[0] != K_MODES - 3:
            raise ValueError(
                f"expected ({K_MODES - 3},g,g) high-order anchors, got {high_anchor.shape}"
            )
        g, g2 = high_anchor.shape[1:]
        if g != g2:
            raise ValueError("high-order anchor grid must be square")
        vectors = high_anchor.reshape(K_MODES - 3, -1).T
        psfs = _exact_tilt_free_psfs(self.wf, vectors)
        side = 2 * PSF_RADIUS_PX + 1
        return psfs.reshape(g, g, side, side)

    def _local_psfs(self, high):
        """Reference full-field path used by diagnostics."""
        _, n, n2 = high.shape
        if n != n2:
            raise ValueError("D3 expects square fields")
        g, w = self._psf_geometry(n)
        high_anchor = _sample_regular(high, g)
        return self._local_psfs_from_anchors(high_anchor), w

    def __call__(self, img, d_over_r0, rng):
        img = np.asarray(img, dtype=np.float32)
        if img.ndim != 2 or img.shape[0] != img.shape[1]:
            raise ValueError(f"D3 expects square 2-D input, got {img.shape}")
        n = img.shape[0]

        # ONE joint realization, so tilt/high-order Noll correlations are kept.
        tilt_coeff, high_anchor = self.coefficient_components(n, d_over_r0, rng)

        # j=2 is x-tilt, j=3 is y-tilt; map_coordinates uses (y,x).
        tilt = np.empty((2, n, n), dtype=np.float32)
        tilt[0] = tilt_coeff[1] * np.float32(self.wf.px_per_rad)
        tilt[1] = tilt_coeff[0] * np.float32(self.wf.px_per_rad)

        # T first.  Content at source p moves to p+T(p), therefore inverse
        # sampling of the warped image reads the source at p-T(p).
        warped = map_coordinates(
            img,
            self._grid(n) - tilt,
            order=SPLINE_ORDER,
            mode="reflect",
        ).astype(np.float32, copy=False)

        # B second.  a2,a3 never enter these PSFs: they are tilt-free.
        _, w = self._psf_geometry(n)
        psfs = self._local_psfs_from_anchors(high_anchor)
        out = _scatter_anchor_blur(warped, psfs, w, w)
        np.clip(out, 0.0, 1.0, out=out)

        # data.py removes only the recoverable global translation.  There is no
        # blur-centroid term here by design.
        shift = (float(tilt[0].mean()), float(tilt[1].mean()))
        return out.astype(np.float32, copy=False), shift
