"""Anisoplanatic finite-mode turbulence: mode-specific fields, tilt then blur.

Exact pupil PSF at every pixel, normalized source-indexed scattering.
The retained anchor helpers are diagnostics only. Production has no PSF
interpolation and does not use pretrained surrogate weights.
"""

import numpy as np
import scipy.fft as sfft
from scipy.signal import fftconvolve

from .base import Degradation
from .dense_zernike import DenseZernikeSampler, TILT_CORR_HALF_PX
from .wavefront import K_MODES, Kolmogorov, PSF_RADIUS_PX
from ..optics import TILT_CLIP

SPLINE_REACH_PX = 8  # conservative existing 40px margin budget; warp is bilinear
# Legacy anchor helpers below are retained only for convergence diagnostics.
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
    """Exact local PSFs from a4..a36 using the same pupil model as D2."""
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
    """Same source-indexed operator, exploiting compact support of each tent."""
    r=psfs.shape[-1]//2
    h,w=src.shape
    src=np.pad(src,r,mode='symmetric')
    wy=np.pad(wy,((0,0),(r,r)),mode='symmetric')
    wx=np.pad(wx,((0,0),(r,r)),mode='symmetric')
    out=np.zeros((h+4*r,w+4*r),np.float32)
    for iy in range(len(wy)):
        ys=np.flatnonzero(wy[iy]); y0,y1=ys[0],ys[-1]+1
        for ix in range(len(wx)):
            xs=np.flatnonzero(wx[ix]); x0,x1=xs[0],xs[-1]+1
            v=src[y0:y1,x0:x1]*wy[iy,y0:y1,None]*wx[ix,None,x0:x1]
            conv=fftconvolve(v,psfs[iy,ix],mode='full')
            out[y0:y1+2*r,x0:x1+2*r]+=conv
    return out[2*r:2*r+h,2*r:2*r+w]


class D3Anisoplanatic(Degradation):
    name = "d3"

    def __init__(self, diffraction_fwhm_px, corr_half_px=TILT_CORR_HALF_PX):
        super().__init__(diffraction_fwhm_px)
        self.wf = Kolmogorov(diffraction_fwhm_px)
        self.sampler = DenseZernikeSampler(K_MODES, corr_half_px)

    def support_radius_px(self, d_over_r0):
        return (PSF_RADIUS_PX + SPLINE_REACH_PX
                + int(np.ceil(TILT_CLIP * self.wf.tilt_sigma_px(d_over_r0))))

    def _tilt_limit(self, d_over_r0):
        return np.float32(
            TILT_CLIP * self.wf.sigma_tilt_rad * float(d_over_r0) ** (5.0 / 6.0)
        )

    def coefficient_field(self, n, d_over_r0, rng):
        """Dense Noll j=2..36 realization; clipping applies only to tilt."""
        a = self.sampler.sample(n, d_over_r0, rng)
        np.clip(a[:2], -self._tilt_limit(d_over_r0), self._tilt_limit(d_over_r0), out=a[:2])
        return a

    def prepare(self, img, d_over_r0, rng):
        """CPU-only randomness, shared identically by CPU and GPU rendering."""
        img = np.asarray(img, dtype=np.float32)
        if img.ndim != 2 or img.shape[0] != img.shape[1] or len(img)<2:
            raise ValueError("D3 expects square 2-D image, size >=2")
        a = self.coefficient_field(len(img), d_over_r0, rng)
        shift = (float(a[1].mean()*self.wf.px_per_rad),
                 float(a[0].mean()*self.wf.px_per_rad))
        return a, shift

    def renderer(self, device="cpu"):
        from .exact_render import ExactRenderer
        key = str(device)
        if not hasattr(self, "_renderers"):
            self._renderers = {}
        if key not in self._renderers:
            self._renderers[key] = ExactRenderer(self.wf, device)
        return self._renderers[key]

    def __call__(self, img, d_over_r0, rng):
        a, shift = self.prepare(img, d_over_r0, rng)
        out = self.renderer()(img, a).cpu().numpy()
        return out, shift
