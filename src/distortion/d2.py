"""Isoplanatic finite-mode Kolmogorov model, with the same T-then-B
rendering convention as D3. One Noll j=2..36 vector per image.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from .base import Degradation
from .wavefront import Kolmogorov, PSF_RADIUS_PX
from ..optics import TILT_CLIP

class D2Kolmogorov(Degradation):
    name = 'd2'

    def __init__(self,diffraction_fwhm_px):
        super().__init__(diffraction_fwhm_px)
        self.wf=Kolmogorov(diffraction_fwhm_px)

    def support_radius_px(self,d_over_r0):
        return PSF_RADIUS_PX+1+int(np.ceil(TILT_CLIP*self.wf.tilt_sigma_px(d_over_r0)))

    def _render_coeffs(self,img,a):
        tilt=tuple(float(v) for v in self.wf.shift_px(a))
        high=a.copy();high[:2]=0
        kernel=self.wf.psf(high,PSF_RADIUS_PX)
        yy,xx=np.mgrid[:img.shape[0],:img.shape[1]]
        # Backward bilinear warp, matching D3's align_corners=True convention.
        warped=map_coordinates(img,[yy-tilt[0],xx-tilt[1]],order=1,mode='mirror')
        out=fftconvolve(np.pad(warped,PSF_RADIUS_PX,mode='symmetric'),kernel,mode='valid')
        return np.clip(out,0,1).astype(np.float32),tilt

    def __call__(self,img,d_over_r0,rng):
        return self._render_coeffs(img,self.wf.coeffs(d_over_r0,rng))
