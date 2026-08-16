import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Degradation
from ..optics import width_sigma


class D0Gaussian(Degradation):
    name = "d0"

    def __call__(self, img, d_over_r0, rng):
        '''Тут rng не нужен, но этоя для единообразия'''
        sigma = width_sigma(d_over_r0, self.diffraction_fwhm_px)
        sigma_axes = (sigma, sigma, 0) if img.ndim == 3 else sigma
        out = gaussian_filter(img, sigma=sigma_axes, mode="reflect")
        return np.clip(out, 0.0, 1.0).astype(np.float32)
