import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Degradation
from ..optics import width_sigma, gaussian_radius_px, TRUNCATE


class D0Gaussian(Degradation):
    name = "d0"

    def support_radius_px(self, d_over_r0):
        return gaussian_radius_px(width_sigma(d_over_r0, self.diffraction_fwhm_px))

    def __call__(self, img, d_over_r0, rng):
        """rng не нужен: усреднённое пятно — детерминированная свёртка.
        Аргумент есть ради общего контракта, поток 2 всё равно изолирован."""
        sigma = width_sigma(d_over_r0, self.diffraction_fwhm_px)
        out = gaussian_filter(img, sigma=sigma, mode="reflect", truncate=TRUNCATE)
        return np.clip(out, 0.0, 1.0).astype(np.float32)


