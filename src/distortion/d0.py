import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Degradation
from ..optics import width_sigma, kernel_truncate


class D0Gaussian(Degradation):
    name = "d0"

    def __call__(self, img, d_over_r0, rng):
        """rng не нужен: длинная экспозиция — детерминированная свёртка.
        Аргумент есть ради общего контракта, поток 2 всё равно изолирован."""
        sigma = width_sigma(d_over_r0, self.diffraction_fwhm_px)
        out = gaussian_filter(img, sigma=sigma, mode="reflect",
                              truncate=kernel_truncate(sigma, self.margin_px))
        return np.clip(out, 0.0, 1.0).astype(np.float32)
