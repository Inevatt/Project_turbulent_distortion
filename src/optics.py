import numpy as np

# FWHM = 2*sqrt(2*ln2) * sigma
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # ≈ 0.4247


def width_fwhm_px(d_over_r0, diffraction_fwhm_px):
    """Ширина турбулентного пятна на половине высоты, в пикселях.

    Идеальное пятно ~ lambda/D, турбулентное ~ lambda/r0.
    Отношение равно D/r0.
    """
    return diffraction_fwhm_px * d_over_r0


def width_sigma(d_over_r0, diffraction_fwhm_px):
    """То же самое в единицах, которые принимает гауссов фильтр."""
    return width_fwhm_px(d_over_r0, diffraction_fwhm_px) * FWHM_TO_SIGMA
