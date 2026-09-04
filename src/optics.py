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

def kernel_truncate(sigma, radius_px):
    """truncate для scipy, при котором ядро не выйдет за radius_px.

    scipy берёт полуширину как int(truncate*sigma + 0.5), дефолт
    truncate=4.0. При sigma=4.25 (D/r0=5) это 17 px против запаса 16
    в data.py: крайний пиксель кропа занял бы один пиксель зеркала.
    Ограничение сверху делает утечку границы нулевой, а не
    пренебрежимо малой, и задаёт бюджет носителя ядра, общий
    для всех четырёх уровней: PSF в D2 обязан уложиться в тот же
    radius_px, иначе краевая подпись у уровней разойдётся.
    """
    return min(4.0, radius_px / sigma)
