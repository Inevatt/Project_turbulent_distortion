import numpy as np

# FWHM = 2*sqrt(2*ln2) * sigma
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # ≈ 0.4247
TRUNCATE = 4.0   # зафиксировано явно, а не унаследовано от дефолта scipy

def width_fwhm_px(d_over_r0, diffraction_fwhm_px):
    """Ширина турбулентного пятна на половине высоты, в пикселях.

    Идеальное пятно ~ lambda/D, турбулентное ~ lambda/r0.
    Отношение равно D/r0.
    """
    return diffraction_fwhm_px * np.hypot(1.0, d_over_r0)


def width_sigma(d_over_r0, diffraction_fwhm_px):
    """То же самое в единицах, которые принимает гауссов фильтр."""
    return width_fwhm_px(d_over_r0, diffraction_fwhm_px) * FWHM_TO_SIGMA


def gaussian_radius_px(sigma, truncate=TRUNCATE):
    """Полуширина ядра, которую построит scipy.

    gaussian_filter берёт radius = int(truncate*sigma + 0.5). Формула
    воспроизведена здесь, чтобы бюджет носителя проверялся до запуска,
    а не обнаруживался по краевым артефактам после обучения.
    """
    return int(truncate * sigma + 0.5)


