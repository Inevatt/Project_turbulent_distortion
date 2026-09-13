import numpy as np

# FWHM = 2*sqrt(2*ln2) * sigma
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # ≈ 0.4247
TRUNCATE = 4.0   # зафиксировано явно, а не унаследовано от дефолта scipy
TILT_SHARE = 0.9    # доля потолка σ²_atm, уходящая в дрожание при D/r₀ = 1
TILT_CLIP  = 4.0    # обрезка сдвига, в сигмах

def width_fwhm_px(d_over_r0, diffraction_fwhm_px):
    """Ширина турбулентного пятна на половине высоты, в пикселях.

    Идеальное пятно ~ lambda/D, турбулентное ~ lambda/r0.
    Отношение равно D/r0.
    """
    return diffraction_fwhm_px * np.hypot(1.0, d_over_r0)


def width_sigma(d_over_r0, diffraction_fwhm_px):
    """То же самое в единицах, которые принимает гауссов фильтр."""
    return width_fwhm_px(d_over_r0, diffraction_fwhm_px) * FWHM_TO_SIGMA

def sigma_pair(d_over_r0, diffraction_fwhm_px):
    """Разделение усреднённого пятна на остаточное размытие и дрожание кадра.
 
    Возвращает (sigma_se, sigma_tilt), px. Тождественно выполняется
        sigma_se**2 + sigma_tilt**2 == width_sigma(...)**2
    потому что оба слагаемых берутся из того же разложения hypot(1, D/r0),
    что и width_sigma: под корнем стоит 1 + (D/r0)**2, то есть
    дисперсия усреднённого пятна есть сумма диффракционной и атмосферной.
 
    Делить можно только атмосферную часть: диффракционное пятно есть и
    в вакууме, оно не дрожит, и забрав из него, мы объявили бы один кадр
    резче диффракционного предела. Поэтому v_atm — потолок для дрожания,
    а alpha — насколько близко к этому потолку:
 
        alpha = TILT_SHARE * (D/r0)**(-1/3)
 
    Показатель -1/3 — разность между ростом дисперсии усреднённого пятна
    (D/r0)**2 и ростом дисперсии угла прихода (D/r0)**(5/3). Дрожание
    задаётся СРЕДНИМ наклоном фронта по апертуре, а чем шире апертура
    в единицах r0, тем больше независимых пятен попадает в это среднее
    и тем сильнее оно гасит разброс.
 
    ВАЖНО: alpha <= 1 только при d_over_r0 >= 1. Ниже единицы sigma_se
    молча провалится под диффракционный предел, без NaN и без падения.
    data.d_over_r0_range начинается с 1.0.
    """
    v_diff = (diffraction_fwhm_px * FWHM_TO_SIGMA) ** 2
    v_atm = v_diff * d_over_r0 ** 2
    alpha = TILT_SHARE * d_over_r0 ** (-1.0 / 3.0)
    return np.sqrt(v_diff + (1.0 - alpha) * v_atm), np.sqrt(alpha * v_atm)


def gaussian_radius_px(sigma, truncate=TRUNCATE):
    """Полуширина ядра, которую построит scipy.

    gaussian_filter берёт radius = int(truncate*sigma + 0.5). Формула
    воспроизведена здесь, чтобы бюджет носителя проверялся до запуска,
    а не обнаруживался по краевым артефактам после обучения.
    """
    return int(truncate * sigma + 0.5)


