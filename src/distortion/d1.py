import numpy as np
from scipy.ndimage import convolve1d

from .base import Degradation
from ..optics import sigma_pair, gaussian_radius_px, TILT_CLIP


def _radius(sigma, shift_px):
    """Полуширина ядра: носитель гауссианы плюс смещение её центра.

    Одна формула на два применения. В __call__ подставляется фактический
    сдвиг (ядро выходит вдвое короче худшего случая), в support_radius_px —
    граница обрезки, потому что margin обязан покрыть худший случай.
    """
    return gaussian_radius_px(sigma) + int(np.ceil(abs(shift_px)))


def _kernel(sigma, shift_px):
    """Гауссиана с центром в shift_px, нормированная.

    Сдвиг вложен в само ядро, а не сделан отдельным ndimage.shift:
    интерполяция добавляет собственное размытие, зависящее от величины
    сдвига, и инвариант «среднее по кадрам = D0» ломается молча.
    Размытие и сдвиг — обе свёртки, значит это одна свёртка.
    """
    r = _radius(sigma, shift_px)
    k = np.exp(-0.5 * ((np.arange(-r, r + 1) - shift_px) / sigma) ** 2)
    return k / k.sum()


class D1TipTilt(Degradation):
    """Одна реализация наклона на кадр, остальное усреднено.

    Бюджет ширины тот же, что у D0: часть дисперсии усреднённого пятна
    вынута из размытия и разыграна явным сдвигом всего кадра. Дисперсии
    независимых смещений складываются, поэтому

        sigma_se**2 + sigma_tilt**2 == width_sigma(D/r0)**2

    и в непрерывной гауссовой модели усреднение D1 даёт D0. Для гауссиан это
    тождество, а не приближение: свёртка гауссиан есть гауссиана,
    оси независимы. Деление задано в optics.sigma_pair.

    Это гауссова аппроксимация с явно выделенным глобальным наклоном.
    После целочисленной регистрации target остаются более узкий blur и
    субпиксельный остаток tilt. D1 не проверяет восстановление глобального
    сдвига из одного изображения. Равенство ширины/MTF с физическими D2/D3
    не гарантируется: совпадает масштаб, а не вся статистика PSF.
    """

    name = "d1"

    def support_radius_px(self, d_over_r0):
        sigma, sigma_tilt = sigma_pair(d_over_r0, self.diffraction_fwhm_px)
        return _radius(sigma, TILT_CLIP * sigma_tilt)

    def __call__(self, img, d_over_r0, rng):
        sigma, sigma_tilt = sigma_pair(d_over_r0, self.diffraction_fwhm_px)

        # Обрезка нужна не ради инварианта, а ради margin: выброс за 4σ
        # притащил бы в кадр краевую подпись массива. Дефицит дисперсии
        # от обрезки 0.1%, на инвариант это не влияет.
        lim = TILT_CLIP * sigma_tilt
        shift = np.clip(rng.normal(0.0, sigma_tilt, size=2), -lim, lim)

        out = img
        for axis, s in enumerate(shift):
            out = convolve1d(out, _kernel(sigma, s), axis=axis, mode="reflect")

        return out.astype(np.float32), (shift[0], shift[1])
