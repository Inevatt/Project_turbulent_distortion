"""Finite Kolmogorov phase model in radians, Noll j=1..36 (piston omitted).

PSF = |FFT(pupil * exp(i phase))|**2. This is a finite-mode, finite-support
approximation, not an exact infinite-mode Fried OTF. Pixel scale lambda/D is
set by optics.lambda_over_d_px. Positive a2 moves an impulse towards +x.
"""

import math

import numpy as np

from ..optics import TILT_CLIP, lambda_over_d_px

K_MODES = 36          # моды Нолля 1..36, как в Chan/P2S dense-field simulator
N_GRID = 128          # сетка FFT, в которую вписан зрачок
PSF_RADIUS_PX = 19    # полуширина вырезаемого ядра
RAD_TO_LOD = 2.0 / np.pi   # a_2 [рад] -> сдвиг в единицах lambda/D


def noll_nm(j):
    """Индекс Нолля -> (n, m). Чётное j — косинус, нечётное — синус."""
    n, rest = 0, j - 1
    while rest > n:
        n += 1
        rest -= n
    m = (-1) ** j * ((n % 2) + 2 * int((rest + ((n + 1) % 2)) / 2.0))
    return n, m


def zernike(j, rho, theta):
    """Полином Цернике в нормировке Нолля: среднее Z_j^2 по диску равно 1."""
    n, m = noll_nm(j)
    am = abs(m)
    rad = np.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        rad += (-1) ** s * math.factorial(n - s) / (
            math.factorial(s)
            * math.factorial((n + am) // 2 - s)
            * math.factorial((n - am) // 2 - s)
        ) * rho ** (n - 2 * s)
    if m == 0:
        return np.sqrt(n + 1.0) * rad
    harm = np.cos(am * theta) if m > 0 else np.sin(am * theta)
    return np.sqrt(2.0 * (n + 1.0)) * rad * harm


def noll_cov(k_modes):
    """Ковариация коэффициентов Цернике при D/r0 = 1, индексы Нолля 1..K.

    Noll 1976; форма записи взята из референсной реализации авторов
    (turbStats._nollCovMat). Ненулевые элементы только у мод с равным
    |m| и одинаковой чётностью индекса — это и есть межмодовая
    корреляция, вынесенная в заголовок Chimitt & Chan 2020.
    Пистон (j = 1) не трогаем: на PSF он не влияет, дальше отбрасывается.
    """
    c = np.zeros((k_modes, k_modes))
    for i in range(1, k_modes + 1):
        ni, mi = noll_nm(i)
        for j in range(1, k_modes + 1):
            nj, mj = noll_nm(j)
            if abs(mi) != abs(mj) or (mi != 0 and (i - j) % 2):
                continue
            num = math.gamma(14.0 / 3.0) * math.gamma((ni + nj - 5.0 / 3.0) / 2.0)
            den = (math.gamma((-ni + nj + 17.0 / 3.0) / 2.0)
                   * math.gamma((ni - nj + 17.0 / 3.0) / 2.0)
                   * math.gamma((ni + nj + 23.0 / 3.0) / 2.0))
            c[i - 1, j - 1] = (0.0072 * np.pi ** (8.0 / 3.0)
                               * np.sqrt((ni + 1.0) * (nj + 1.0))
                               * (-1.0) ** ((ni + nj - 2 * abs(mi)) / 2.0)
                               * num / den)
    c[0, :] = 0.0
    c[:, 0] = 0.0  # piston variance diverges; piston has no image effect
    return c


def centroid_px(kernel):
    """Centroid (dy,dx) for diagnostics, not the target-registration rule."""
    ax = np.arange(len(kernel)) - len(kernel) // 2
    return float(kernel.sum(1) @ ax), float(kernel.sum(0) @ ax)


class Kolmogorov:
    """Генератор фазы и PSF. Строится один раз на уровень, дальше только вызовы."""

    def __init__(self, diffraction_fwhm_px, k_modes=K_MODES, n_grid=N_GRID):
        self.n_grid = int(n_grid)
        self.px_per_rad = RAD_TO_LOD * lambda_over_d_px(diffraction_fwhm_px)

        # Диаметр зрачка в ячейках сетки задаётся масштабом изображения:
        # lambda/D в пикселях есть отношение сетки FFT к зрачку. Целым
        # его делать незачем, маска — просто круг радиуса d/2.
        d_cells = self.n_grid / lambda_over_d_px(diffraction_fwhm_px)
        ax = np.arange(self.n_grid) - self.n_grid / 2.0
        x, y = np.meshgrid(ax, ax)
        rho = np.hypot(x, y) / (d_cells / 2.0)
        self.mask = rho <= 1.0
        theta = np.arctan2(y, x)[self.mask]
        rho = rho[self.mask]

        # Пистон выброшен, поэтому индекс 0 массива — это j = 2, наклон по x.
        self.basis = np.stack(
            [zernike(j, rho, theta) for j in range(2, k_modes + 1)])
        cov = noll_cov(k_modes)[1:, 1:]
        w, v = np.linalg.eigh(cov)
        self.root = v * np.sqrt(np.maximum(w, 0.0))
        self.sigma_tilt_rad = np.sqrt(cov[0, 0])   # при D/r0 = 1

    def coeffs(self, d_over_r0, rng):
        """Одна реализация коэффициентов. Тратит ровно K-1 случайных чисел."""
        a = (self.root @ rng.normal(size=len(self.root))) * d_over_r0 ** (5.0 / 6.0)
        # Обрезка наклона нужна не ради статистики (дефицит дисперсии
        # 0.01%), а ради margin: выброс за 4 сигмы утащил бы окно ядра
        # за пределы кропа с запасом. Та же обрезка и тем же TILT_CLIP,
        # что в D1.
        lim = TILT_CLIP * self.sigma_tilt_rad * d_over_r0 ** (5.0 / 6.0)
        a[:2] = np.clip(a[:2], -lim, lim)
        return a

    def tilt_sigma_px(self, d_over_r0):
        """Сигма дрожания кадра по одной оси, px."""
        return self.px_per_rad * self.sigma_tilt_rad * d_over_r0 ** (5.0 / 6.0)

    def shift_px(self, a):
        """Сдвиг кадра (dy, dx) из наклонных мод. a_2 — по x, a_3 — по y."""
        return self.px_per_rad * a[1], self.px_per_rad * a[0]

    def psf(self, a, radius, sigma_px = 0.0):
        """Ядро (2r+1, 2r+1), сумма 1, центр реализации сдвинут на shift_px(a).

        Наклон оставлен внутри фазы: FFT сдвигает пятно на сетке ТОЧНО,
        без интерполяции, поэтому дробная часть сдвига попадает в ядро
        даром. Отдельный ndimage.shift добавил бы своё размытие,
        зависящее от величины сдвига — ровно то, от чего отказались в D1.
        """
        u = np.zeros((self.n_grid, self.n_grid), dtype=np.complex128)
        u[self.mask] = np.exp(1j * (a @ self.basis))
        h = np.abs(np.fft.fft2(u)) ** 2
        if sigma_px:
            f2 = np.fft.fftfreq(self.n_grid) ** 2
            g = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (f2[:, None] + f2))
            h = np.fft.ifft2(np.fft.fft2(h) * g).real
        h = np.fft.fftshift(h)
        c = self.n_grid // 2
        h = h[c - radius:c + radius + 1, c - radius:c + radius + 1]
        return h / h.sum()
