"""Приёмка D2. Запуск из корня проекта: python3 -m scripts.check_d2 [N]

Ничего не подгоняет: все эталоны — либо таблица Нолля, либо замкнутая
формула Фрида, либо контракт data.py. N — число реализаций в проверке 4,
по умолчанию 1000 (около 20 с).
"""

import sys
import time

import numpy as np
import yaml

from src.distortion import LEVELS
from src.distortion.wavefront import (K_MODES, N_GRID, PSF_RADIUS_PX,
                                      Kolmogorov, centroid_px, noll_cov,
                                      noll_nm)
from src.optics import TILT_CLIP, lambda_over_d_px

CFG = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
F0 = CFG["degradation"]["diffraction_fwhm_px"]
MARGIN = CFG["data"]["margin_px"]
DMAX = CFG["data"]["d_over_r0_range"][1]
LOD = lambda_over_d_px(F0)


def fwhm_px(h):
    """FWHM по срезу через максимум, с линейной интерполяцией краёв."""
    i, j = np.unravel_index(h.argmax(), h.shape)
    row = h[i] / h[i, j]
    r = j + np.flatnonzero(row[j:] <= 0.5)[0]
    l = j - np.flatnonzero(row[j::-1] <= 0.5)[0]
    xr = r - 1 + (row[r - 1] - 0.5) / (row[r - 1] - row[r])
    xl = l + 1 - (row[l + 1] - 0.5) / (row[l + 1] - row[l])
    return xr - xl


def centroid(h):
    ax = np.arange(len(h)) - len(h) // 2
    return (h.sum(1) @ ax) / h.sum(), (h.sum(0) @ ax) / h.sum()


def h_le(nu, d_over_r0):
    """Длинноэкспозиционная OTF Фрида, nu = f / f_срез."""
    return np.exp(-3.44 * (nu * d_over_r0) ** (5.0 / 3.0))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wf = Kolmogorov(F0)
    d2 = LEVELS["d2"](F0)
    print(f"K = {K_MODES} (порядок {noll_nm(K_MODES)[0]}), сетка {N_GRID}, "
          f"зрачок {N_GRID / LOD:.1f} ячеек, точек в маске {wf.mask.sum()}, "
          f"lambda/D = {LOD:.4f} px")

    # 1. Базис и ковариация против Нолля -----------------------------------
    g = wf.basis @ wf.basis.T / wf.mask.sum()
    cov = noll_cov(K_MODES)[1:, 1:]
    resid = 0.2944 * K_MODES ** (-0.866)
    print("\n1. базис и статистика коэффициентов")
    print(f"   ортонормальность на сетке зрачка: max|G-I| = "
          f"{np.abs(g - np.eye(len(g))).max():.4f}")
    print(f"   <a2^2> = {cov[0, 0]:.4f}, <a3^2> = {cov[1, 1]:.4f}   Нолль: 0.4480")
    print(f"   сумма дисперсий = {np.trace(cov):.4f}   "
          f"Нолль без обрезки 1.0299 - остаток {resid:.4f} = {1.0299 - resid:.4f}")
    print(f"   минимальное собственное число R_Z = "
          f"{np.linalg.eigvalsh(cov).min():.2e} (обязано быть > 0)")

    # 2. Дифракционный предел ----------------------------------------------
    h0 = wf.psf(np.zeros(len(wf.root)), PSF_RADIUS_PX)
    print("\n2. нулевая фаза = пятно Айри")
    print(f"   FWHM = {fwhm_px(h0):.4f} px   задано diffraction_fwhm_px = {F0}")
    print(f"   центроид = ({centroid(h0)[0]:+.4f}, {centroid(h0)[1]:+.4f}) px "
          f"(смещения быть не должно)")

    # 3. Наклон в пиксели ---------------------------------------------------
    print("\n3. наклон: центроид PSF против (2/pi)*(lambda/D)*a")
    for j, a_val in ((0, 1.0), (0, 3.0), (1, 3.0)):
        a = np.zeros(len(wf.root))
        a[j] = a_val
        cy, cx = centroid(wf.psf(a, PSF_RADIUS_PX))
        want = wf.shift_px(a)
        print(f"   a_{j + 2} = {a_val}: центроид ({cy:+.4f}, {cx:+.4f}), "
              f"формула ({want[0]:+.4f}, {want[1]:+.4f})")

    # 4. Средняя OTF против H_LE -------------------------------------------
    print(f"\n4. средняя OTF полной фазы против H_LE Фрида, n = {n}")
    k = np.fft.fftfreq(N_GRID) * N_GRID
    nu = (np.hypot(*np.meshgrid(k, k)) / N_GRID * LOD).ravel()
    u = np.zeros((N_GRID, N_GRID), dtype=np.complex128)
    u[wf.mask] = 1.0
    h = np.abs(np.fft.fft2(u)) ** 2
    otf_diff = np.fft.fft2(h / h.sum()).real.ravel()
    edges = [0.05, 0.15, 0.3, 0.45, 0.6, 0.8]
    mid = np.array([(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
    bands = [(nu > a) & (nu <= b) for a, b in zip(edges[:-1], edges[1:])]
    print("   nu:        " + " ".join(f"{v:8.3f}" for v in mid))
    for d in (1.0, 2.0, 3.0, 5.0):
        rng = np.random.default_rng(7)
        acc = np.zeros(N_GRID * N_GRID)
        for _ in range(n):
            u[wf.mask] = np.exp(1j * (wf.coeffs(d, rng) @ wf.basis))
            h = np.abs(np.fft.fft2(u)) ** 2
            acc += np.fft.fft2(h / h.sum()).real.ravel()
        got = np.array([acc[b].sum() / otf_diff[b].sum() / n for b in bands])
        want = h_le(mid, d)
        rel = " ".join(f"{(a / b - 1) * 100:+7.0f}%" if b > 0.02 else "       -"
                       for a, b in zip(got, want))
        print(f"   D/r0={d}   " + " ".join(f"{v:8.4f}" for v in got))
        print(f"     H_LE     " + " ".join(f"{v:8.4f}" for v in want) + "   " + rel)

    # 5. Носитель ядра и бюджет margin --------------------------------------
    print("\n5. энергия в окне и бюджет margin_px")
    full = N_GRID // 2 - 1          # ядро без обрезки, вся сетка
    rs = (11, PSF_RADIUS_PX, 32)
    take = lambda big, r: big[full - r:full + r + 1, full - r:full + r + 1].sum()
    print("   дифракция:        " + "/".join(
        f"{take(wf.psf(np.zeros(len(wf.root)), full), r):.4f}" for r in rs)
        + "   <- крылья Айри, турбулентность ни при чём")
    rng = np.random.default_rng(11)
    for d in (1.0, 3.0, 5.0):
        frac = np.zeros(3)
        for _ in range(200):
            a = wf.coeffs(d, rng)
            a[:2] = 0.0
            big = wf.psf(a, full)
            frac += [take(big, r) for r in rs]
        need = {k: c(F0).support_radius_px(d) for k, c in LEVELS.items()}
        print(f"   D/r0={d}: в +-11/{PSF_RADIUS_PX}/32 = "
              + "/".join(f"{v:.4f}" for v in frac / 200)
              + f"   носители {need}")
    worst = max(c(F0).support_radius_px(DMAX) for c in LEVELS.values())
    print(f"   худший носитель при D/r0={DMAX}: {worst}, margin_px = {MARGIN}"
          + ("  ок" if worst <= MARGIN else "  НЕ ВЛЕЗАЕТ"))

    # 6. Контракт со сдвигом таргета ---------------------------------------
    print("\n6. заявленный сдвиг против фактического смещения содержимого")
    print("   центроид минус наклон | невязка таргета после округления, RMS px")
    for d in (1.0, 3.0, 5.0):
        rng = np.random.default_rng(3)
        bias, resid = [], []
        for _ in range(400):
            a = wf.coeffs(d, rng)
            tilt = np.array(wf.shift_px(a))
            r = PSF_RADIUS_PX + int(np.ceil(np.abs(tilt).max()))
            c = np.array(centroid_px(wf.psf(a, r)))
            bias.append(c - tilt)
            resid.append(c - np.round(c))
        bias, resid = np.array(bias), np.array(resid)
        print(f"   D/r0={d}: смещение {bias.mean(0)[0]:+.3f}/"
              f"{bias.mean(0)[1]:+.3f}, разброс {bias.std(0).mean():.3f} | "
              f"RMS {np.sqrt((resid ** 2).mean()):.3f} px "
              f"(D1: только округление, 0.289)")

    out, shift = d2(np.zeros((192, 192), dtype=np.float32), DMAX,
                    np.random.default_rng(0))
    print(f"   форма выхода {out.shape}, dtype {out.dtype}, "
          f"сдвиг {tuple(round(v, 3) for v in shift)}")

    # 7. Цена вызова --------------------------------------------------------
    print("\n7. время одного вызова на кропе 192x192")
    img = np.asarray(np.random.default_rng(0).random((192, 192)), dtype=np.float32)
    img.flags.writeable = False
    for name in sorted(LEVELS):
        lvl = LEVELS[name](F0)
        rng = np.random.default_rng(0)
        lvl(img, 3.0, rng)
        t0 = time.time()
        for _ in range(50):
            lvl(img, float(rng.uniform(1.0, DMAX)), rng)
        print(f"   {name}: {(time.time() - t0) / 50 * 1e3:6.2f} мс")


if __name__ == "__main__":
    main()
