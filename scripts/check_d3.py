"""Приёмка D3. Запуск из корня проекта: python3 -m scripts.check_d3 [N]

Эталоны: форма C0 из интегралов Chimitt & Chan, сила дрожания из Нолля,
средняя PSF из уже принятой D2. N — число реализаций, по умолчанию 300.
"""

import sys
import time

import numpy as np
import yaml

from src.distortion import LEVELS
from src.distortion.d3 import (SPLINE_REACH_PX, TILT_CORR_HALF_PX,
                               D3Anisoplanatic)
from src.distortion.wavefront import PSF_RADIUS_PX

CFG = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
F0 = CFG["degradation"]["diffraction_fwhm_px"]
MARGIN, CROP = CFG["data"]["margin_px"], CFG["data"]["crop_px"]
DMAX = CFG["data"]["d_over_r0_range"][1]
BIG = CROP + 2 * MARGIN


def fields(d3, d_over_r0, n, seed=0):
    """n реализаций поля тем же кодом, которым его строит сам уровень."""
    rng = np.random.default_rng(seed)
    return np.stack([d3.field(BIG, d_over_r0, rng) for _ in range(n)])


def mean_psf(level, n, d_over_r0, seed=5):
    """Средняя по реализациям локальная PSF: отклик на дельту в центре."""
    img = np.zeros((BIG, BIG), dtype=np.float32)
    img[BIG // 2, BIG // 2] = 1.0
    img.flags.writeable = False
    rng = np.random.default_rng(seed)
    acc = np.zeros((BIG, BIG))
    for _ in range(n):
        acc += level(img, d_over_r0, rng)[0]
    c, r = BIG // 2, 48
    return acc[c - r:c + r + 1, c - r:c + r + 1] / n


def otf_bands(psf):
    h = np.abs(np.fft.fft2(np.fft.ifftshift(psf)))
    k = np.fft.fftfreq(len(psf))
    nu = np.hypot(*np.meshgrid(k, k)).ravel()
    h = h.ravel() / h.ravel()[0]
    return np.array([h[(nu > a) & (nu <= b)].mean()
                     for a, b in ((0.02, 0.06), (0.06, 0.12), (0.12, 0.2), (0.2, 0.3))])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    d3 = LEVELS["d3"](F0)
    half_s = np.interp(0.5, d3.c_tab[::-1], d3.s_tab[::-1])
    print(f"кадр {BIG}, кроп {CROP}, длина полукорреляции "
          f"{TILT_CORR_HALF_PX:g} px (C0 = 0.5 при s = {half_s:.2f}, "
          f"{d3.px_per_s:.2f} px на диаметр)")

    # 1. Форма реализованной корреляции против C0 -------------------------
    f = fields(d3, DMAX, n)[:, 0]
    v = float((f ** 2).mean())
    print(f"\n1. корреляция поля: измерено против C0(lag/{d3.px_per_s:.2f})")
    for lag in (1, 2, 4, 8, 16, 32, 64, 96):
        got = float((f[:, :, :-lag] * f[:, :, lag:]).mean()) / v
        want = float(np.interp(lag / d3.px_per_s, d3.s_tab, d3.c_tab))
        print(f"   lag {lag:3d} px: {got:+.4f} против {want:+.4f} "
              f"({(got - want) * 100:+.1f} п.п.)")

    # 2. Сила дрожания против Нолля ---------------------------------------
    print("\n2. поточечная сигма поля против ноллевской (та же, что у D1 и D2)")
    print("   (нормировка спектра аналитическая, поэтому равенство должно")
    print("    держаться с точностью выборки, около 3% при таком числе реализаций)")
    for i, d in enumerate((1.0, 3.0, 5.0)):
        ff = fields(d3, d, max(40, n // 4), seed=100 + i)
        got = float(np.sqrt((ff.astype(np.float64) ** 2).mean()))
        want = d3.wf.tilt_sigma_px(d)
        print(f"   D/r0={d:g}: измерено {got:.4f} px, "
              f"Нолль {want:.4f} px, откл {got / want - 1:+.3%}")

    # 3. Какая доля дисперсии остаётся локальной --------------------------
    print("\n3. доля локальной дисперсии после снятия среднего по кропу")
    lo, hi = MARGIN, MARGIN + CROP
    dxy = np.arange(-(CROP - 1), CROP)
    wt = (CROP - np.abs(dxy))
    rr = np.hypot(*np.meshgrid(dxy, dxy))
    analytic = 1.0 - (np.outer(wt, wt) *
                      np.interp(rr / d3.px_per_s, d3.s_tab, d3.c_tab)).sum() / CROP ** 4
    crop = f[:, lo:hi, lo:hi]
    local = crop - crop.mean(axis=(1, 2), keepdims=True)
    print(f"   измерено {float((local ** 2).mean()) / v:.3f}, "
          f"аналитика {analytic:.3f}")
    print(f"   локальная сигма при D/r0={DMAX:g}: "
          f"{np.sqrt(float((local ** 2).mean())):.2f} px "
          f"(полная {d3.wf.tilt_sigma_px(DMAX):.2f} px)")

    # 4. Вырождение в D2 --------------------------------------------------
    print("\n4. длина корреляции -> бесконечность: D3 обязан стать D2")
    far = D3Anisoplanatic(F0, corr_half_px=1.0e6)
    ff = fields(far, DMAX, 40, seed=2)[:, 0, lo:hi, lo:hi]
    spread = float((ff - ff.mean(axis=(1, 2), keepdims=True)).std())
    print(f"   разброс поля внутри кропа: {spread:.4f} px "
          f"(полная сигма {d3.wf.tilt_sigma_px(DMAX):.2f} px)")
    m2, m3 = mean_psf(LEVELS["d2"](F0), n, DMAX), mean_psf(far, n, DMAX)
    print("   средняя PSF, полосы OTF: "
          + "  ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(otf_bands(m2), otf_bands(m3))))

    # 5. Калибровка: средняя локальная PSF совпадает с D2 ------------------
    print("\n5. средняя локальная PSF D3 против D2 (тот же D/r0 значит то же)")
    for d in (1.0, 5.0):
        b2 = otf_bands(mean_psf(LEVELS["d2"](F0), n, d))
        b3 = otf_bands(mean_psf(d3, n, d))
        print(f"   D/r0={d:g}: D2 " + " ".join(f"{v:.4f}" for v in b2))
        print(f"            D3 " + " ".join(f"{v:.4f}" for v in b3)
              + "   откл " + " ".join(f"{(a/b-1)*100:+.0f}%" if b > 0.05 else "   шум"
                                      for a, b in zip(b3, b2)))

    # 6. Бюджет margin ----------------------------------------------------
    print("\n6. носители и бюджет")
    need = {k: c(F0).support_radius_px(DMAX) for k, c in LEVELS.items()}
    worst = max(need.values())
    print(f"   {need}, margin_px = {MARGIN}"
          + ("  ок" if worst <= MARGIN else "  НЕ ВЛЕЗАЕТ"))
    print(f"   разбор d3: ядро {PSF_RADIUS_PX} + хвост сплайна {SPLINE_REACH_PX} "
          f"+ поле {int(np.ceil(4 * d3.wf.tilt_sigma_px(DMAX)))} = {need['d3']}")

    # 7. Цена -------------------------------------------------------------
    print("\n7. время одного вызова на кадре", BIG)
    img = np.asarray(np.random.default_rng(0).random((BIG, BIG)), dtype=np.float32)
    img.flags.writeable = False
    for name in sorted(LEVELS):
        lvl = LEVELS[name](F0)
        rng = np.random.default_rng(0)
        lvl(img, 3.0, rng)
        t0 = time.time()
        for _ in range(20):
            lvl(img, float(rng.uniform(1.0, DMAX)), rng)
        print(f"   {name}: {(time.time() - t0) / 20 * 1e3:6.2f} мс")


if __name__ == "__main__":
    main()
