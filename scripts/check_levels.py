"""Финальная проверка всех зарегистрированных уровней на малом подмножестве тайлов.

Запуск: python -m scripts.check_levels [--tiles 64] [--realizations 256]

Три таблицы. Первые две ловят молчаливые поломки, третья только печатает числа,
которые надо знать ДО единственного прогона, а не объяснять после.

  КОНТРАКТ    форма, dtype, конечность, детерминизм при одном и том же rng,
              фактический |сдвиг| против margin_px и support_radius_px.
              Недетерминизм — самая дорогая из молчаливых поломок: если уровень
              где-то дёрнул глобальный np.random, замороженный тест перестаёт
              быть замороженным, и клетки матрицы считаются на разных картинках.

  СДВИГ       СКО сдвига, который уровень отдаёт наружу, против sigma_tilt
              из optics.sigma_pair. Отношение обязано быть около единицы у D1
              и D2 (весь наклон уезжает в окно таргета) и заметно меньше
              у D3: там наружу идёт СРЕДНЕЕ поля по тайлу, а разница между
              единицей и этим числом и есть анизопланатический остаток —
              единственный новый эффект верхней ступени.

              Здесь НЕ проверяется среднее по реализациям против D0 картинкой.
              Проверено: при разумном числе реализаций это не работает. Ошибка
              такого среднего почти целиком определяется остаточным средним
              сдвигом, то есть двумя степенями свободы, а не десятками тысяч
              пикселей. Усреднение по пикселям её не давит: на стенде разброс
              оценки составил 10 дБ от посева к посеву, и уровень с ядром на
              10% шире положенного дал ровно то же число, что правильный.
              Ширину ядра ловят H_LE и verify_generators, а не эта проверка.

  СИЛА        PSNR(вход, таргет) по уровням и по D/r0. Числа НЕ обязаны
              совпадать между уровнями: в D1-D3 глобальный наклон уезжает
              в окно таргета, поэтому сеть видит sigma_SE, а не sigma_LE.

  СТОИМОСТЬ   мс на пример на одном CPU-ядре. Умножь на num_workers и сравни
              с потребностью GPU (batch_size / время шага): если меньше —
              прогон упрётся в генерацию данных, а не в видеокарту.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from src.data import DegradedPairs, random_crop, split_indices
from src.distortion import LEVELS
from src.metrics import psnr
from src.optics import sigma_pair

TILES_DIR = "data/tiles"          # тот же литерал, что в scripts/make_tiles.py
SEED = 20260914                   # посев только этой проверки, ни на что не влияет
D_VALUES = [1.0, 3.0, 5.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--realizations", type=int, default=256)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fwhm = cfg["degradation"]["diffraction_fwhm_px"]
    n, m = cfg["data"]["crop_px"], cfg["data"]["margin_px"]
    d_hi = cfg["data"]["d_over_r0_range"][1]
    names = sorted(LEVELS)
    levels = {k: LEVELS[k](fwhm) for k in names}

    train_idx, _, _ = split_indices(TILES_DIR, cfg["data"]["val_frac"],
                                    cfg["data"]["test_frac"], cfg["split_seed"])
    tiles = np.load(Path(TILES_DIR) / "tiles.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)
    bigs = []
    for i in rng.choice(train_idx, args.tiles, replace=False):
        big = np.asarray(random_crop(tiles[i], n + 2 * m, rng), np.float32) / 255.0
        big.flags.writeable = False     # уровень, пишущий по входу, упадёт здесь
        bigs.append(big)

    print(f"уровни: {', '.join(names)} | тайлов {args.tiles} | crop {n} | margin {m}\n")

    # ---- КОНТРАКТ -----------------------------------------------------------
    print(f"КОНТРАКТ (худший конец D/r0 = {d_hi:g})")
    print(f"{'':6}{'форма':>7}{'dtype':>9}{'конечн.':>9}{'детерм.':>9}"
          f"{'min':>8}{'max':>8}{'|сдвиг|':>9}{'support':>9}{'margin':>8}")
    verdict = []
    for k in names:
        lvl, shift, lo, hi, shape_ok, fin = levels[k], 0.0, 1e9, -1e9, True, True
        for j, big in enumerate(bigs):
            out, sh = lvl(big, d_hi, np.random.default_rng([SEED, j]))
            shape_ok &= out.shape == big.shape and out.dtype == np.float32
            fin &= bool(np.isfinite(out).all())
            lo, hi = min(lo, float(out.min())), max(hi, float(out.max()))
            shift = max(shift, abs(sh[0]), abs(sh[1]))
        a, _ = lvl(bigs[0], d_hi, np.random.default_rng([SEED, 0]))
        b, _ = lvl(bigs[0], d_hi, np.random.default_rng([SEED, 0]))
        det = np.array_equal(a, b)
        sup = lvl.support_radius_px(d_hi)
        ok = shape_ok and fin and det and round(shift) <= m and sup <= m
        verdict.append(ok)
        print(f"{k:6}{'да' if shape_ok else 'НЕТ':>7}{'f32' if shape_ok else '?':>9}"
              f"{'да' if fin else 'НЕТ':>9}{'да' if det else 'НЕТ':>9}"
              f"{lo:8.3f}{hi:8.3f}{shift:9.2f}{sup:9.1f}{m:8d}")
    print(f"  -> {'ВСЁ ОК' if all(verdict) else 'ЕСТЬ ПРОВАЛ, ЗАПУСКАТЬ НЕЛЬЗЯ'}\n")

    # ---- СДВИГ --------------------------------------------------------------
    print(f"СДВИГ НАРУЖУ: СКО по {args.realizations} реализациям, px "
          f"(в скобках отношение к sigma_tilt, точность оценки "
          f"{100.0 / np.sqrt(4.0 * args.realizations):.1f}%)")
    print(f"{'':10}" + "".join(f"{'D/r0 = ' + format(d, 'g'):>17}" for d in D_VALUES))
    expected = [sigma_pair(d, fwhm)[1] for d in D_VALUES]
    print(f"{'ожидание':10}" + "".join(f"{e:10.2f}{'':7}" for e in expected))
    img = bigs[0]
    for k in names:
        row = ""
        for d, e in zip(D_VALUES, expected):
            sh = np.array([levels[k](img, d, np.random.default_rng([SEED, 1, j]))[1]
                           for j in range(args.realizations)])
            sd = float(np.sqrt(np.mean(sh ** 2)))      # СКО на ось, сразу по обеим
            row += f"{sd:10.2f} ({sd / e:4.2f})"
        print(f"{k:10}{row}")
    print()

    # ---- СИЛА И СТОИМОСТЬ ---------------------------------------------------
    np.asarray(tiles[np.sort(train_idx[:args.tiles])]).sum()   # прогрев страниц
    print("СИЛА PSNR(вход, таргет), дБ | СТОИМОСТЬ мс/пример на ядро")
    print(f"{'':6}" + "".join(f"{'D/r0 = ' + format(d, 'g'):>21}" for d in D_VALUES))
    for k in names:
        row = ""
        for d in D_VALUES:
            ds = DegradedPairs(TILES_DIR, train_idx, levels[k], margin_px=m,
                               crop_px=n, d_over_r0_range=(d, d), seed=cfg["seed"])
            ds.set_epoch(0)
            t0 = time.perf_counter()
            batch = [ds[i] for i in range(args.tiles)]
            ms = (time.perf_counter() - t0) / args.tiles * 1e3
            val = np.mean([psnr(x[None], y[None]) for x, y, _ in batch])
            row += f"{val:12.2f}{ms:7.1f} мс"
        print(f"{k:6}{row}")


if __name__ == "__main__":
    main()
