"""Как выглядит каждый уровень лестницы на всём диапазоне D/r0.

Две картинки, обе строки = уровни, столбцы = D/r0.

  levels_scene.png  реальный тайл. Кадры берутся из DegradedPairs, то есть
                    это ровно то, что увидит сеть, вместе с выравниванием
                    таргета. Первый столбец — сам таргет при D/r0 = максимум:
                    если выравнивание сломано, содержимое уедет относительно
                    соседнего столбца, и это видно глазом.

  levels_dots.png   та же сетка на решётке точек. Точка проявляет PSF:
                    у D0 круглый блин, у D1 тот же блин, сдвинутый ОДИНАКОВО
                    по всему кадру, у D1.5 гладкое несимметричное пятно,
                    у D2 спекл, у D3 тот же спекл, но сдвиги в разных местах
                    кадра РАЗНЫЕ — это и есть анизопланатизм, единственное
                    отличие D3 от D2. Красные крестики стоят в исходных узлах
                    решётки, по ним сдвиг виден глазом.

Клетки точечной сетки нормированы каждая на свой максимум и показаны с гаммой
0.5: без этого при D/r0 = 5 спекл тонет. Сцена показана как есть, без гаммы.

Запуск: python -m scripts.preview_levels [--tile N]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data import DegradedPairs, split_indices
from src.distortion import LEVELS

TILES_DIR = "data/tiles"          # тот же литерал, что в scripts/make_tiles.py
OUT_DIR = Path("experiments/preview")
D_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
DOT_STEP = 32                     # шаг решётки точек, px
DOT_SEED = 20260914               # одна реализация на все уровни и все D/r0


def grid(rows, row_labels, col_labels, path, title, marks=None):
    nr, nc = len(rows), len(rows[0])
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc, 1.8 * nr), squeeze=False)
    for i in range(nr):
        for j in range(nc):
            ax = axes[i][j]
            ax.imshow(rows[i][j], cmap="gray", vmin=0.0, vmax=1.0,
                      interpolation="nearest")
            if marks is not None:
                ax.plot(marks[0], marks[1], "+", color="red", ms=4, mew=0.4)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_labels[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tile", type=int, default=0, help="номер примера в train-сплите")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fwhm = cfg["degradation"]["diffraction_fwhm_px"]
    n, m = cfg["data"]["crop_px"], cfg["data"]["margin_px"]
    names = sorted(LEVELS)

    train_idx, _, _ = split_indices(TILES_DIR, cfg["data"]["val_frac"],
                                    cfg["data"]["test_frac"], cfg["split_seed"])

    # Решётка точек. Вход помечен read-only: уровень, пишущий по своему входу,
    # упадёт здесь, а не испортит вторую строку картинки молча.
    dots_src = np.zeros((n + 2 * m, n + 2 * m), np.float32)
    k0 = m + DOT_STEP // 2          # решётка привязана к кропу, а не к полю с маржой
    dots_src[k0::DOT_STEP, k0::DOT_STEP] = 1.0
    dots_src.flags.writeable = False
    sl = slice(m, m + n)

    scene_rows, dots_rows = [], []
    for name in names:
        lvl = LEVELS[name](fwhm)          # один экземпляр на строку: базис Зернике
        scene, dots, target = [], [], None
        for d in D_VALUES:
            ds = DegradedPairs(TILES_DIR, train_idx, lvl, margin_px=m, crop_px=n,
                               d_over_r0_range=(d, d), seed=cfg["seed"])
            ds.set_epoch(0)
            degraded, clean, _ = ds[args.tile]
            scene.append(degraded[0])
            target = clean[0]

            out, _ = lvl(dots_src, d, np.random.default_rng(DOT_SEED))
            out = out[sl, sl]
            dots.append(np.sqrt(np.clip(out, 0.0, None) / out.max()))

        scene_rows.append([target] + scene)
        dots_rows.append([dots_src[sl, sl]] + dots)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [f"D/r0 = {d:g}" for d in D_VALUES]
    grid(scene_rows, names, ["таргет"] + cols, OUT_DIR / "levels_scene.png",
         f"тайл {args.tile}: вход сети по уровням и силе турбулентности")
    grid(dots_rows, names, ["точки"] + cols, OUT_DIR / "levels_dots.png",
         "PSF на решётке точек (клетки нормированы, гамма 0.5)",
         marks=[a.ravel() for a in np.meshgrid(*(np.arange(DOT_STEP // 2, n,
                                                            DOT_STEP),) * 2)])


if __name__ == "__main__":
    main()
