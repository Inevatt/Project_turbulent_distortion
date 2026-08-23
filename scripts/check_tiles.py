# -*- coding: utf-8 -*-
"""Осмотр нарезанных тайлов глазами.

Запуск из корня проекта:
    python3 -m scripts.check_tiles

Печатает сводку и кладёт рядом с тайлами две картинки:
    preview_random.png — случайная выборка тайлов со всего датасета
    preview_source.png — все тайлы ОДНОЙ фотографии, собранные обратно в сетку
                         (должна читаться как исходный снимок с обрезанными краями)
"""

from pathlib import Path

import numpy as np
from PIL import Image

TILES = Path("data/tiles")
GRID = 8      # сторона сетки для preview_random
COLS = 7      # тайлов в ряду у исходной фотографии: 2040 // 256 = 7 для DIV2K
GAP = 4       # белая рамка между тайлами, чтобы видеть границы


def sheet(tiles, cols, gap=GAP):
    """Склейка списка тайлов в одно полотно с зазорами."""
    n, t = len(tiles), tiles[0].shape[0]
    rows = -(-n // cols)
    out = np.full((rows * (t + gap) - gap, cols * (t + gap) - gap), 255, np.uint8)
    for k, tile in enumerate(tiles):
        y, x = divmod(k, cols)
        out[y * (t + gap) : y * (t + gap) + t, x * (t + gap) : x * (t + gap) + t] = tile
    return Image.fromarray(out)


def main():
    tiles = np.load(TILES / "tiles.npy", mmap_mode="r")
    sid = np.load(TILES / "source_id.npy")
    n, h, w = tiles.shape

    print(f"тайлов: {n}   размер: {h}x{w}   тип: {tiles.dtype}")
    print(f"на диске: {n * h * w / 1e9:.2f} ГБ")
    print(f"исходных фотографий: {len(np.unique(sid))}")

    per = np.bincount(sid)
    per = per[per > 0]
    print(f"тайлов с фотографии: от {per.min()} до {per.max()}, в среднем {per.mean():.1f}")

    # выборочная статистика яркости: пустые или засвеченные тайлы видно сразу
    probe = np.asarray(tiles[:: max(1, n // 500)], dtype=np.float32)
    flat = int((probe.std(axis=(1, 2)) < 1.0).sum())
    print(f"яркость: среднее {probe.mean():.1f}, разброс {probe.std():.1f}, "
          f"диапазон {probe.min():.0f}..{probe.max():.0f}")
    print(f"почти однотонных тайлов в пробе: {flat} из {len(probe)}")

    # 1) случайная выборка со всего датасета
    pick = np.random.default_rng(0).choice(n, size=min(GRID * GRID, n), replace=False)
    sheet([np.asarray(tiles[i]) for i in sorted(pick)], GRID).save(TILES / "preview_random.png")

    # 2) одна фотография, собранная обратно
    src = int(np.bincount(sid).argmax())
    idx = np.flatnonzero(sid == src)
    sheet([np.asarray(tiles[i]) for i in idx], COLS).save(TILES / "preview_source.png")

    print(f"\nсохранено: {TILES / 'preview_random.png'}")
    print(f"           {TILES / 'preview_source.png'}  "
          f"(фотография {src}, {len(idx)} тайлов)")


if __name__ == "__main__":
    main()
