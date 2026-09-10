# -*- coding: utf-8 -*-
"""Нарезка исходных изображений на тайлы фиксированного размера.

Запуск из корня проекта:
    python3 -m scripts.make_tiles

Результат — два файла в data/tiles:
    tiles.npy      (N, TILE, TILE) uint8 — сами тайлы, сплошной сырой массив
    source_id.npy  (N,)            int32 — номер исходного изображения

Зачем сырой массив, а не папка PNG: из него читается срез по смещению,
без декодирования. При обучении он открывается через mmap_mode='r',
и кэшированием занимается page cache ядра — общий для всех воркеров.

Зачем source_id: разбиение train/val/test делается ПО ИСХОДНЫМ КАРТИНКАМ.
Тайлы одной фотографии почти одинаковы, разбиение по тайлам превратило бы
тест в замаскированную утечку обучающей выборки.
"""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TILE = 256
OUT = Path("data/tiles")


def list_images(root):
    """Сортировка обязательна: иначе порядок зависит от файловой системы
    и source_id поедет на другой машине."""
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"нет изображений в {root}")
    return files


def grid(h, w):
    """Левые верхние углы тайлов. Шаг равен размеру тайла: перекрытия нет,
    остаток по краям отбрасывается. Картинка меньше TILE даёт пустой список."""
    return [
        (y, x)
        for y in range(0, h - TILE + 1, TILE)
        for x in range(0, w - TILE + 1, TILE)
    ]


def main():
    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))

    need = cfg["data"]["crop_px"] + 2 * cfg["data"]["margin_px"]
    if need > TILE:
        raise ValueError(f"тайл {TILE} меньше окна {need} = crop_px + 2*margin_px")

    paths = list_images(cfg["data"]["root"])

    # проход 1: сколько всего тайлов. Image.open читает только заголовок,
    # декодирования здесь нет, поэтому проход почти бесплатный
    total = 0
    for p in paths:
        w, h = Image.open(p).size
        total += len(grid(h, w))
    if total == 0:
        raise ValueError(f"ни одна картинка не больше {TILE}x{TILE}")

    print(f"{len(paths)} изображений -> {total} тайлов "
          f"({total * TILE * TILE / 1e9:.2f} ГБ)")

        # проход 2: заполнение. open_memmap пишет прямо на диск,
    # в оперативке ни один момент не лежит больше одной картинки
    OUT.mkdir(parents=True, exist_ok=True)
    # source_id.npy пишется последним и работает маркером завершения. Старый
    # снимается здесь: иначе оборванный повторный прогон оставит новый
    # недописанный tiles.npy рядом со старым source_id, и split_indices
    # молча разложит тайлы по чужим фотографиям.
    (OUT / "source_id.npy").unlink(missing_ok=True)
    tiles = np.lib.format.open_memmap(
        OUT / "tiles.npy", mode="w+", dtype=np.uint8, shape=(total, TILE, TILE)
    )
    source_id = np.empty(total, dtype=np.int32)

    k = 0
    for i, p in enumerate(paths):
        img = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        h, w = img.shape
        for y, x in grid(h, w):
            tiles[k] = img[y : y + TILE, x : x + TILE]
            source_id[k] = i
            k += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}")

    assert k == total, f"проходы разошлись: {k} против {total}"

    tiles.flush()
    np.save(OUT / "source_id.npy", source_id)
    print(f"готово: {OUT / 'tiles.npy'}, {OUT / 'source_id.npy'}")


if __name__ == "__main__":
    main()
