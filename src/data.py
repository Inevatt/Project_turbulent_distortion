"""Конвейер пар «испорченная / чистая».

Источник данных — нарезанные тайлы из scripts/make_tiles.py: сплошной
массив uint8 формы (N, TILE, TILE), открываемый через mmap_mode='r'.
Кэшированием занимается page cache ядра: он общий для всех воркеров,
страницы read-only, вытеснение бесплатное. Своего кэша здесь нет.

Посев случайности идёт ОТ НОМЕРА примера, а не от порядка вызовов.
Каждый пример получает два независимых генератора: один выбирает кроп,
другой разыгрывает искажение. Поэтому все четыре уровня видят одни
и те же куски одних и тех же картинок, сколько бы случайных чисел
ни потратил конкретный генератор искажения.
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def _rng(*key):
    """Детерминированный генератор от целочисленного ключа.
    SeedSequence разворачивает последовательность в состояние PCG64,
    поэтому близкие ключи дают некоррелированные потоки."""
    return np.random.default_rng([int(k) for k in key])


def random_crop(tile, size, rng):
    h, w = tile.shape
    if h < size or w < size:
        raise ValueError(f"тайл {h}x{w} меньше кропа {size}")
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return tile[y : y + size, x : x + size]


def split_indices(tiles_dir, val_frac=0.05, test_frac=0.10, seed=42):
    """Разбиение ПО ИСХОДНЫМ ИЗОБРАЖЕНИЯМ. По тайлам нельзя: соседние тайлы
    одной фотографии почти одинаковы, тест окажется утечкой обучающей выборки.

    Возвращает три массива индексов тайлов: train, val, test."""
    source_id = np.load(Path(tiles_dir) / "source_id.npy")
    sources = np.unique(source_id)

    order = np.random.default_rng(seed).permutation(len(sources))
    n_test = int(len(sources) * test_frac)
    n_val = int(len(sources) * val_frac)
    take = lambda ii: np.flatnonzero(np.isin(source_id, sources[ii]))

    return (
        take(order[n_test + n_val :]),
        take(order[n_test : n_test + n_val]),
        take(order[:n_test]),
    )


class DegradedPairs(Dataset):
    """Пары генерируются на лету. Эпоха задаётся снаружи через set_epoch()."""

    def __init__(
        self,
        tiles_dir,
        indices,
        degradation,
        crop_px=128,
        d_over_r0_range=(1.0, 5.0),
        seed=1337,
        samples_per_tile=1,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.deg = degradation
        self.crop_px = int(crop_px)
        self.d_lo, self.d_hi = d_over_r0_range
        self.seed = int(seed)
        self.samples_per_tile = int(samples_per_tile)
        self.epoch = 0
        self.tiles = np.load(Path(tiles_dir) / "tiles.npy", mmap_mode="r")


    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices) * self.samples_per_tile

    def __getitem__(self, idx):
        tile = self.tiles[self.indices[idx % len(self.indices)]]

        # поток 1: геометрия выборки — одинаков на всех уровнях
        rng_crop = _rng(self.seed, self.epoch, idx, 1)
        clean = random_crop(tile, self.crop_px, rng_crop)
        clean = np.asarray(clean, dtype=np.float32) / 255.0  # копия из memmap
        d_over_r0 = float(rng_crop.uniform(self.d_lo, self.d_hi))

        # поток 2: реализация искажения — своя на каждом уровне
        rng_deg = _rng(self.seed, self.epoch, idx, 2)
        degraded = self.deg(clean, d_over_r0, rng_deg)

        return (
            np.expand_dims(degraded, axis=0).astype(np.float32),
            np.expand_dims(clean, axis=0).astype(np.float32),
            np.float32(d_over_r0),
        )


class FrozenDegraded(DegradedPairs):
    """Валидация и тест: эпоха не меняется, набор заморожен навсегда."""

    def set_epoch(self, epoch):
        pass
