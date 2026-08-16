# -*- coding: utf-8 -*-
"""Конвейер пар «испорченная / чистая».

Посев случайности идёт ОТ НОМЕРА примера, а не от порядка вызовов.
Каждый пример получает два независимых генератора: один выбирает кроп,
другой разыгрывает искажение. Поэтому все четыре уровня видят одни
и те же куски одних и тех же картинок, сколько бы случайных чисел
ни потратил конкретный генератор искажения.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root):
    """Все изображения в папке. Сортировка обязательна: иначе порядок
    зависит от файловой системы и разбиение поедет на другой машине."""
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"нет изображений в {root}")
    return files


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _rng(*key):
    """Детерминированный генератор от целочисленного ключа."""
    return np.random.default_rng([int(k) for k in key])


def random_crop(img, size, rng):
    h, w = img.shape
    if h < size or w < size:
        raise ValueError(f"изображение {h}x{w} меньше кропа {size}")
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return img[y : y + size, x : x + size]


def split_paths(paths, val_frac=0.05, test_frac=0.10, seed=42):
    """Разбиение ПО ИЗОБРАЖЕНИЯМ. По кропам нельзя: соседние кропы одной
    фотографии почти одинаковы, тест окажется утечкой обучающей выборки."""
    paths = list(paths)
    order = np.random.default_rng(seed).permutation(len(paths))
    n_test = int(len(paths) * test_frac)
    n_val = int(len(paths) * val_frac)
    take = lambda ii: [paths[i] for i in sorted(ii)]
    return (
        take(order[n_test + n_val :]),
        take(order[n_test : n_test + n_val]),
        take(order[:n_test]),
    )


class DegradedPairs(Dataset):
    """Пары генерируются на лету. Эпоха задаётся снаружи через set_epoch()."""

    def __init__(
        self,
        paths,
        degradation,
        crop_px=128,
        d_over_r0_range=(1.0, 5.0),
        seed=1337,
        samples_per_image=8,
    ):
        self.paths = list(paths)
        self.deg = degradation
        self.crop_px = int(crop_px)
        self.d_lo, self.d_hi = d_over_r0_range
        self.seed = int(seed)
        self.samples_per_image = int(samples_per_image)
        self.epoch = 0
        self._cache = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.paths) * self.samples_per_image

    def _image(self, i):
        if i not in self._cache:
            self._cache[i] = load_gray(self.paths[i])
        return self._cache[i]

    def __getitem__(self, idx):
        clean_full = self._image(idx % len(self.paths))

        # поток 1: геометрия выборки — одинаков на всех уровнях
        rng_crop = _rng(self.seed, self.epoch, idx, 1)
        clean = random_crop(clean_full, self.crop_px, rng_crop)
        d_over_r0 = float(rng_crop.uniform(self.d_lo, self.d_hi))

        # поток 2: реализация искажения — своя на каждом уровне
        rng_deg = _rng(self.seed, self.epoch, idx, 2)
        degraded = self.deg(clean, d_over_r0, rng_deg)

        return (
            degraded[None].astype(np.float32),
            clean[None].astype(np.float32),
            np.float32(d_over_r0),
        )


class FrozenDegraded(DegradedPairs):
    """Валидация и тест: эпоха не меняется, набор заморожен навсегда."""

    def set_epoch(self, epoch):
        pass
