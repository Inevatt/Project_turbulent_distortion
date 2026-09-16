"""Конвейер пар «испорченная / чистая».

Источник данных — нарезанные тайлы из scripts/make_tiles.py: сплошной
массив uint8 формы (N, TILE, TILE), открываемый через mmap_mode='r'.
Кэшированием занимается page cache ядра: он общий для всех воркеров,
страницы read-only, вытеснение бесплатное. Своего кэша здесь нет.

Посев случайности идёт ОТ НОМЕРА примера, а не от порядка вызовов.
Каждый пример получает два независимых генератора: один выбирает кроп,
другой разыгрывает искажение. Поэтому все четыре уровня видят одни
и те же куски одних и тех же картинок, сколько бы случайных чисел
ни потратил конкретный генератор искажения. С точностью до целого
сдвига: уровни с наклоном смещают окно таргета, см. ниже.

Кроп берётся с запасом margin_px с каждой стороны, портится целиком
и только потом обрезается до crop_px. Иначе краевой пиксель получает
свет не от настоящих соседей, а от того, что дорисует генератор
на границе массива: у D0 это зеркало scipy, у D2 через FFT — заворот
с противоположного края. Такая подпись границы своя у каждого уровня
и никак не связана с физикой, а сеть выучит именно её: диагональ
матрицы завысится, внедиагональные ячейки занизятся.

margin_px обязан быть одинаковым на всех шести уровнях и покрывать
носитель ядра при максимальном D/r0

Уровень возвращает применённый сдвиг, и окно таргета смещается на его
целую часть. Иначе один и тот же вход встречался бы с таргетами,
сдвинутыми на разные величины (сдвиг из кадра невосстановим), и лосс
увёл бы сеть к их смеси — то есть научил размывать выход везде.
Берётся ровно целая часть: дробный сдвиг чистой картинки потребовал бы
интерполяции, а размытый эталон хуже любого рассогласования. Остаётся
<= 0.5 px на ось, против размытия sigma >= 0.89 px это уже не решает.
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
    tiles = np.load(Path(tiles_dir) / "tiles.npy", mmap_mode="r")
    if source_id.ndim != 1 or len(source_id) != len(tiles):
        raise ValueError("source_id must contain one source ID per tile")
    if not (0 <= val_frac < 1 and 0 <= test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("invalid split fractions")
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
        margin_px,
        crop_px=128,
        d_over_r0_range=(1.0, 5.0),
        seed=1337,
        samples_per_tile=1,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.deg = degradation
        self.crop_px = int(crop_px)
        self.margin_px = int(margin_px)
        self.d_lo, self.d_hi = d_over_r0_range
        self.seed = int(seed)
        self.samples_per_tile = int(samples_per_tile)
        self.epoch = 0
        self.tiles = np.load(Path(tiles_dir) / "tiles.npy", mmap_mode="r")
        if self.tiles.ndim != 3 or self.tiles.dtype != np.uint8:
            raise ValueError("tiles.npy must be uint8 (N,H,W), grayscale")
        if self.indices.ndim != 1 or (len(self.indices) and
                (self.indices.min() < 0 or self.indices.max() >= len(self.tiles))):
            raise ValueError("invalid tile indices")
        if self.crop_px < 1 or self.margin_px < 0 or self.samples_per_tile < 1:
            raise ValueError("crop and samples_per_tile must be positive; margin nonnegative")
        if not (np.isfinite(self.d_lo) and np.isfinite(self.d_hi) and 1 <= self.d_lo <= self.d_hi):
            raise ValueError("dataset requires 1 <= D/r0 min <= max")
        if min(self.tiles.shape[1:]) < self.crop_px + 2 * self.margin_px:
            raise ValueError("tiles are smaller than crop + 2*margin")
        if self.deg.support_radius_px(self.d_hi) > self.margin_px:
            raise ValueError("margin does not cover degradation support")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.indices) * self.samples_per_tile

    def __getitem__(self, idx):
        tile = self.tiles[self.indices[idx % len(self.indices)]]
        m, n = self.margin_px, self.crop_px

        # поток 1: геометрия выборки — одинаков на всех уровнях
        rng_crop = _rng(self.seed, self.epoch, idx, 1)
        big = random_crop(tile, n + 2 * m, rng_crop)
        big = np.asarray(big, dtype=np.float32) / 255.0  # копия из memmap
        d_over_r0 = float(rng_crop.uniform(self.d_lo, self.d_hi))

        # big — источник и входа, и таргета, а таргет вырезается уже после
        # вызова. Генератор, пишущий по своему входу, упадёт здесь, а не
        # испортит таргеты молча.
        big.flags.writeable = False

        # поток 2: реализация искажения — своя на каждом уровне
        rng_deg = _rng(self.seed, self.epoch, idx, 2)
        degraded, shift = self.deg(big, d_over_r0, rng_deg)

        # Содержимое уехало на shift, значит в центре испорченного кадра
        # лежит то, что в big было на shift выше и левее. Границы проверять
        # не нужно: |round(shift)| <= support_radius_px <= margin_px, и это
        # неравенство train.py уже проверяет перед запуском.
        sl = slice(m, m + n)
        oy, ox = (int(round(v)) for v in shift)
        clean = big[m - oy : m - oy + n, m - ox : m - ox + n]

        return (
            np.expand_dims(degraded[sl, sl], axis=0).astype(np.float32),
            np.expand_dims(clean, axis=0).astype(np.float32),
            np.float32(d_over_r0),
        )


class FrozenDegraded(DegradedPairs):
    """Валидация и тест: эпоха не меняется, набор заморожен навсегда."""

    def set_epoch(self, epoch):
        pass
