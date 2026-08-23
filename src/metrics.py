# -*- coding: utf-8 -*-
"""Метрики качества восстановления.

Единственное место в проекте, где считаются PSNR и SSIM: валидация
в train.py, матрица в eval_matrix.py и оценка на OTIS импортируют
отсюда, иначе ячейки матрицы перестают быть сравнимыми.

Зафиксированные конвенции (менять только сразу для всех 16 ячеек):
  * усреднение по изображениям, а не по всем пикселям разом — логарифм
    нелинеен, и пул MSE по батчу занижает результат тем сильнее, чем
    больше разброс сложности внутри батча;
  * предсказание обрезается в [0, data_range]: выбросы за диапазон
    иначе штрафуются, хотя при сохранении в uint8 они исчезают;
  * SSIM по Wang et al. — гауссово окно 11x11, sigma=1.5, ковариация
    с нормировкой на N. Дефолт skimage другой: 7x7 равномерное и N-1.

Обе функции возвращают вектор (B,) по кадрам. Редукция — на стороне
вызывающего, чтобы eval_matrix мог сохранить разброс между
реализациями турбулентности, а не только среднее.
"""
import numpy as np
from skimage.metrics import structural_similarity


def _prepare(pred, target, data_range):
    """torch/numpy -> float64 (B, C, H, W); pred обрезается в диапазон.

    Проверки формы не косметические: несовпадающие формы numpy молча
    растянет через broadcasting, а (B, H, W) без оси канала skimage
    примет за одну картинку и посчитает SSIM по срезу.
    """
    arrays = []
    for x in (pred, target):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        arrays.append(np.asarray(x, dtype=np.float64))
    pred, target = arrays

    if pred.shape != target.shape:
        raise ValueError(f"формы не совпадают: {pred.shape} и {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"ожидается (B, C, H, W), получено {pred.shape}")

    return np.clip(pred, 0.0, data_range), target


def psnr(pred, target, data_range=1.0):
    """PSNR по кадрам, дБ. Возвращает (B,)."""
    pred, target = _prepare(pred, target, data_range)
    mse = ((pred - target) ** 2).reshape(len(pred), -1).mean(axis=1)
    mse = np.maximum(mse, 1e-12 * data_range ** 2)   # потолок ~120 дБ
    return 10.0 * np.log10(data_range ** 2 / mse)


def ssim(pred, target, data_range=1.0):
    """SSIM по кадрам, усреднение по каналам. Возвращает (B,)."""
    pred, target = _prepare(pred, target, data_range)
    return np.array([
        structural_similarity(
            p, t,
            data_range=data_range,
            channel_axis=0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        for p, t in zip(pred, target)
    ])
