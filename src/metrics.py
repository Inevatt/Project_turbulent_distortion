# -*- coding: utf-8 -*-
"""Метрики качества восстановления.

Единственное место в проекте, где считаются PSNR и SSIM. Валидация
в train.py, матрица в eval_matrix.py и оценка на реальных данных
импортируют отсюда — иначе числа перестанут быть сравнимыми.

Обе метрики усредняются ПО ИЗОБРАЖЕНИЯМ батча, а не по всем пикселям
разом. Для PSNR это принципиально: логарифм нелинеен, и общий MSE
с последующим переводом в децибелы даёт другой ответ. При разбросе
D/r0 от 1 до 5 внутри батча расхождение доходит до 3-4 дБ и плавает
от батча к батчу — то есть того же порядка, что измеряемый эффект.
"""

import numpy as np
from skimage.metrics import structural_similarity


def _to_numpy(x):
    """Принимает torch-тензор или ndarray, отдаёт float64 ndarray."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _check(pred, target):
    """Приводит к numpy и проверяет формы.

    Без этой проверки три ошибки проходят молча и дают правдоподобные,
    но неверные числа: несовпадение форм numpy растягивает через
    broadcasting, а массив (B, H, W) без оси канала SSIM принимает
    за картинку и меряет по одной строке пикселей.
    """
    pred, target = _to_numpy(pred), _to_numpy(target)
    if pred.shape != target.shape:
        raise ValueError(f"формы не совпадают: {pred.shape} и {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"ожидается (B, C, H, W), получено {pred.shape}")
    return pred, target


def psnr(pred, target, data_range=1.0):
    """Среднее по изображениям батча, дБ. Форма (B, C, H, W), значения [0,1]."""
    pred, target = _check(pred, target)
    mse = ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(axis=1)
    mse = np.maximum(mse, 1e-12)          # защита от совпадающих картинок
    return float(np.mean(10.0 * np.log10(data_range**2 / mse)))


def ssim(pred, target, data_range=1.0):
    """Среднее по изображениям батча. Реализация из skimage"""
    pred, target = _check(pred, target)
    n_ch = pred.shape[1]
    vals = [
        np.mean([
            structural_similarity(t[c], p[c], data_range=data_range)
            for c in range(n_ch)
        ])
        for p, t in zip(pred, target)
    ]
    return float(np.mean(vals))
