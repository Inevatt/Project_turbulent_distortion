"""Реестр уровней деградации.

Единственное место, где строка уровня связывается с классом.
train.py и eval_matrix.py импортируют LEVELS отсюда: два независимых
словаря могли бы разойтись в том, что означает "d2".

Ключ берётся из атрибута name самого класса, а не пишется руками.
Иначе опечатка `{"d1": D2Zernike}` молча положила бы веса D2
в каталог d1, и поле level в чекпоинте тоже сказало бы "d1".
"""

from .d0 import D0Gaussian
from .d1 import D1TipTilt
from .d2 import D2Kolmogorov

_CLASSES = (D0Gaussian, D1TipTilt, D2Kolmogorov)   # D3 добавляется одной строкой

LEVELS = {c.name: c for c in _CLASSES}

# Не assert: под python -O проверка исчезла бы вместе с гарантией.
if len(LEVELS) != len(_CLASSES) or "base" in LEVELS:
    raise ImportError(
        "у генераторов искажений name не задан или не уникален: "
        f"{[c.__name__ for c in _CLASSES]} -> {sorted(LEVELS)}")
