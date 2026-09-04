"""Реестр уровней деградации.

Единственное место, где строка уровня связывается с классом.
train.py и eval_matrix.py импортируют LEVELS отсюда: два независимых
словаря могли бы разойтись в том, что означает "d2".

Ключ берётся из атрибута name самого класса, а не пишется руками.
Иначе опечатка `{"d1": D2Zernike}` молча положила бы веса D2
в каталог d1, и поле level в чекпоинте тоже сказало бы "d1".
"""

from .d0 import D0Gaussian

_CLASSES = (D0Gaussian,)          # D1-D3 добавляются одной строкой каждый

LEVELS = {c.name: c for c in _CLASSES}

# Не assert: под python -O проверка исчезла бы вместе с гарантией.
if len(LEVELS) != len(_CLASSES) or "base" in LEVELS:
    raise ImportError(
        "у генераторов искажений name не задан или не уникален: "
        f"{[c.__name__ for c in _CLASSES]} -> {sorted(LEVELS)}")
