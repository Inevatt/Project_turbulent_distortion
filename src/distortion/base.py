from abc import ABC, abstractmethod


class Degradation(ABC):
    """Общий интерфейс всех генераторов искажений.

    Вход:  img float32, форма (H, W), значения в [0, 1]
           d_over_r0 — D/r0 насколько больше пятно, чем в идеале
           rng — np.random.Generator, единственный источник случайности
    Выход: float32 той же формы, значения в [0, 1]

    Требование про rng не стилистическое: без него не воспроизвести
    замороженный тестовый набор, а без него все четыре модели
    тестируются на разных картинках и таблица ничего не значит.
    """

    name = "base"
    
    def __init__(self, diffraction_fwhm_px=2.0, margin_px=16):
        self.diffraction_fwhm_px = float(diffraction_fwhm_px)
        self.margin_px = int(margin_px)

    @abstractmethod
    def __call__(self, img, d_over_r0, rng):
        ...
