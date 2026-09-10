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
    
    def __init__(self, diffraction_fwhm_px):
        self.diffraction_fwhm_px = float(diffraction_fwhm_px)

    @abstractmethod
    def support_radius_px(self, d_over_r0):
        """Насколько далеко тянется ядро, px. Монотонна по d_over_r0."""

    @abstractmethod
    def __call__(self, img, d_over_r0, rng): ...
