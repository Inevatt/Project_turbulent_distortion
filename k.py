import numpy as np
from scipy.ndimage import fourier_shift

tiles = np.load("data/tiles/tiles.npy", mmap_mode="r")
rng = np.random.default_rng(0)
p = []
for i in rng.choice(len(tiles), 500, replace=False):
    a = tiles[i].astype(np.float64) / 255.0
    b = np.fft.ifftn(fourier_shift(np.fft.fftn(a), rng.uniform(-0.5, 0.5, 2))).real
    c = slice(24, -24)          # обрезка заворота FFT
    p.append(10 * np.log10(1.0 / ((a[c, c] - b[c, c]) ** 2).mean()))
print(f"потолок: {np.mean(p):.1f} дБ")   # сдвиг через Фурье, без своего размытия
