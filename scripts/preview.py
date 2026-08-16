# scripts/preview.py — запуск: python3 -m scripts.preview
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from src.distortion.d0 import D0Gaussian

cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
FWHM_DIFF = cfg["degradation"]["diffraction_fwhm_px"]
ROOT = Path(cfg["data"]["root"])

LEVEL = "d0"
STRENGTHS = [0.0, 1.0, 3.0, 5.0, 10.0]   # 0.0 = оригинал
CROP = 256
N_IMAGES = 3

deg = D0Gaussian(diffraction_fwhm_px=FWHM_DIFF)
rng = np.random.default_rng(0)

paths = sorted(ROOT.glob("*.png"))[:N_IMAGES]

rows = []
for p in paths:
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    top, left = (h - CROP) // 2, (w - CROP) // 2
    crop = img[top : top + CROP, left : left + CROP]
    cells = [crop if d == 0.0 else deg(crop, d, rng) for d in STRENGTHS]
    rows.append(np.concatenate(cells, axis=1))

grid = np.concatenate(rows, axis=0)
Image.fromarray((grid * 255.0).round().astype(np.uint8)).save(f"preview_{LEVEL}.png")
print(f"силы: {STRENGTHS} -> preview_{LEVEL}.png")
