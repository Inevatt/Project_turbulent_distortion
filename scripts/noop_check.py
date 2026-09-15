# scripts/noop_check.py
import numpy as np, yaml
from torch.utils.data import DataLoader
from src.data import FrozenDegraded, split_indices
from src.distortion import LEVELS
from src.metrics import psnr, ssim

cfg = yaml.safe_load(open("configs/seed1.yaml", encoding="utf-8"))
d, f0 = cfg["data"], cfg["degradation"]["diffraction_fwhm_px"]
_, va, _ = split_indices(d["tiles"], d["val_frac"], d["test_frac"], cfg["split_seed"])

for name in ("d0", "d3"):
    ds = FrozenDegraded(d["tiles"], va, LEVELS[name](f0),
                        crop_px=d["crop_px"], margin_px=d["margin_px"],
                        d_over_r0_range=tuple(d["d_over_r0_range"]),
                        seed=cfg["eval_seed"])
    ps, ss = [], []
    for degraded, clean, _ in DataLoader(ds, batch_size=32, num_workers=8):
        ps.append(psnr(degraded, clean))
        ss.append(ssim(degraded, clean))
    print(f"no-op {name}: PSNR {np.concatenate(ps).mean():6.2f}  "
          f"SSIM {np.concatenate(ss).mean():.4f}")
