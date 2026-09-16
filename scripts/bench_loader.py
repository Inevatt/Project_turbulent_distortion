"""Пропускная способность загрузчика: 30 батчей вместо целой эпохи."""
import sys, time, yaml
from torch.utils.data import DataLoader
from src.data import DegradedPairs, split_indices
from src.distortion import LEVELS

level = sys.argv[1]
cfg = yaml.safe_load(open("configs/seed1.yaml", encoding="utf-8"))
d, t = cfg["data"], cfg["train"]
tr, _, _ = split_indices(d["tiles"], d["val_frac"], d["test_frac"], cfg["split_seed"])

ds = DegradedPairs(d["tiles"], tr, LEVELS[level](cfg["degradation"]["diffraction_fwhm_px"]),
                   crop_px=d["crop_px"], margin_px=d["margin_px"],
                   d_over_r0_range=tuple(d["d_over_r0_range"]),
                   seed=cfg["seed"], samples_per_tile=d["samples_per_tile"])
dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                num_workers=t["num_workers"], drop_last=True)

it = iter(dl)
for _ in range(5): next(it)          # прогрев: старт воркеров
N, t0 = 30, time.perf_counter()
for _ in range(N): next(it)
dt = (time.perf_counter() - t0) / N
print(f"{level}: {dt*1e3:6.0f} мс/батч -> эпоха {dt*1374:6.0f} с")
