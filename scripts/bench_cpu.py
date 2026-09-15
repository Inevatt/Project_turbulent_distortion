# scripts/bench_cpu.py
import os, time, yaml
from src.data import DegradedPairs, split_indices
from src.distortion import LEVELS

cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
d, t = cfg["data"], cfg["train"]
fwhm = cfg["degradation"]["diffraction_fwhm_px"]

tr, _, _ = split_indices(d["tiles"], d["val_frac"], d["test_frac"], cfg["split_seed"])

print(f"vCPU {os.cpu_count()}   batch {t['batch_size']}   num_workers {t['num_workers']}")

N = 50
for name, cls in LEVELS.items():
    ds = DegradedPairs(d["tiles"], tr, cls(fwhm),
                       margin_px=d["margin_px"], crop_px=d["crop_px"],
                       d_over_r0_range=d["d_over_r0_range"])
    ds[0]                                   # прогрев
    t0 = time.perf_counter()
    for i in range(N):
        ds[i * 37]
    print(f"{name:6s} {(time.perf_counter()-t0)/N*1e3:7.1f} мс/сэмпл")
