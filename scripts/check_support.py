# scripts/check_support.py
import numpy as np, yaml
from src.distortion import LEVELS

cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
d = cfg["data"]
f0 = cfg["degradation"]["diffraction_fwhm_px"]
lo, hi = d["d_over_r0_range"]
margin, crop = d["margin_px"], d["crop_px"]

print(f"f0 = {f0}, D/r0 max = {hi}, margin_px = {margin}, crop_px = {crop}")
print(f"окно = {crop + 2*margin} px из тайла 256\n")

for name, cls in LEVELS.items():
    r = cls(f0).support_radius_px(hi)
    print(f"{name:6s} R = {r:6.2f} px   запас {margin - r:6.2f} px   "
          f"{'ok' if r <= margin else 'НЕ ВЛЕЗАЕТ'}")

# train.py проверяет носитель ТОЛЬКО при D/r0 = hi, опираясь на обещание
# монотонности в докстринге base.py. Проверяем само обещание.
for name, cls in LEVELS.items():
    deg = cls(f0)
    rs = [deg.support_radius_px(x) for x in np.linspace(lo, hi, 9)]
    assert all(b >= a - 1e-9 for a, b in zip(rs, rs[1:])), f"{name}: не монотонна"
print("\nмонотонность по D/r0: ok")
