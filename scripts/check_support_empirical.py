import numpy as np, yaml
from src.distortion import LEVELS

cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
d = cfg["data"]
f0, hi = cfg["degradation"]["diffraction_fwhm_px"], d["d_over_r0_range"][1]
n, TILE = d["crop_px"], 256
REF = (TILE - n) // 2                      # 64 — максимум, что даёт тайл

tiles = np.load(f"{d['tiles']}/tiles.npy", mmap_mode="r")

for name, cls in LEVELS.items():
    deg = cls(f0)
    print(f"\n{name}")
    for m in (32, 40, 48, 56):
        worst, bad = 0.0, False
        for k in range(200):
            tile = np.asarray(tiles[k * 211 % len(tiles)], np.float32) / 255.0

            def run(mm):
                o = REF - mm
                w = np.ascontiguousarray(tile[o:TILE - o, o:TILE - o])
                w.flags.writeable = False
                out, sh = deg(w, hi, np.random.default_rng(k))
                return out[mm:mm + n, mm:mm + n], sh

            a, sa = run(m)
            b, sb = run(REF)
            if not np.allclose(sa, sb):
                print(f"  m={m:3d}  реализация зависит от размера окна, "
                      f"тест неприменим"); bad = True; break
            worst = max(worst, float(np.abs(a - b).max()))
        if not bad:
            print(f"  m={m:3d}  max|разница| = {worst:.2e}   "
                  f"{'чисто' if worst < 1e-5 else 'ЗАГРЯЗНЕНИЕ'}")
