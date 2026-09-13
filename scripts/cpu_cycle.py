"""scripts/cpu_overfit.py — переобучение на 16 тайлах, с val и test, на CPU.

Смысл: train L1 должен уйти почти в нуль — это доказывает, что unet.py и
цикл обучения исполняются и что таргет для d1 привязан правильно. Val и test
доказывают другое: путь метрик исполняется и возвращает конечные числа
(в --overfit режиме train.py валидацию отключает).

Val и test падать НЕ должны: на 16 тайлах сеть их запоминает, обобщения нет.
Полка или рост на val/test — признак работающего переобучения, не дефект.

    python3 -m scripts.cpu_overfit --level d1
"""
import argparse
import time
from pathlib import Path

import torch
import yaml

from src.data import FrozenDegraded, split_indices
from src.distortion import LEVELS
from src.metrics import psnr, ssim
from src.unet import UNet

ROOT = Path(__file__).resolve().parents[1]


def make_batch(tiles, idx, deg, d, seed, n):
    ds = FrozenDegraded(tiles, idx[:n], deg, d["margin_px"],
                        crop_px=d["crop_px"],
                        d_over_r0_range=tuple(d["d_over_r0_range"]),
                        seed=seed, samples_per_tile=d["samples_per_tile"])
    pairs = [ds[i] for i in range(len(ds))]
    return (torch.stack([torch.as_tensor(p[0], dtype=torch.float32) for p in pairs]),
            torch.stack([torch.as_tensor(p[1], dtype=torch.float32) for p in pairs]))


def scalar(v):
    return float(torch.as_tensor(v).mean())


@torch.no_grad()
def evaluate(model, x, y):
    p = model(x)
    return (scalar((p - y).abs()), scalar(psnr(p, y)), scalar(ssim(p, y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="d1")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--every", type=int, default=25)
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / a.config).read_text())
    d, m = cfg["data"], cfg["model"]
    tiles = ROOT / d["tiles"]

    tr, va, te = split_indices(tiles, d["val_frac"], d["test_frac"],
                               cfg["split_seed"])
    deg = LEVELS[a.level](**cfg["degradation"])

    torch.manual_seed(cfg["seed"])
    sets = {
        "train": make_batch(tiles, tr, deg, d, cfg["seed"], a.n),
        "val":   make_batch(tiles, va, deg, d, cfg["eval_seed"], a.n),
        "test":  make_batch(tiles, te, deg, d, cfg["eval_seed"], a.n),
    }
    xt, yt = sets["train"]

    model = UNet(m["channels"], m["channels"], m["base"], m["depth"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    print(f"уровень {a.level}, потоков {torch.get_num_threads()}, "
          f"параметров {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    print(f"сплиты целиком: train {len(tr)}, val {len(va)}, test {len(te)}; "
          f"взято по {a.n}")
    print(f"батч: вход {tuple(xt.shape)} {xt.dtype}, таргет {tuple(yt.shape)}, "
          f"вход [{xt.min():.3f}, {xt.max():.3f}]")
    print(f"\n{'эпоха':>6}  {'train L1':>9}  {'val L1':>8}  {'val PSNR':>8}  "
          f"{'val SSIM':>8}  {'test L1':>8}  {'test PSNR':>9}  {'test SSIM':>9}")

    first = last = None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = (model(xt) - yt).abs().mean()
        loss.backward()
        opt.step()
        last = loss.item()
        if first is None:
            first = last

        if ep % a.every == 0 or ep == 1:
            model.eval()
            v = evaluate(model, *sets["val"])
            t = evaluate(model, *sets["test"])
            print(f"{ep:>6}  {last:>9.5f}  {v[0]:>8.5f}  {v[1]:>8.2f}  "
                  f"{v[2]:>8.4f}  {t[0]:>8.5f}  {t[1]:>9.2f}  {t[2]:>9.4f}")

    model.eval()
    tr_m = evaluate(model, xt, yt)
    print(f"\nвремя {time.time() - t0:.0f} s, {(time.time() - t0) / a.epochs:.2f} s/эпоха")
    print(f"train L1: {first:.5f} -> {last:.5f}  (в {first / last:.0f} раз)")
    print(f"train PSNR на запомненных парах: {tr_m[1]:.2f} dB")
    print("итог: " + ("train L1 ниже 0.01, цикл и таргет корректны"
                      if last < 0.01 else
                      f"train L1 {last:.4f} НЕ ниже 0.01 — смотри, полка это "
                      f"или просто медленная сходимость"))


if __name__ == "__main__":
    main()
