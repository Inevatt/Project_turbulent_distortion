"""Оценка обученных моделей на синтетических тестовых столбцах.

Запуск на инстансе, по реплике на карту:
    CUDA_VISIBLE_DEVICES=0 python3 -m src.eval_matrix --config configs/s1.yaml

Результат — по файлу на (реплика, обучающий уровень):
    results/s1__d0.npz    ключи psnr_{столбец}_dr{v}, ssim_{...}, tile_idx
    results/s1__noop.npz  метрики самого испорченного входа

Всё остальное — матрица, retained, размах по репликам — считается офлайн
из этих файлов. Повторных прогонов сети не требуется ни для одного разреза.

СЕТКА D/r0. Тестовый набор берётся при d_over_r0_range=(v, v): uniform(v, v)
возвращает ровно v и тратит то же одно случайное число, поэтому кропы
на всех пяти значениях сетки совпадают побитово. Бины выходят точные
и сбалансированные, отдельный кэш не нужен.

СТОЛБЕЦ d3n. Ключ ГСЧ в data.py не содержит имени уровня, а шум в d3n
разыгрывается ПОСЛЕ поля смещений и коэффициентов. Значит столбцы d3
и d3n при одном (eval_seed, 0, idx) несут одну и ту же реализацию
турбулентности и отличаются ровно шумом — сравнение парное по построению.

СТРОКА no-op. Метрики испорченного входа против эталона. Без неё числа
в матрице не имеют масштаба: retained считается от (диагональ - no-op).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import FrozenDegraded, split_indices
from .distortion import LEVELS
from .metrics import psnr, ssim
from .unet import UNet

TEST_LEVELS = ("d0", "d1", "d2", "d3", "d3n")
DR_GRID = (1.0, 2.0, 3.0, 4.0, 5.0)
OUT = Path("results")


def run_column(net, ds, batch, workers, device):
    """(psnr, ssim) по кадрам. net=None — метрики самого входа."""
    dl = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=device.type == "cuda")
    ps, ss = [], []
    with torch.no_grad():
        for degraded, clean, _ in dl:
            pred = degraded if net is None else net(degraded.to(device)).cpu()
            ps.append(psnr(pred, clean))
            ss.append(ssim(pred, clean))
    return np.concatenate(ps), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    f0 = cfg["degradation"]["diffraction_fwhm_px"]
    tag = Path(cfg["out_dir"]).name.replace("experiments_", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    _, _, te = split_indices(d["tiles"], d["val_frac"], d["test_frac"],
                             cfg["split_seed"])
    kw = dict(crop_px=d["crop_px"], margin_px=d["margin_px"], seed=cfg["eval_seed"])
    OUT.mkdir(exist_ok=True)
    print(f"{tag}: тестовых тайлов {len(te)}, столбцов {len(TEST_LEVELS)}, "
          f"сетка D/r0 {DR_GRID}")

    for row in ("noop",) + tuple(sorted(LEVELS)):
        out = OUT / f"{tag}__{row}.npz"
        if out.exists():                      # оценка идемпотентна: упала — перезапусти
            print(f"  {row:5s} уже есть, пропуск")
            continue

        net = None
        if row != "noop":
            ckpt = Path(cfg["out_dir"]) / row / "last.pt"
            if not ckpt.exists():
                raise SystemExit(f"нет чекпоинта {ckpt}")
            state = torch.load(ckpt, map_location=device, weights_only=False)
            # Молчаливая пара «чужой чекпоинт + этот конфиг» дала бы ячейку,
            # посчитанную не тем разбиением данных. Сверка полная.
            if state["cfg"] != cfg:
                raise SystemExit(f"{ckpt} обучен другим конфигом")
            if state["level"] != row:
                raise SystemExit(f"{ckpt} содержит уровень {state['level']}")
            if state["epoch"] != int(t["epochs"]):
                raise SystemExit(f"{ckpt} недоучен: {state['epoch']} эпох "
                                 f"из {t['epochs']}")
            net = UNet(in_ch=m["channels"], out_ch=m["channels"],
                       base=m["base"], depth=m["depth"]).to(device)
            net.load_state_dict(state["model"])
            net.eval()

        data = {}
        for col in TEST_LEVELS:
            deg = LEVELS[col](f0)
            for v in DR_GRID:
                ds = FrozenDegraded(d["tiles"], te, deg,
                                    d_over_r0_range=(v, v), **kw)
                p, s = run_column(net, ds, int(t["batch_size"]),
                                  int(t["num_workers"]), device)
                data[f"psnr_{col}_dr{int(v)}"] = p.astype(np.float32)
                data[f"ssim_{col}_dr{int(v)}"] = s.astype(np.float32)
            print(f"  {row:5s} {col:4s} PSNR "
                  + " ".join(f"{data[f'psnr_{col}_dr{int(v)}'].mean():5.2f}"
                             for v in DR_GRID))

        np.savez_compressed(out, tile_idx=te.astype(np.int64), **data)

    print(f"готово: {OUT}/{tag}__*.npz")


if __name__ == "__main__":
    main()
