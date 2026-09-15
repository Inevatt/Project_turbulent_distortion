# scripts/diag_dc.py
"""Разложение ошибки старого D0 на постоянную составляющую и остальное.

Точное тождество MSE(e) = Var(e) + (mean e)^2, поэтому доля DC в MSE
буквально говорит, какая часть ошибки создана покадровым сдвигом яркости.

PSNR берётся из src.metrics, а не считается формулой на месте: там обрезка
в [0,1], без неё числа несравнимы ни с log.csv, ни с no-op 29.42.
"""
import numpy as np, torch, yaml
from pathlib import Path
from torch.utils.data import DataLoader

from src.data import FrozenDegraded, split_indices
from src.distortion import LEVELS
from src.metrics import psnr
from src.unet import UNet

cfg = yaml.safe_load(Path("configs/seed1.yaml").read_text(encoding="utf-8"))
d, m = cfg["data"], cfg["model"]
f0 = cfg["degradation"]["diffraction_fwhm_px"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st = torch.load(Path(cfg["out_dir"]) / "d0" / "last.pt",
                map_location=dev, weights_only=False)
net = UNet(in_ch=m["channels"], out_ch=m["channels"],
           base=m["base"], depth=m["depth"]).to(dev)
net.load_state_dict(st["model"]); net.eval()

_, va, _ = split_indices(d["tiles"], d["val_frac"], d["test_frac"], cfg["split_seed"])
ds = FrozenDegraded(d["tiles"], va, LEVELS["d0"](f0), crop_px=d["crop_px"],
                    margin_px=d["margin_px"],
                    d_over_r0_range=tuple(d["d_over_r0_range"]),
                    seed=cfg["eval_seed"])

acc = {k: [] for k in ("noop","model","dc","aff","share","shift",
                       "m_in","m_pred","m_clean")}
with torch.no_grad():
    for degraded, clean, _ in DataLoader(ds, batch_size=32, num_workers=8):
        pred = net(degraded.to(dev)).cpu()
        err = pred - clean
        dc = err.mean((-2, -1), keepdim=True)

        # полная аффинная компенсация: не только сдвиг, но и масштаб —
        # GroupNorm делит на sigma, контраст мог сломаться вместе с яркостью
        pc = pred - pred.mean((-2, -1), keepdim=True)
        cc = clean - clean.mean((-2, -1), keepdim=True)
        a = (pc * cc).sum((-2, -1), keepdim=True) / \
            (pc * pc).sum((-2, -1), keepdim=True).clamp_min(1e-12)
        aff = a * pc + clean.mean((-2, -1), keepdim=True)

        mse = (err ** 2).mean((-3, -2, -1))
        acc["noop"].append(psnr(degraded, clean))
        acc["model"].append(psnr(pred, clean))
        acc["dc"].append(psnr(pred - dc, clean))
        acc["aff"].append(psnr(aff, clean))
        acc["share"].append((dc.flatten(1).squeeze(1) ** 2 / mse).numpy())
        acc["shift"].append(dc.flatten().numpy())
        acc["m_in"].append(degraded.mean((-3, -2, -1)).numpy())
        acc["m_pred"].append(pred.mean((-3, -2, -1)).numpy())
        acc["m_clean"].append(clean.mean((-3, -2, -1)).numpy())

r = {k: np.concatenate(v) for k, v in acc.items()}
print(f"no-op                 {r['noop'].mean():6.2f} дБ")
print(f"модель                {r['model'].mean():6.2f} дБ")
print(f"модель без DC         {r['dc'].mean():6.2f} дБ   <-- ключевая")
print(f"модель аффинно        {r['aff'].mean():6.2f} дБ")
print(f"доля DC в MSE         {r['share'].mean():6.3f}")
print(f"|DC|: сред {np.abs(r['shift']).mean():.4f}  макс {np.abs(r['shift']).max():.4f}")
print(f"corr(mean_вход,  mean_эталон) {np.corrcoef(r['m_in'],  r['m_clean'])[0,1]:.4f}")
print(f"corr(mean_предск,mean_эталон) {np.corrcoef(r['m_pred'],r['m_clean'])[0,1]:.4f}")
