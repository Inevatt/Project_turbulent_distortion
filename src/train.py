"""Обучение UNet на одном уровне деградации.

Между четырьмя прогонами меняется РОВНО один аргумент --level. Всё
остальное — стартовые веса, порядок батчей, кропы, значения D/r0, число
шагов, расписание lr — обязано совпадать, иначе разница между строками
матрицы 4x4 перестанет объясняться только физикой обучающих данных.

Зафиксированные решения (менять только сразу для всех четырёх уровней):

  * нет early stopping и нет отбора лучшего чекпоинта по валидации.
    Иначе четыре модели получат разное число шагов, и часть разницы
    в матрице объяснится длительностью обучения. Бюджет шагов фиксирован,
    в eval_matrix.py идёт ПОСЛЕДНИЙ чекпоинт. Валидация здесь нужна
    только чтобы видеть, что обучение не разошлось;

  * torch.manual_seed(cfg["seed"]) вызывается до создания сети — все
    четыре модели стартуют из одних и тех же весов;

  * порядок батчей задаётся отдельным torch.Generator от того же seed,
    поэтому перестановка индексов на всех уровнях одинакова. Вместе
    с двумя потоками ГСЧ внутри data.py это даёт полное совпадение
    обучающих кропов и значений D/r0 между уровнями;

  * set_epoch вызывается каждую эпоху, а persistent_workers НЕ
    включается. Воркеры живут одну эпоху и только поэтому забирают новое
    значение epoch: с persistent_workers=True их копии останутся
    на нулевой эпохе, и все 60 эпох пройдут по одним и тем же примерам
    без единого предупреждения;

  * d_over_r0 (третий элемент батча) в сеть не подаётся. Условие на силу
    турбулентности сделало бы задачу non-blind, а на OTIS r0 неизвестен.

Запуск из корня проекта:
    python3 -m src.train --level d0
Отладка перед первым настоящим прогоном:
    python3 -m src.train --level d0 --overfit 8 --epochs 1000
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import DegradedPairs, FrozenDegraded, split_indices
from .distortion import LEVELS   # реестр один на train.py и eval_matrix.py
from .metrics import psnr, ssim
from .unet import UNet

# Уровень выбирается строкой, никаких if по уровням в коде обучения.
# D1-D3 добавляются одной строкой каждый.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=sorted(LEVELS))
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить с last.pt (инстанс AutoDL могут погасить)")
    ap.add_argument("--overfit", type=int, default=0, metavar="N",
                    help="ОТЛАДКА: обучаться на N замороженных тайлах без валидации")
    ap.add_argument("--epochs", type=int, default=None,
                    help="ОТЛАДКА: перебить число эпох из конфига")
    args = ap.parse_args()
    if args.resume and args.epochs:
        ap.error("--resume и --epochs вместе нельзя: T_max косинуса привязан "
                 "к бюджету шагов, lr пойдёт вверх вместо нуля")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    epochs = int(t["epochs"]) if args.epochs is None else args.epochs
    overfit = args.overfit > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   # формы фиксированы, подбор алгоритмов бесплатен

    # --- данные ------------------------------------------------------------
    deg = LEVELS[args.level](
        diffraction_fwhm_px=cfg["degradation"]["diffraction_fwhm_px"],
        margin_px=d["margin_px"])

    tr, va, _ = split_indices(d["tiles"], d["val_frac"], d["test_frac"],
                              cfg["split_seed"])
    if overfit:
        # с шагом, иначе 8 подряд идущих тайлов окажутся кусками одной
        # фотографии и проверка станет слишком лёгкой
        tr = tr[:: max(1, len(tr) // args.overfit)][: args.overfit]

    kw = dict(crop_px=d["crop_px"], margin_px=d["margin_px"],                                                 d_over_r0_range=tuple(d["d_over_r0_range"]))
    # В режиме --overfit обучающий набор тоже заморожен: смысл проверки в том,
    # чтобы сеть выучила КОНКРЕТНЫЕ примеры. С живым DegradedPairs кропы
    # менялись бы каждую эпоху и лосс никогда не ушёл бы в ноль.
    if overfit:
        train_ds = FrozenDegraded(d["tiles"], tr, deg, seed=cfg["seed"], **kw)
    else:
        train_ds = DegradedPairs(d["tiles"], tr, deg, seed=cfg["seed"],
                                 samples_per_tile=d["samples_per_tile"], **kw)
    # Валидация замораживается своим сидом и живёт на кропах того же размера,
    # что обучение: GroupNorm нормирует по всему полю, поэтому на кадре
    # другого размера числа несравнимы с обучающим режимом.
    val_ds = FrozenDegraded(d["tiles"], va, deg, seed=cfg["eval_seed"], **kw)

    batch = min(int(t["batch_size"]), len(train_ds))
    workers = 0 if overfit else int(t["num_workers"])
    pin = device.type == "cuda"

    # Генератор пересевается в начале каждой эпохи, поэтому порядок батчей —
    # функция от (seed, epoch), а не накопленное состояние. Иначе процесс,
    # поднятый с --resume, начал бы перестановку заново и увидел бы порядок
    # нулевой эпохи: прогон, переживший обрыв, перестал бы быть сравнимым
    # с прогоном без обрыва.
    gen = torch.Generator()
    train_dl = DataLoader(
        train_ds, batch_size=batch, shuffle=not overfit, generator=gen,
        num_workers=workers, pin_memory=pin, drop_last=not overfit,
        # persistent_workers НЕ ставить: сломает set_epoch, см. докстринг
    )
    val_dl = DataLoader(val_ds, batch_size=int(t["batch_size"]), shuffle=False,
                        num_workers=workers, pin_memory=pin)
    do_val = not overfit and len(val_ds) > 0

    # --- модель ------------------------------------------------------------
    torch.manual_seed(cfg["seed"])           # одинаковые стартовые веса на всех уровнях
    net = UNet(in_ch=m["channels"], out_ch=m["channels"],
               base=m["base"], depth=m["depth"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(t["lr"]))
    steps_per_epoch = len(train_dl)
    total_steps = epochs * steps_per_epoch
    # Косинус до нуля, привязанный к тому же бюджету шагов. Расписание
    # детерминированное, поэтому одинаково на всех четырёх уровнях.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    loss_fn = torch.nn.L1Loss()

    # --- каталог прогона ---------------------------------------------------
    run = Path(cfg["out_dir"]) / args.level
    run.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = run / "last.pt", run / "log.csv"

    start_epoch = 0
    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"нечего продолжать: {ckpt_path} нет")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Продолжать с другим конфигом нельзя: ячейка матрицы окажется
        # посчитанной двумя разными наборами гиперпараметров.
        if state["cfg"] != cfg:
            raise SystemExit("конфиг изменился с момента чекпоинта — "
                             "либо верните его, либо начните прогон заново")
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"]
    else:
        # Лог перезаписывается, а не дописывается. Иначе строки от прошлого
        # прогона с другим конфигом молча смешаются с новыми, и по log.csv
        # уже не понять, чем посчитана эта строка матрицы.
        if ckpt_path.exists():
            print(f"ВНИМАНИЕ: перезаписываю прошлый прогон в {run}")
        log_path.write_text("epoch,step,lr,train_l1,val_psnr,val_ssim,sec\n",
                            encoding="utf-8")

    # Снимок конфига делается ПОСЛЕ сверки при resume, иначе несовпавший
    # конфиг успел бы затереть тот, которым прогон реально считался.
    shutil.copy(args.config, run / "config.yaml")

    print(f"уровень {args.level} | {device} | тайлов train {len(tr)}, "
          f"примеров {len(train_ds)}, val {len(val_ds)}")
    print(f"бюджет: {steps_per_epoch} шагов x {epochs} эпох = {total_steps} шагов")

    # --- обучение ----------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        # Новые кропы и новые D/r0. Работает только потому, что воркеры
        # пересоздаются каждую эпоху.
        train_ds.set_epoch(epoch)
        gen.manual_seed(cfg["seed"] + epoch)
        net.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        lr_epoch = opt.param_groups[0]["lr"]   # с чем эпоха шла, а не следующий

        for degraded, clean, _d_over_r0 in train_dl:   # D/r0 в обучении не нужен, см. докстринг
            degraded = degraded.to(device, non_blocking=pin)
            clean = clean.to(device, non_blocking=pin)
            loss = loss_fn(net(degraded), clean)
            if not torch.isfinite(loss):
                raise SystemExit(f"лосс не конечен на эпохе {epoch + 1}: "
                                 "один NaN из генератора убивает все веса за шаг")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(degraded)
            seen += len(degraded)

        train_l1 = loss_sum / max(seen, 1)
        val_p = val_s = float("nan")
        if do_val:
            net.eval()
            ps, ss = [], []
            with torch.no_grad():
                for degraded, clean, _ in val_dl:
                    pred = net(degraded.to(device, non_blocking=pin))
                    ps.append(psnr(pred, clean))       # (B,), редукция здесь
                    ss.append(ssim(pred, clean))
            val_p = float(np.concatenate(ps).mean())
            val_s = float(np.concatenate(ss).mean())

        sec = time.time() - t0
        step = (epoch + 1) * steps_per_epoch
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{step},{lr_epoch:.3e},{train_l1:.6f},"
                    f"{val_p:.4f},{val_s:.5f},{sec:.1f}\n")
        print(f"эпоха {epoch + 1:3d}/{epochs} | шаг {step:6d} | L1 {train_l1:.5f} | "
              f"val PSNR {val_p:6.2f} дБ | SSIM {val_s:.4f} | lr {lr_epoch:.2e} | {sec:5.1f} с")

        # Сохраняем каждую эпоху через временный файл: если инстанс погаснет
        # в момент записи, last.pt останется целым и с прошлой эпохи.
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"level": args.level, "epoch": epoch + 1, "cfg": cfg,
                    "model": net.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict()}, tmp)
        tmp.replace(ckpt_path)

    print(f"готово, чекпоинт {ckpt_path}")


if __name__ == "__main__":
    main()
