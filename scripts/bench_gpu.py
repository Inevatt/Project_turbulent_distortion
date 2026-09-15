"""Чистое время шага обучения на GPU.

Запуск на арендованном инстансе, первым делом:
    python3 -m scripts.bench_gpu

Данные синтетические и уже лежат на карте, поэтому загрузка в замер не
попадает: меряется forward + backward + step и больше ничего. Вместе
с t_gen из bench_cpu это даёт число воркеров, при котором GPU не голодает:

    W = batch_size * t_gen / t_step

Флаги cudnn выставлены как в train.py. Иначе замер покажет скорость,
которой в настоящих прогонах не будет.
"""

import time

import torch
import yaml

from src.unet import UNet

WARMUP, REPEAT = 10, 50
STEPS_PER_EPOCH = 1374          # из check_split при текущем сплите
# мс/сэмпл с bench_cpu; ПЕРЕМЕРЬ на инстансе, серверный CPU медленнее
T_GEN = {"d15": 5.8, "d2": 3.7, "d3": 19.3}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA не видна — мерить нечего")

    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    d, t, m = cfg["data"], cfg["train"], cfg["model"]

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    net = UNet(in_ch=m["channels"], out_ch=m["channels"],
               base=m["base"], depth=m["depth"]).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=float(t["lr"]))
    loss_fn = torch.nn.L1Loss()

    n, batch = d["crop_px"], int(t["batch_size"])
    x = torch.randn(batch, m["channels"], n, n, device="cuda")
    y = torch.randn_like(x)

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn(net(x), y).backward()
        opt.step()

    for _ in range(WARMUP):      # аллокатор и первый выбор ядер cudnn
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(REPEAT):
        step()
    torch.cuda.synchronize()     # без него замеряется постановка в очередь
    t_step = (time.perf_counter() - t0) / REPEAT

    print(torch.cuda.get_device_name(0))
    print(f"батч {batch} x {n}x{n}, параметров "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
    print(f"память {torch.cuda.max_memory_allocated() / 1e9:.2f} ГБ\n")
    print(f"t_step = {t_step * 1e3:.1f} мс")
    print(f"эпоха  = {STEPS_PER_EPOCH * t_step:.0f} с")
    print(f"прогон = {STEPS_PER_EPOCH * 60 * t_step / 3600:.2f} ч")
    print(f"30 прогонов на 5 картах = "
          f"{STEPS_PER_EPOCH * 60 * t_step * 6 / 3600:.1f} ч стены\n")

    print("нужно воркеров W = batch * t_gen / t_step:")
    for name, tg in T_GEN.items():
        print(f"  {name:4s} t_gen={tg:5.1f} мс  ->  W = {batch * tg * 1e-3 / t_step:5.1f}")
    print(f"\nnum_workers в конфиге: {t['num_workers']}. "
          f"Если W больше — уровень упрётся в CPU.")


if __name__ == "__main__":
    main()
