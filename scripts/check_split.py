"""Проверка разбиения train/val/test.

Запуск из корня проекта:
    python3 -m scripts.check_split

Печатает размеры сплитов по тайлам и по ФОТОГРАФИЯМ и падает, если
фотография попала сразу в два сплита. Число фотографий в test — это
эффективный размер тестовой выборки: тайлы одного снимка почти
не добавляют независимости, поэтому в текст работы идёт именно оно.

Прогоняется по всем пяти split_seed сразу: реплики отличаются в том
числе разбиением, и лучше увидеть все пять до запуска, чем одно.
"""

import numpy as np
import yaml
from pathlib import Path

from src.data import split_indices

SEEDS = [42, 43, 44, 45, 46]        # как в configs/s1..s5.yaml


def main():
    cfg = yaml.safe_load(open("configs/seed1.yaml", encoding="utf-8"))
    d, t = cfg["data"], cfg["train"]

    source_id = np.load(Path(d["tiles"]) / "source_id.npy")
    n_src = len(np.unique(source_id))
    print(f"тайлов {len(source_id)}, фотографий {n_src}\n")

    for seed in SEEDS:
        tr, va, te = split_indices(d["tiles"], d["val_frac"], d["test_frac"], seed)

        # фотографии, а не тайлы: именно они разводятся по сплитам
        s_tr, s_va, s_te = (set(np.unique(source_id[i]).tolist())
                            for i in (tr, va, te))

        # Утечка тихая: метрики просто окажутся выше, ничего не упадёт.
        # Поэтому проверка жёсткая и здесь, а не в комментарии.
        assert not (s_tr & s_va or s_tr & s_te or s_va & s_te), \
            f"seed {seed}: фотография в двух сплитах"
        assert len(tr) + len(va) + len(te) == len(source_id), \
            f"seed {seed}: тайлы потерялись"

        print(f"split_seed {seed}")
        for name, idx, src in (("train", tr, s_tr),
                               ("val", va, s_va),
                               ("test", te, s_te)):
            print(f"  {name:5s} тайлов {len(idx):6d}   фотографий {len(src):5d}")
        print(f"  шагов в эпохе {len(tr) // t['batch_size']}")

    # Разные split_seed обязаны давать разные тестовые наборы, иначе
    # реплики меряют одно и то же и разброс между ними занижен.
    tests = [frozenset(np.unique(source_id[split_indices(
        d["tiles"], d["val_frac"], d["test_frac"], s)[2]]).tolist())
        for s in SEEDS]
    pair = min(len(a & b) / len(a) for i, a in enumerate(tests)
               for b in tests[i + 1:])
    print(f"\nминимальное перекрытие тестовых наборов между репликами: "
          f"{pair:.1%}")


if __name__ == "__main__":
    main()
