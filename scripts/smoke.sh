#!/usr/bin/env bash
# scripts/smoke.sh — проверка перед боевым прогоном. Запускать из корня проекта.
#   bash scripts/smoke.sh d0
set -euo pipefail

LEVEL=${1:-d0}
RUN=experiments/${LEVEL}-overfit      # отладочный каталог из train.py
EP_OVERFIT=1000                       # 8 тайлов < batch => 1 шаг на эпоху, это 1000 шагов

rm -rf "$RUN"

echo "=== 1. переобучение на 8 тайлах ==="
python3 -m src.train --level "$LEVEL" --overfit 8 --epochs "$EP_OVERFIT"
echo "--- хвост log.csv ---"
tail -5 "$RUN/log.csv"

echo
echo "=== 2. одна эпоха на полных данных ==="
rm -rf "$RUN"
nvidia-smi dmon -s u -d 1 -c 3600 > /tmp/dmon.txt &
DMON=$!
SECONDS=0
python3 -m src.train --level "$LEVEL" --epochs 1
kill $DMON 2>/dev/null || true
echo "эпоха: ${SECONDS} s"
awk '!/^#/ {s+=$2; n++} END {if (n) printf "GPU sm: %.0f%% в среднем по %d отсчётам\n", s/n, n}' /tmp/dmon.txt
tail -2 "$RUN/log.csv"
ls "$RUN"

if [ -e "experiments/${LEVEL}/last.pt" ]; then
  echo "ВНИМАНИЕ: отладочный прогон записал в боевой каталог experiments/${LEVEL}"
fi

rm -rf "$RUN"
