#!/usr/bin/env bash
# основная серия экспериментов: фрагменты по 400 кадров (20 с), не пересекаются,
# 3 зерна, синхронный и асинхронный режим + полные полёты (одно зерно).
# если metrics.json уже есть, запуск пропускается, так что можно перезапускать
#   bash scripts/run_main_experiments.sh [sync|async|full|all]
# другой python: PY=/путь/к/python bash scripts/run_main_experiments.sh
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=${OUT:-outputs/stage5}
MODE=${1:-all}
declare -A FR=(
  [V1_01_easy]="100 500 900 1300 1700 2100 2400"
  [V1_02_medium]="100 500 900 1250"
  [MH_01_easy]="950 1350 1750 2150 2550 2950 3250"
  [MH_05_difficult]="450 850 1250 1650"
)
declare -A FULL=( [V1_01_easy]=0 [V1_02_medium]=0 [MH_01_easy]=950 [MH_05_difficult]=450 )
run() {  # seq tag args...
  local seq=$1 tag=$2; shift 2
  [ -f "$OUT/${seq}_${tag}/metrics.json" ] && return
  echo "$(date +%T) $seq $tag"
  $PY -u scripts/run_slam.py data/euroc/$seq --tracker direct --out $OUT --tag $tag "$@" > "$OUT/logs/${seq}_${tag}.log" 2>&1 \
    || echo "  ОШИБКА, см. $OUT/logs/${seq}_${tag}.log"
  grep -E "^ATE" "$OUT/logs/${seq}_${tag}.log" | sed 's/^/  /'
}
mkdir -p $OUT/logs
for seq in V1_01_easy V1_02_medium MH_01_easy MH_05_difficult; do
  for st in ${FR[$seq]}; do
    for s in 0 1 2; do
      [[ $MODE == sync || $MODE == all ]] && run $seq f${st}_sync_s$s --start $st --n-frames 400 --seed $s
      [[ $MODE == async || $MODE == all ]] && run $seq f${st}_async_s$s --start $st --n-frames 400 --seed $s --async
    done
  done
done
if [[ $MODE == full || $MODE == all ]]; then
  for seq in V1_01_easy V1_02_medium MH_01_easy MH_05_difficult; do
    run $seq full_sync_s0 --start ${FULL[$seq]} --seed 0
  done
fi
echo STAGE5_DONE
