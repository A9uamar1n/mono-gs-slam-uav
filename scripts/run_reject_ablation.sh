#!/usr/bin/env bash
# абляция отбраковки поз на итоговом трекере, те же 22 фрагмента и 3 зерна:
#   async_norej - асинхронный режим без отбраковки (сравнивать с основной серией, async)
#   sync_rej    - синхронный режим с отбраковкой (сравнивать с основной серией, sync)
# сравнение: python scripts/paired_comparison.py outputs/stage5 outputs/stage6/async_norej --mode-a async --mode-b async
# готовые запуски пропускаются
#   bash scripts/run_reject_ablation.sh [async_norej|sync_rej|all]
cd "$(dirname "$0")/.."
PY=${PY:-python}
ROOT=${ROOT:-outputs/stage6}
MODE=${1:-all}
declare -A FR=(
  [V1_01_easy]="100 500 900 1300 1700 2100 2400"
  [V1_02_medium]="100 500 900 1250"
  [MH_01_easy]="950 1350 1750 2150 2550 2950 3250"
  [MH_05_difficult]="450 850 1250 1650"
)
run() {  # out seq tag args...
  local out=$1 seq=$2 tag=$3; shift 3
  [ -f "$out/${seq}_${tag}/metrics.json" ] && return
  mkdir -p "$out/logs"
  echo "$(date +%T) $out $seq $tag"
  $PY -u scripts/run_slam.py data/euroc/$seq --tracker direct --out $out --tag $tag "$@" > "$out/logs/${seq}_${tag}.log" 2>&1 \
    || echo "  ОШИБКА, см. $out/logs/${seq}_${tag}.log"
  grep -E "^ATE" "$out/logs/${seq}_${tag}.log" | sed 's/^/  /'
}
series() {  # name mode args...
  local name=$1 mode=$2; shift 2
  for seq in V1_01_easy V1_02_medium MH_01_easy MH_05_difficult; do
    for st in ${FR[$seq]}; do
      for s in 0 1 2; do
        run $ROOT/$name $seq f${st}_${mode}_s$s --start $st --n-frames 400 --seed $s "$@"
      done
    done
  done
}
[[ $MODE == async_norej || $MODE == all ]] && series async_norej async --async --reject off
[[ $MODE == sync_rej || $MODE == all ]] && series sync_rej sync --reject on
echo STAGE6_DONE
