#!/usr/bin/env bash
# Runs ON the GCE VM: OOT hyper-parameter sweep with scripts/exp_harness.py (monthly cohorts).
# Each harness job uses 4 LightGBM threads; JOBS jobs run in parallel.
#   bash scripts/remote_sweep.sh <GCS_BUCKET_URI> [JOBS]
set -uo pipefail
cd "$HOME/sasol"
bucket="${1:?bucket}"; bucket="${bucket%/}"; jobs="${2:-3}"
mkdir -p logs/sweep data/raw data/processed
source .venv/bin/activate
export PYTHONUNBUFFERED=1
gcloud storage rsync "$bucket/data/raw" data/raw >/dev/null
gcloud storage ls "$bucket/processed/" >/dev/null 2>&1 && gcloud storage rsync "$bucket/processed" data/processed >/dev/null

# name|args
GRID_NAME="${3:-r1}"
# Tuned trees (gcp_tuned_config) shared by the cm* grids; OOT evals include the seasonal 2024-12-01 cohort.
T="--n 400 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3 --clf-n 500 --clf-leaves 15 --clf-mcs 200 --evals 2024-12-01,2025-06-01,2025-09-01"
if [[ "$GRID_NAME" == cm1 ]]; then
grid=(
  "cm1_tuned|$T --catmeta"
  "cm1_tuned_first06|$T --catmeta --first 2024-06-01"
  "cm1_tuned_decay09|$T --catmeta --decay 0.9"
  "cm1_tuned_cm500_pos80|$T --catmeta --cm-n 500 --cm-min-pos 80"
)
elif [[ "$GRID_NAME" == r2 ]]; then
grid=(
  "r2_l15_m100_c03|--n 600 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3"
  "r2_l15_m200_c03|--n 600 --lr 0.02 --leaves 15 --mcs 200 --colsample 0.3"
  "r2_l7_m100_c03|--n 1000 --lr 0.02 --leaves 7 --mcs 100 --colsample 0.3"
  "r2_l15_m100_c03_n400|--n 400 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3"
  "r2_l15_m100_c05_clf31m200|--n 600 --lr 0.02 --leaves 15 --mcs 100 --clf-n 200 --clf-leaves 31 --clf-mcs 200"
  "r2_l15_m100_c03_clf31m200c03|--n 600 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3 --clf-n 200 --clf-leaves 31 --clf-mcs 200"
  "r2_l15_m100_c03_clf15m300c03|--n 600 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3 --clf-n 400 --clf-leaves 15 --clf-mcs 300"
  "r2_l15_m100_c03_clf31m400c03|--n 600 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3 --clf-n 200 --clf-leaves 31 --clf-mcs 400"
  "r2_l15_m100_c03_clfn500|--n 600 --lr 0.02 --leaves 15 --mcs 100 --colsample 0.3 --clf-n 500 --clf-leaves 15 --clf-mcs 200"
)
else
grid=(
  "m_n400_lr03_l31_m50|--n 400 --lr 0.03 --leaves 31 --mcs 50"
  "m_n800_lr015_l31_m50|--n 800 --lr 0.015 --leaves 31 --mcs 50"
  "m_n800_lr03_l31_m50|--n 800 --lr 0.03 --leaves 31 --mcs 50"
  "m_n200_lr03_l31_m50|--n 200 --lr 0.03 --leaves 31 --mcs 50"
  "m_n600_lr02_l15_m100|--n 600 --lr 0.02 --leaves 15 --mcs 100"
  "m_n400_lr03_l63_m100|--n 400 --lr 0.03 --leaves 63 --mcs 100"
  "m_n400_lr03_l31_m200|--n 400 --lr 0.03 --leaves 31 --mcs 200"
  "m_n1000_lr02_l7_m50|--n 1000 --lr 0.02 --leaves 7 --mcs 50"
  "m_n400_lr03_l31_m50_c03|--n 400 --lr 0.03 --leaves 31 --mcs 50 --colsample 0.3"
  "m_n400_lr03_l31_m50_c07|--n 400 --lr 0.03 --leaves 31 --mcs 50 --colsample 0.7"
  "m_emb_n400|--n 400 --lr 0.03 --leaves 31 --mcs 50 --emb"
)
fi
run_one() { local name="${1%%|*}" args="${1#*|}"; python scripts/exp_harness.py --exp "$name" --monthly $args > "logs/sweep/$name.log" 2>&1; grep MEAN "logs/sweep/$name.log"; }
export -f run_one
# First job alone so the feature/label caches for monthly cohorts are built once (no write races).
run_one "${grid[0]}"
printf '%s\n' "${grid[@]:1}" | xargs -P "$jobs" -I{} bash -c 'run_one "$@"' _ {}
echo "SWEEP_DONE"
grep -h MEAN logs/sweep/${GRID_NAME}_*.log 2>/dev/null || grep -h MEAN logs/sweep/*.log | sort -t= -k5 -n
gcloud storage rsync data/processed "$bucket/processed" >/dev/null 2>&1 || true
gcloud storage cp logs/sweep/*.log "$bucket/sessions/vm-logs/sweep/" >/dev/null 2>&1 || true
