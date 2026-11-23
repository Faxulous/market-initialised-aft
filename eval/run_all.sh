#!/usr/bin/env bash
# Full evaluation pipeline: team parameters -> forecasts (+ market initialisation) -> paper outputs.
# Usage: eval/run_all.sh [--skip-params]      (from repo root; set PY to choose the interpreter)
# Each strategy writes eval/predictions/24-25_<strategy>.parquet, MAXJ at a time; --merge builds predictions.parquet.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
S=24-25; D=24_25; L="eval/log_$S"
MAXJ=3   # ~30 GB RAM: each strategy run holds all seasons' events (~5 GB)
W=5      # prediction/calibration workers per job: MAXJ*W ~ 16 cores
T="$PY eval/test.py --eval-season $S --strategy"
job() { while (( $(jobs -rp | wc -l) >= MAXJ )); do sleep 5; done; "$@" & }
echo "Start: $(date)"

if [[ "${1:-}" != "--skip-params" ]]; then   # expanding-window team parameters
  rm -rf "models/weibull/team_params/pl_$D" "models/maia/params/pl_$D" "models/zoubirthprocess/params/pl_$D"
  $PY -m models.weibull.team_params.estimate --eval-season "$S" > "eval/log_params_weibull_$S.log" 2>&1 &
  $PY -m models.maia.model --eval-season "$S" > "eval/log_params_maia_$S.log" 2>&1 &
  $PY -m models.zoubirthprocess.model --eval-season "$S" > "eval/log_params_zou_$S.log" 2>&1 &
  wait
  echo "Params done: $(date)"
fi

$PY eval/test.py --clear
$T betfair > "${L}_betfair.log" 2>&1 && $PY eval/test.py --merge   # calibrations take the eval match ids from the merged file (prices: models/market.py)
$PY eval/test.py --eval-season "$S" --fit-only > "${L}_fit_weibull.log" 2>&1
# uninitialised strategies + single-process Weibull calibration, MAXJ at a time
job $PY -m models.weibull.calibrate --season "$S" > "${L}_cal_weibull.log" 2>&1
for M in weibull_no_cov weibull zou maia; do job $T $M --workers $W > "${L}_$M.log" 2>&1; done
wait
# market-initialised strategies concurrently, each as soon as its own calibration is done
$T weibull_calibd --workers $W > "${L}_weibull_calibd.log" 2>&1 &
{ $PY -m models.zoubirthprocess.calibrate --season "$S" --workers $W > "${L}_cal_zou.log" 2>&1 &&
  $T zou_calibd --workers $W > "${L}_zou_calibd.log" 2>&1; } &
# Maia calibration is the long pole; its workers are light (~0.3 GB)
{ $PY -m models.maia.calibrate --season "$S" --workers 8 > "${L}_cal_maia.log" 2>&1 &&
  $T maia_calibd --workers $W > "${L}_maia_calibd.log" 2>&1; } &
wait
for M in betfair weibull_no_cov weibull zou maia weibull_calibd zou_calibd maia_calibd; do
  [[ -f "eval/predictions/${S}_$M.parquet" ]] || { echo "MISSING ${S}_$M (see ${L}_$M.log)"; exit 1; }
done
$PY eval/test.py --merge

# Paper outputs
$PY eval/analysis/benchmarks.py   # market-implied benchmarks, read by every script below
$PY eval/acc.py
$PY eval/bet.py
$PY eval/analysis/uncertainty.py
$PY eval/analysis/bet_robust.py
$PY eval/analysis/market_data.py
$PY eval/analysis/sanity.py
$PY eval/analysis/goal_latency.py
$PY eval/refit.py
$PY eval/test_stoppage.py
$PY eval/shift.py
echo "Done: $(date)"
