#!/usr/bin/env bash
# Re-run the MSCI November 2026 shadow prediction (paper only; public files; no accounts) and rebuild the engine
# inputs for the 30 Nov close. MSCI's price cutoff is one unannounced day in Mon 19 - Fri 30 Oct 2026 (gimi §5);
# the last three cutoffs were the 1st/2nd day of their windows (15 Oct 2025, 20 Apr 2026, 20 Jul 2026).
#
#   ./rerun.sh 2026-10-19            # one candidate cutoff date; run the NEXT morning (UCITS files post overnight)
#   ./rerun.sh window                # every window date up to yesterday that has no logged run yet
#   ./rerun.sh 2026-10-19 cutoff-1019-v2   # explicit tag (a same-tag rerun archives the old files first)
#
# Schedule (README §Runbook): Tue 20 Oct morning -> 19 Oct; Wed 21 Oct -> 20 Oct; ... Mon 2 Nov -> 30 Oct.
# Each run fetches iShares UCITS holdings for that date into raw/<today>/, Yahoo closes for Canadian and European
# non-members, writes predictions/ and appends a hashed block to shadow_log.md; then rewrites
# 16_flow_engine/inputs/2026-11-30/msci_{canada,europe}.csv from that run.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)"   # code/, for r2
FD="$(date +%Y-%m-%d)"
mkdir -p logs
run_one () {
  local ASOF="$1" TAG="$2"
  python3 -W ignore predict.py --asof "$ASOF" --fetch-date "$FD" --tag "$TAG" 2>&1 | tee "logs/predict_${FD}_${TAG}.log"
  python3 build_engine_inputs.py --run "predictions/pred_${ASOF}_${TAG}_changes.csv" 2>&1 | tee -a "logs/predict_${FD}_${TAG}.log"
}
if [ "${1:-}" = "window" ]; then
  for D in 2026-10-19 2026-10-20 2026-10-21 2026-10-22 2026-10-23 2026-10-26 2026-10-27 2026-10-28 2026-10-29 2026-10-30; do
    if [[ "$D" < "$FD" ]]; then
      T="cutoff-${D:5:2}${D:8:2}"
      if ls predictions/pred_${D}_${T}_changes.csv >/dev/null 2>&1; then echo "skip $D (logged)"; else run_one "$D" "$T"; fi
    fi
  done
else
  ASOF="${1:?usage: ./rerun.sh YYYY-MM-DD [TAG] | ./rerun.sh window}"
  run_one "$ASOF" "${2:-cutoff-${ASOF:5:2}${ASOF:8:2}}"
fi
