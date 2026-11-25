#!/usr/bin/env bash
# Re-run the December 2026 European shadow predictions (STOXX Europe 600, DAX family, FTSE 100/250). Paper only.
#   ./rerun.sh                         # latest complete close, fresh fetch (lists, holdings, Yahoo) into raw/<today>/
#   ./rerun.sh 2026-10-30 oct30        # Fri 30 Oct, evening: prices at the Nov-list cut-off (old lists)
#   ./rerun.sh 2026-11-04 nov-lists    # Wed 4 Nov after 22:00 CET: new STOXX (2 Nov) and DAX (4 Nov) lists, cut-off 30 Oct
#   ./rerun.sh 2026-11-23 ff-freeze    # Mon 23 Nov evening: STOXX free-float/share freeze date
#   ./rerun.sh 2026-11-24 ftse-indicative raw/2026-11-24/ftse_indicative.csv   # Tue 24 Nov: FTSE indicative list
#   ./rerun.sh 2026-11-30 cutoff       # Mon 30 Nov after the close (after 17:30 CET): STOXX and DAX cut-off
#   ./rerun.sh 2026-12-01 ftse-cutoff  # Tue 1 Dec after the UK close: FTSE cut-off (STOXX announces the same evening)
# Inputs go to raw/<today>/ (a same-day re-run resumes the Yahoo pulls), outputs to predictions/<family>/ (old files with
# the same name move to superseded/), the console to logs/, a block with SHA-256 hashes to shadow_log.md, and the
# engine inputs 16_flow_engine/inputs/2026-12-18/{stoxx600,dax_family,ftse_uk}.csv are rewritten; then the engine runs
# (sheet 16_flow_engine/sheets/2026-12-18_best.*). After the announcements use score.py (see README).
set -euo pipefail
cd "$(dirname "$0")"
ASOF="${1:-}"
TAG="${2:-rerun}"
IND="${3:-}"
FD="$(date +%Y-%m-%d)"
mkdir -p logs
ARGS=(--fetch-date "$FD" --tag "$TAG")
if [ -n "$ASOF" ]; then ARGS+=(--asof "$ASOF"); fi
if [ -n "$IND" ]; then ARGS+=(--ftse-indicative "$IND"); fi
START=$(date +%s)
python3 -B predict.py "${ARGS[@]}" 2>&1 | tee "logs/predict_${FD}_${TAG}.log"
# engine sheet for the 18 Dec close (status best: announced rows, once entered, win over these predicted rows).
# run_engine.py calls the unchanged engine (engine.run) with market data from our throttled Yahoo pull, because
# `engine.py --md live` downloads in one threaded burst (see run_engine.py docstring).
PYTHONPATH="$(cd ../../.. && pwd)" python3 -B run_engine.py --fetch-date "$FD" 2>&1 | tee -a "logs/predict_${FD}_${TAG}.log"
echo "rerun.sh finished in $(( $(date +%s) - START ))s"
