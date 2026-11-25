#!/usr/bin/env bash
# Re-run the December 2026 Russell shadow prediction (paper only; public data; no accounts).
#   ./rerun.sh                       # latest complete close, fresh fetch of holdings / listings / Yahoo shares / prices
#   ./rerun.sh 2026-10-30 rankday    # rank-day run: start after ~18:00 ET on Fri 30 Oct so the close is in
#   ./rerun.sh 2026-10-30 pre-prelim # e.g. Thu 12 Nov: same rank-day prices, refreshed shares and holdings
# Inputs go to raw/<today>/ (a same-day re-run resumes the Yahoo shares pull), outputs to predictions/,
# the console log to logs/, and a block with SHA-256 hashes is appended to shadow_log.md.
set -euo pipefail
cd "$(dirname "$0")"
ASOF="${1:-}"
TAG="${2:-rerun}"
FD="$(date +%Y-%m-%d)"
mkdir -p logs
ARGS=(--fetch-date "$FD" --tag "$TAG")
if [ -n "$ASOF" ]; then ARGS+=(--asof "$ASOF"); fi
START=$(date +%s)
python3 predict.py "${ARGS[@]}" 2>&1 | tee "logs/predict_${FD}_${TAG}.log"
echo "rerun.sh finished in $(( $(date +%s) - START ))s"
