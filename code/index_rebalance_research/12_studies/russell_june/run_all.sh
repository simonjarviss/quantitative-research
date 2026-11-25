#!/bin/zsh
# Reproduce the Russell June reconstitution study from scratch (~10 min; first R2 pull ~6 min, 585 MB cache).
# Needs R2 credentials (python3 -m r2 --check-keys). Delete _cache/ to force a fresh pull.
set -e
STUDY="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$STUDY/../../.." && pwd):$STUDY"   # code/ (for r2) and the study
cd $STUDY
python3 s01_fetch.py            # R2 -> _cache/*.parquet
python3 s02_events.py           # Datastream lists -> CRSP permco; actual events; recon-day volume check
python3 s03_predict.py --rankfit   # rank-day candidate fit (validates assumed rank days)
python3 s03_predict.py          # PIT rank rebuild at H6/H3/HR, predictions, accuracy
python3 s04_returns.py          # BHARs vs matched controls, stats, spreads, daily CAR paths
python3 s05_footprints.py       # 13F pods, short interest, turnover
python3 s06_capacity.py         # ADV, spreads, deployable $, gross P&L
python3 s07_report.py           # output/results_tables.md + charts
