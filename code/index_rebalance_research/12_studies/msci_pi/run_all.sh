#!/bin/zsh
# MSCI passive share: time-varying visible pi and the true-to-visible multiplier R for MSCI Canada and MSCI Europe
# (Standard and Small Cap, European country splits), and the rows appended to ../net_flow/pi_calibration.csv.
# About 3 minutes when data/ishch is cached (the first run of s01 makes ~700 polite requests, ~30 minutes).
# Needs R2 credentials (python3 -m r2 --check-keys) and the sister studies' panels:
#   ../tsx_composite/data/panel.npz + panel_lines.parquet, ../stoxx600_dax/data/panel.npz + panel_lines.parquet,
#   ../net_flow/common.py (fe_ols, winsor), ../close_dislocation/data/index_mv.parquet (checks only).
# NOTE: ../net_flow/run_all.sh (s08) rewrites pi_calibration.csv from scratch; re-run s07 here afterwards.
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # code/
STUDY=$ROOT/index_rebalance_research/12_studies/msci_pi
export PYTHONPATH=$ROOT:$STUDY
cd $STUDY
python3 s01_fetch_ishch.py > data/s01_fetch.log         # iShares UCITS daily holdings around each review + quarter-ends (cached)
python3 s02_pull_r2.py refs s12agg fnames dsetf isinmap  # R2: s12 rulers, s12 tracker aggregates, fund names, ETF shares, ISIN history
python3 s03_pi.py                                       # visible pi per quarter, segment, market/country -> data/pi_quarterly*.parquet
python3 s05_pi_series.py                                # dated family series, extended to 2026-07 -> data/pi_series.parquet
python3 s04_events.py > data/s04_events.log             # holdings-implied index changes per review -> events.csv
python3 -W ignore s06_calibrate.py > data/s06.log       # demand.csv, table_R.csv, table_R_by_type.csv
python3 s07_write_pi.py                                 # backup + append rows to ../net_flow/pi_calibration.csv
python3 -W ignore s08_checks.py > data/s08.log          # table_checks.csv, table_known_events.csv
python3 -W ignore s09_report.py                         # report_tables.md, table_pi_summary.csv
