#!/bin/zsh
# Reproduce the STOXX Europe 600 / DAX-family study end to end (~35 min; R2 reads ~25 min, data/ ~0.7 GB).
# Needs R2 credentials (python3 -m r2 --check-keys) and the r2 package on PYTHONPATH.
set -e
STUDY="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$(cd "$STUDY/../../.." && pwd):$STUDY"   # code/ (for r2) and the study
cd $STUDY
python3 01_pull_data.py members lines secm fx index exch eushort sdc   # pass 1: membership, month-end prices (~6 min)
python3 02_universe.py                                                  # candidate universe -> data/universe_lines.parquet
python3 01_pull_data.py secd dsnames float worldscope dsprim           # pass 2: daily prices, Datastream (~10 min)
python3 03_panel.py                                                     # daily panel, float, index proxies
python3 04_events.py                                                    # STOXX 600 + DAX-family events
python3 05_predict_stoxx.py                                             # PIT STOXX 600 predictions (H6/H3/C)
python3 06_predict_dax.py                                               # PIT DAX/MDAX/SDAX/TecDAX predictions
python3 07_returns.py                                                   # matched-control and index-adjusted CARs, paths
python3 08_volume.py prep                                               # control lists + names for the MTF pull
python3 01_pull_data.py dsmtf                                           # lit MTF volume (~6 min)
python3 08_volume.py                                                    # turnover footprint, trade-day multiples
python3 09_shorts.py                                                    # pod net-short disclosures (EU SSR)
python3 10_report.py                                                    # table_*.csv, report_tables.md
python3 11_capacity.py                                                  # capacity, off-cycle
python3 12_write_results.py                                             # RESULTS.md from RESULTS_template.md + tables
