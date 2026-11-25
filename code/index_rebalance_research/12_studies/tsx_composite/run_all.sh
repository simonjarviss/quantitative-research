#!/bin/sh
# Reproduce the study end to end. Run from anywhere; needs the r2 package (R2_export) on PYTHONPATH.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../.. && pwd)"   # code/, for r2
python3 01_pull_data.py            # R2 reads (~10 min): members lines secd index dsnames float consvol worldscope sdc
python3 02_events.py
python3 03_panel.py
python3 04_predict.py
FLOAT=all100 python3 04_predict.py # float sensitivity
python3 05_filters.py
python3 06_returns.py
python3 07_volume.py
python3 08_report.py
python3 09_capacity_offcycle.py
python3 10_13f.py                  # optional pod 13F footprint (reads tr_13f.s34)
python3 11_write_results.py        # RESULTS.md from RESULTS_template.md + table CSVs
