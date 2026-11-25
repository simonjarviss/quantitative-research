#!/bin/zsh
# MSCI review closes (MSCI Canada, MSCI Europe Standard changes, Feb 2018 - May 2026): close dislocation and reversal.
# About 15 minutes: R2 reads ~13 min (ds2primqtprc / ds2primqtri, sorted by infocode, so the id lists touch most row
# groups), builders ~1 min. Needs R2 credentials (python3 -m r2 --check-keys) and, read-only:
#   ../msci_pi/demand.csv and common.REVIEWS; ../close_dislocation/common.py, legs.py, events_legs.csv, table_pricing.csv;
#   ../tsx_composite/data (panel.npz, panel_lines.parquet, ds_names.parquet, tsx_calendar, member_matrix, index_proxy);
#   ../stoxx600_dax/data (panel.npz, panel_lines.parquet, eu_calendar, member_stoxx, index_proxy).
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # code/
STUDY=$ROOT/index_rebalance_research/12_studies/msci_close
export PYTHONPATH=$ROOT:$STUDY
cd $STUDY
python3 s01_pull_ds.py main ca_extra   # R2 -> data/ds_px.parquet (Datastream primary-quote prices + ri), data/ca_extra_ics.parquet
python3 s02_events.py                  # events_msci.csv (MSCI Standard events, sign, drop rules)
python3 s03_legs.py                    # data/legs_msci.parquet (legs, matched controls, Compustat cross-check)
python3 s04_tables.py                  # events_legs_msci.csv, table_legs.csv, table_diff.csv, table_flow.csv, table_by_review.csv
python3 s05_pricing.py                 # table_pricing_msci.csv (same columns as close_dislocation/table_pricing.csv)
python3 s06_checks.py                  # table_checks.csv
