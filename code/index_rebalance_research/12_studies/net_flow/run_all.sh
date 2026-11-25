#!/bin/zsh
# Net passive index demand per stock from replicating-ETF holdings; time-varying visible pi; R = true / visible;
# the flow test on the close; size of weight-change flows.
# About 20 minutes when data/ishca is cached (the first run of s01 makes ~700 polite requests, ~35 minutes).
# Needs R2 credentials (python3 -m r2 --check-keys) and the sister studies' files:
#   ../close_dislocation (events_legs.csv, legs.py, common.py, data/etf_aum.parquet, index_mv.parquet, opens_*.parquet)
#   ../tsx_composite (reviews.csv, events.csv, data/panel.npz, member_matrix.npy, panel_lines.parquet, float_monthly.parquet)
#   ../russell_june (output/events_actual.csv, _cache/dsf, _cache/ccm.parquet, common.py)
#   ../stoxx600_dax (reviews_*.csv, events_stoxx.csv, data/panel.npz, data/panel_lines.parquet)
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # code/
STUDY=$ROOT/index_rebalance_research/12_studies/net_flow
export PYTHONPATH=$ROOT:$STUDY
cd $STUDY
python3 s01_fetch_ishca.py > data/fetch_ishca.log   # iShares Canada XIC/XIU holdings around each review (cached)
python3 s02_pull_r2.py                              # R2 reads -> data/*.parquet (CRSP MF, s12 + cusip map, Datastream)
python3 s03_pi.py                                   # visible pi per family -> data/pi_daily.parquet, pi_us_monthly
python3 s04_ca_demand.py                            # Canada demand per review x line -> ca_demand.csv
python3 s05_us_demand.py                            # Russell + S&P 500 demand -> us_demand.csv, us_sp500_*.csv
python3 s06_eu_demand.py                            # STOXX 600 / DAX family / FTSE UK demand -> eu_demand.csv
python3 -W ignore s07_legs.py                       # close legs for every line -> data/legs_net.parquet
python3 -W ignore s08_calibrate.py                  # R fits -> table_R.csv, table_R_selected.csv, pi_calibration.csv
python3 -W ignore s09_flow_test.py                  # flow test -> table_flow_test.csv, table_flow_terciles.csv, table_flow_wctype_ca.csv
python3 -W ignore s10_sizes.py                      # sizes -> table_sizes_*.csv
python3 -W ignore s11_checks.py                     # checks -> table_checks.csv, table_known_events.csv
python3 -W ignore s12_report.py                     # report_tables.md, table_sizes_headline.csv, table_mean_legs.csv
