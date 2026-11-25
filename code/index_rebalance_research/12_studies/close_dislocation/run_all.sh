#!/bin/zsh
# Close dislocation and reversal on index trade days (US Russell, S&P/TSX, STOXX 600, DAX family).
# About 25 minutes: R2 reads ~10 min (dsf_v2, secd, g_secd and ds2primqtprc date-filtered scans), builders ~10 min.
# Needs R2 credentials (python3 -m r2 --check-keys) and the sister studies' caches:
#   ../russell_june/_cache (s01_fetch.py), ../tsx_composite/data (01_pull_data.py, 03_panel.py),
#   ../stoxx600_dax/data (01_pull_data.py, 03_panel.py) plus their event, prediction and CAR CSVs.
# s12 reads data/alpaca_1559_2025.csv, which was copied out of the Alpaca MCP by hand (see the script); it is not re-pulled.
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"   # code/
STUDY=$ROOT/index_rebalance_research/12_studies/close_dislocation
export PYTHONPATH=$ROOT:$STUDY
cd $STUDY
python3 s01_pull_data.py pi ftse opens_ca opens_eu opens_ds opens_us   # R2 -> data/*.parquet
python3 s02_pi.py              # passive shares per family -> table_pi.csv
python3 s03_us.py              # Russell events -> data/legs_us.parquet
python3 s04_ca.py              # S&P/TSX events -> data/legs_ca.parquet
python3 s05_eu.py              # STOXX 600 + DAX family events -> data/legs_eu.parquet
python3 s06_tables.py          # events_legs.csv, table_legs.csv, table_dax_moves.csv
python3 s07_regressions.py     # table_regressions.csv
python3 s08_tax.py             # table_tax.csv
python3 s09_economics.py       # table_lp_pnl.csv, table_pricing.csv
python3 s10_checks.py          # table_checks.csv, table_checks_volume_by_date.csv
python3 s11_borrow.py          # table_borrow_2025_26.csv (iBorrowDesk, descriptive)
python3 s12_auction_2025.py    # table_auction_2025.csv, auction_2025_names.csv (Alpaca SIP, 27 Jun 2025 only)
python3 s13_report.py          # report_tables.md
python3 s14_tsx_redesign.py     # table_tsx_redesign.csv (added 2026-10-14)
python3 s15_tsx_costs.py        # table_tsx_costs.csv, tsx_adds_post2021_liquidity.csv (R2 pull; added 2026-10-14)
python3 s16_pricing_post2021.py # table_pricing_post2021.csv (S&P/TSX pricing rows since 18 Oct 2021; added 2026-10-15)
