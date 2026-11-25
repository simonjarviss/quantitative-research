#!/bin/sh
# expiry_basis: pull from R2, compute, write tables (about 2 minutes)
set -e
cd "$(dirname "$0")"
python3 s01_pull.py
python3 s02_auction_move.py
python3 s03_tables.py > /dev/null
