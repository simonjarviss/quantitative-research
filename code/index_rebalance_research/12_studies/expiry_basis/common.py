"""Shared settings for the expiry_basis study (12_studies/expiry_basis/).

Question (plan item 7, data part): when a desk hedges an index-day risk basket with index futures, how far does the
cash index move in the closing auction after the futures have settled, on rebalance/expiry days vs normal days, and
does that move persist? Measured from Datastream futures daily settlements vs Datastream index closes (R2).
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
START = "2008-01-01"

# (future ticker, Datastream futures contrcode, Datastream dsindexcode, label). Index = total-return series `ri`
# (so ex-dividend drops cancel); DAX is itself a performance index.
PAIRS = [("FESX", 801, 13210, "EURO STOXX 50"), ("FXXP", 3284, 13635, "STOXX 600"), ("FDAX", 775, 13095, "DAX"),
         ("Z", 1691, 36888, "FTSE 100"), ("Y2", 4164, 36890, "FTSE 250"), ("FSMI", 2557, 47832, "SMI"),
         ("FIE", 1790, 37778, "IBEX 35"), ("FIB", 1926, 77161, "FTSE MIB"), ("FCE", 647, 35252, "CAC 40"),
         ("FTI", 630, 124, "AEX"), ("SXF", 298, 48553, "S&P/TSX 60"), ("SCF", 2674, 48552, "S&P/TSX Comp")]

# MSCI review implementation closes, from 12_studies/msci_pi/common.py (REVIEWS), to the end of R2 data.
MSCI = pd.to_datetime([
    "2018-02-28", "2018-05-31", "2018-08-31", "2018-11-30", "2019-02-28", "2019-05-28", "2019-08-27", "2019-11-26",
    "2020-02-28", "2020-05-29", "2020-08-31", "2020-11-30", "2021-02-26", "2021-05-27", "2021-08-31", "2021-11-30",
    "2022-02-28", "2022-05-31", "2022-08-31", "2022-11-30", "2023-02-28", "2023-05-31", "2023-08-31", "2023-11-30",
    "2024-02-29", "2024-05-31", "2024-08-30", "2024-11-25", "2025-02-28", "2025-05-30", "2025-08-26", "2025-11-24",
    "2026-02-27", "2026-05-29"])

TYPES = ["1 Q-expiry/rebal", "2 MSCI review", "3 quarter-end", "4 other month-end", "5 monthly expiry", "6 other"]


def third_friday(y: int, m: int) -> pd.Timestamp:
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)
