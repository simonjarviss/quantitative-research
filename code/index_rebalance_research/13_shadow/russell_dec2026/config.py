"""Settings for the December 2026 Russell US reconstitution shadow prediction (paper only).

Dates: RUS-DEC26-PR (lseg.com press release, 1 Sep 2026) and the spec
index_rebalance_research/11_methodology/specs/ftse_russell/russell_us.md.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"                  # date-stamped raw downloads (one folder per fetch date)
PRED = Path(os.environ.get("SHADOW_PRED", HERE / "predictions"))   # prediction outputs
SCORES = Path(os.environ.get("SHADOW_SCORES", HERE / "scores"))     # scoring outputs
LOG = Path(os.environ.get("SHADOW_LOG", HERE / "shadow_log.md"))    # append-only timestamp log (env overrides: tests)
for _d in (RAW, PRED, SCORES):
    _d.mkdir(exist_ok=True)

# ---- event calendar (December 2026 recon) ----
RANK_DAY = pd.Timestamp("2026-10-30")       # last business day of October
PRELIM = pd.Timestamp("2026-11-13")         # preliminary lists after the close
LOCKDOWN = pd.Timestamp("2026-11-30")       # Monday open
RECON = pd.Timestamp("2026-12-11")          # effective after the close
JUNE_RANK_DAY = pd.Timestamp("2026-04-30")  # rank day of the previous (June 2026) recon: used for "known excluded"
# NYSE holidays Oct 2026 - Jan 2027 (for trading-day counts)
HOLIDAYS = ["2026-11-26", "2026-12-25", "2027-01-01"]

# ---- rules (Russell US methodology v7.2) ----
N_R1000, N_R3000, N_R3000E = 1000, 3000, 4000
BAND_1000 = 2.5        # +/- cumulative-cap percentile points around #1000
BAND_2000 = 0.5        # +/- around #2000 (Microcap / R2500 boundaries only; does not move R2000 membership)
MIN_PRICE = 1.0
MIN_CAP = 30e6
MIN_FLOAT = 0.05
BDI_COUNTRIES = {"Bermuda", "Cayman Islands", "British Virgin Islands", "Jersey", "Guernsey", "Isle of Man",
                 "Marshall Islands", "Liberia", "Panama", "Curacao", "Gibraltar", "Bahamas", "Puerto Rico"}

# Backtest refinement (12_studies/russell_june/s03_predict.py known_excluded): companies our rebuild ranked
# inside the R3000 on the previous rank day that Russell still left out are treated as Russell-ineligible.
# Backtest used ranks 1001..2700; live data has a weaker country screen, so default 1..2700 (see README).
EXCL_MINRANK, EXCL_MAXRANK = 1, 2700

# ---- confidence model ----
SIGMA_DATA = 0.05        # log-cap measurement error vs FTSE's own shares (clean single-class names)
SIGMA_DATA_FLAG = 0.15   # ... for names with share-count flags (multi-class, implied vs listed mismatch, fallback)
VOL_WINDOW = 60          # trading days for relative volatility

# ---- sources ----
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
ISHARES = {   # fund: (product id, slug)
    "IWB": ("239707", "ishares-russell-1000-etf"),
    "IWM": ("239710", "ishares-russell-2000-etf"),
    "IWV": ("239714", "ishares-russell-3000-etf"),
}
ISHARES_URL = "https://www.ishares.com/us/products/{pid}/{slug}/latest-holdings.csv"
ISHARES_AJAX = ("https://www.ishares.com/us/products/{pid}/{slug}/1467271812596.ajax"
                "?fileType=csv&fileName={fund}_holdings&dataType=fund")   # returned HTML on 2026-10-09
SYMDIR = {"nasdaqlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
          "otherlisted": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"}
NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
# FTSE Russell public reconstitution lists (PDF; Company / Symbol / Industry). Prelim date = list date.
FTSE_LIST_URL = "https://www.lseg.com/content/dam/ftse-russell/en_us/documents/other/{kind}-{side}-{yyyymmdd}.pdf"
# kind: ru3000 | rmicro ; side: additions | deletions.  Dec 2026 prelim expected: ...ru3000-additions-20261113.pdf

CATS = ["ADD_R2000", "MIG_R1000_TO_R2000", "DEL_R2000", "MIG_R2000_TO_R1000", "ADD_R1000", "DEL_R1000"]
CAT_LABEL = {"ADD_R2000": "new R2000 add (out->R2000)", "MIG_R1000_TO_R2000": "R1000->R2000",
             "DEL_R2000": "R2000 delete (R2000->out)", "MIG_R2000_TO_R1000": "R2000->R1000",
             "ADD_R1000": "R1000 add (out->R1000)", "DEL_R1000": "R1000 delete (R1000->out)"}
BACKTEST_GROUP = {"ADD_R2000": "BUY_add", "MIG_R1000_TO_R2000": "BUY_mig", "DEL_R2000": "SELL_del",
                  "MIG_R2000_TO_R1000": "SELL_mig", "ADD_R1000": "OTHER", "DEL_R1000": "OTHER"}
SIDE = {"ADD_R2000": 1, "MIG_R1000_TO_R2000": 1, "ADD_R1000": 1,
        "DEL_R2000": -1, "MIG_R2000_TO_R1000": -1, "DEL_R1000": -1}


def trading_days_between(a, b) -> int:
    """Trading days after a, up to and including b (0 if b <= a)."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if b <= a:
        return 0
    import numpy as np
    return int(np.busday_count((a + pd.Timedelta(days=1)).date(), (b + pd.Timedelta(days=1)).date(),
                               holidays=HOLIDAYS))
