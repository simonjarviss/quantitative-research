"""Settings for the December 2026 European shadow predictions (paper only): STOXX Europe 600, the DAX family
(DAX, MDAX, SDAX, TecDAX) and the FTSE 100/250. All three trade the close of Fri 18 Dec 2026.

Rules: 11_methodology/specs/stoxx/stoxx_europe_600.md, dax_family.md, _shared.md, params/*.yaml;
       11_methodology/specs/ftse_russell/ftse_uk_100_250_allshare.md, _shared.md.
Dates: 02_markets/event_calendar.md and the spec calendars (cited per constant below).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACK = HERE.parent.parent                                   # index_rebalance_research/
RAW = HERE / "raw"                                          # raw/<fetch date>/... (one folder per fetch date)
PRED = Path(os.environ.get("SHADOW_PRED", HERE / "predictions"))
SCORES = Path(os.environ.get("SHADOW_SCORES", HERE / "scores"))
LOG = Path(os.environ.get("SHADOW_LOG", HERE / "shadow_log.md"))
ENGINE = PACK / "16_flow_engine"                            # read-only imports (holdings parser, pi.index_mv)
STUDY = PACK / "12_studies" / "stoxx600_dax"                # read-only (backtest accuracy tables)
ENGINE_INPUTS = ENGINE / "inputs" / "2026-12-18"            # we write only stoxx600.csv, dax_family.csv, ftse_uk.csv
for _d in (RAW, PRED, SCORES):
    _d.mkdir(exist_ok=True)

FAMILIES = ["stoxx600", "dax", "ftse_uk"]

# ------------------------------------------------------------------------------------------ calendar
TRADE = pd.Timestamp("2026-12-18")          # 3rd Friday; all three families trade this close
CAL = {
    # STOXX Europe 600 [stoxx_europe_600.md §5 table "Next four reviews"; STOXX-MG §5.2, §7.2.2; STOXX-CG §8.2]
    "stoxx600": dict(
        list_cutoff_oct=pd.Timestamp("2026-10-30"),   # Nov monthly selection list cut-off (published Mon 2 Nov)
        ff_freeze=pd.Timestamp("2026-11-23"),         # free float and shares cut-off (5 trading days before C)
        cutoff=pd.Timestamp("2026-11-30"),            # ranking cut-off C (prices)
        announce=pd.Timestamp("2026-12-01"),          # first trading day of Dec (time of day not stated)
        uda=pd.Timestamp("2026-12-11"),
        trade=TRADE),
    # DAX family [dax_family.md §5 table; DAX-MG §4.2.1 (cut-off 17:30 CET, list after 22:00 CET on 3rd trading day),
    # §7.1.2 (20-day VWAP; Fast rules only in Jun/Dec); DAX-PR-202609 confirms 3 Dec 2026]
    "dax": dict(
        list_cutoff_oct=pd.Timestamp("2026-10-30"),   # Nov selection list cut-off; list published Wed 4 Nov 22:00 CET
        cutoff=pd.Timestamp("2026-11-30"),            # Dec selection list cut-off (17:30 CET)
        announce=pd.Timestamp("2026-12-03"),          # after 22:00 CET
        weights_freeze=pd.Timestamp("2026-12-10"),    # t-6
        uda=pd.Timestamp("2026-12-11"),
        trade=TRADE),
    # FTSE UK [ftse_uk_100_250_allshare.md §5 table; FTSE-UK-GR §8.1.1; FTSE-UK-FAQ26 p.3-4]
    "ftse_uk": dict(
        shares_cutoff=pd.Timestamp("2026-10-30"),     # shares/float cut-off for the Dec review (Monitored List)
        indicative_data=pd.Timestamp("2026-11-20"),   # indicative list data date
        indicative=pd.Timestamp("2026-11-24"),        # indicative FTSE 100/250 list published
        cutoff=pd.Timestamp("2026-12-01"),            # Tue before the first Friday: rank cut-off (close)
        announce=pd.Timestamp("2026-12-02"),          # Wed after the UK close
        trade=TRADE),
}
# Holidays between now and Jan 2027 (trading-day counts); Xetra: 24, 25, 31 Dec; LSE: 25, 28 Dec, 1 Jan
HOLIDAYS = {"XETR": ["2026-12-24", "2026-12-25", "2026-12-31", "2027-01-01"],
            "XLON": ["2026-12-25", "2026-12-28", "2027-01-01"],
            "EU": ["2026-12-25", "2027-01-01"]}


def trading_days_between(a, b, cal: str = "EU") -> int:
    """Trading days after a, up to and including b (0 if b <= a)."""
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if b <= a:
        return 0
    return int(np.busday_count((a + pd.Timedelta(days=1)).date(), (b + pd.Timedelta(days=1)).date(),
                               holidays=HOLIDAYS.get(cal, [])))


# ------------------------------------------------------------------------------------------ rules
# STOXX Europe 600 [STOXX-MG §7.2.2, p.67]: ranks 1-550 in; members 551-750 fill to 600 in rank order; then largest.
STOXX = dict(n=600, upper=550, lower=750, cap=0.20)
# DAX family, Fast rules only in December [DAX-MG §7.1.2 "Overview of rules", p.37; params/dax_family.yaml]
DAX_N = dict(DAX=40, MDAX=50, SDAX=70, TecDAX=30)
DAX_THR = dict(DAX=dict(fx=60, fe=33, rx=53, rxa=47, re=40, rea=47),
               MDAX=dict(fx=110, fe=83, rx=103, rxa=97, re=90, rea=97),
               SDAX=dict(fx=180, fe=153, rx=173, rxa=167, re=160, rea=167),
               TecDAX=dict(fx=45, fe=25, rx=40, rxa=35, re=30, rea=35))
DAX_REGULAR = False                         # December: Fast Exit / Fast Entry only [DAX-MG §7.1.2, p.36-37]
DAX_CAP = 0.15                              # 15% single cap + group-entity 30/15 [DAX-MG §7.1.1, §5.11.1]
VWAP_DAYS = 20                              # 20-trading-day adjusted Xetra VWAP [DAX-MG §7.1.2, p.34; §5.12]
# FTSE UK [FTSE-UK-GR §8.3.4, p.19]: FTSE 100 in <=90, out >=111; FTSE 250 in <=325, out >=376; constant counts.
FTSE = dict(n100=100, in100=90, out100=111, n250=250, in250=325, out250=376)

# ------------------------------------------------------------------------------------------ confidence model
SIGMA_DATA = dict(member=0.04, nonmember=0.06, flagged=0.12)   # log-FFMC / log-cap measurement error at the cut-off
VOL_WINDOW = 60                                                  # trading days for relative volatility
P_ENGINE_MIN = 0.20                                              # engine rows: P(change) >= this
WATCH_MARGIN = np.log(1.15)                                      # watchlist: unchanged names within 15% of a line
N_MC = 4000                                                      # Monte Carlo cross-check paths

# ------------------------------------------------------------------------------------------ sources
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
MIN_GAP_S = 1.6                     # >= 1.5 s between requests (task rule)
# STOXX public selection lists ("slpublic" files are unrestricted; stoxx.com > Resources > Reports > Selection Lists)
SL_PAGE = ("https://www.stoxx.com/selection-lists?p_p_id=STOXXNewSelectionlistportlet_WAR_STOXXNewSelectionlistportlet"
           "&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_cacheability=cacheLevelPage&p_p_col_id=column-1"
           "&p_p_col_count=2")
SL_PARAM = "_STOXXNewSelectionlistportlet_WAR_STOXXNewSelectionlistportlet_"
SL_FILE = "https://www.stoxx.com/document/Reports/{kind}SelectionList/{yyyy}/{Month}/{fname}"
SL_SYMBOL = {"stoxx600": ("STOXX", ["sxxp"]), "dax": ("DAX", ["daxk", "mkdx", "sdxk", "tdxk"])}
# iShares UCITS holdings (16_flow_engine/holdings.py, parser reused read-only; cache redirected to raw/<date>/)
FUNDS = {"EXSA": "STOXX600", "EXS1": "DAX", "EXS3": "MDAX", "EXS2": "TECDAX", "ISF": "FTSE100", "MIDD": "FTSE250"}
FTSE_NOTICES = ("https://research.ftserussell.com/products/index-notices/Backend/GetNotices?id=UK&productTitle="
                "&pageNum={page}&pageSize=50&ts=1")
STOXX_RSS = "https://stoxx.com/feed/"
LEVEL_TICKERS = {"STOXX600": "^STOXX", "FTSE100": "^FTSE", "FTSE250": "^FTMC", "DAX": "^GDAXI", "MDAX": "^MDAXI",
                 "TECDAX": "^TECDAX", "SDAX": "^SDAXI"}
FX_TICKERS = {"CAD": "CADEUR=X", "GBP": "GBPEUR=X", "CHF": "CHFEUR=X", "SEK": "SEKEUR=X", "DKK": "DKKEUR=X", "NOK": "NOKEUR=X",
              "PLN": "PLNEUR=X", "USD": "USDEUR=X", "CZK": "CZKEUR=X", "HUF": "HUFEUR=X"}
PRICE_START = "2026-01-02"           # history for vol, the June calibration replay and the 30 Sep anchors

# ------------------------------------------------------------------------------------------ backtest accuracy
# 12_studies/stoxx600_dax/RESULTS.md §8 (rules model; precision / recall). H3 = 3 calendar months before T,
# C = ranking cut-off. The initial run (12 Oct close) sits between H3 (18 Sep) and C (30 Nov): quote H3; cut-off runs quote C.
# DAX rows for Fast-only (Jun/Dec) reviews of era 3 are recomputed in README from predictions_dax.csv.
BACKTEST = {
    ("stoxx600", "add"): {"H6": (0.07, 0.14), "H3": (0.15, 0.34), "C": (0.24, 0.67)},
    ("stoxx600", "delete"): {"H6": (0.10, 0.14), "H3": (0.25, 0.38), "C": (0.36, 0.71)},
    ("dax", "promotion"): {"H6": (0.11, 0.30), "H3": (0.17, 0.45), "C": (0.26, 0.74)},
    ("dax", "demotion"): {"H6": (0.14, 0.14), "H3": (0.19, 0.19), "C": (0.33, 0.43)},
}
# Per-index rows of the same table (rules model, DAX family; precision, recall), used to tag each DAX-family call.
BACKTEST_DAX_INDEX = {
    ("DAX", "add"): {"H6": (0.18, 0.59), "H3": (0.24, 0.67), "C": (0.28, 0.88)},
    ("DAX", "delete"): {"H6": (0.19, 0.28), "H3": (0.19, 0.23), "C": (0.24, 0.42)},
    ("MDAX", "add"): {"H6": (0.11, 0.27), "H3": (0.18, 0.39), "C": (0.29, 0.73)},
    ("MDAX", "delete"): {"H6": (0.16, 0.29), "H3": (0.23, 0.37), "C": (0.31, 0.63)},
    ("SDAX", "add"): {"H6": (0.10, 0.21), "H3": (0.17, 0.32), "C": (0.30, 0.61)},
    ("SDAX", "delete"): {"H6": (0.10, 0.18), "H3": (0.15, 0.25), "C": (0.26, 0.49)},
    ("TECDAX", "add"): {"H6": (0.06, 0.28), "H3": (0.13, 0.51), "C": (0.16, 0.66)},
    ("TECDAX", "delete"): {"H6": (0.33, 0.06), "H3": (0.73, 0.21), "C": (0.85, 0.43)},
}
# FTSE UK: the pack has no historical backtest. The only calibration is the June 2026 replay (one annual review,
# scores/validate_june2026.md); quoted per call as such.
BACKTEST_FTSE_NOTE = "no historical backtest in the pack; June 2026 replay only"
# FTSE UK names whose free-data cap is known to disagree with FTSE's (flagged on every call, probability unchanged)
FTSE_KNOWN = {"GB00BMBVGQ36": "Harbour Energy: June 2026 replay predicted a FTSE 100 entry in all three variants and it "
                              "did not happen (FTSE's count for the listed line is likely below Yahoo/Datastream's)"}
