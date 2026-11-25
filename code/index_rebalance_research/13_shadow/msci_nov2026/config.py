"""Settings for the MSCI November 2026 Index Review shadow prediction (paper only; public data; no accounts).

Rules: index_rebalance_research/11_methodology/specs/msci/ (gimi_size_segments.md = "gimi", msci_canada.md = "ca",
msci_europe.md = "eu", _shared.md = "shared", params/*.yaml). Section references are given next to each value.
Tags: V = from a primary MSCI document via the specs; I = our inference / approximation.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PACK = HERE.parent.parent                                   # index_rebalance_research/
RAW = HERE / "raw"                                          # date-stamped raw downloads, raw/<fetch-date>/
CACHE = HERE / "cache"                                      # R2 pulls and yfinance caches (regenerable)
PRED = Path(os.environ.get("SHADOW_PRED", HERE / "predictions"))
SCORES = Path(os.environ.get("SHADOW_SCORES", HERE / "scores"))
REPLAY = Path(os.environ.get("SHADOW_REPLAY", HERE / "replay"))
LOG = Path(os.environ.get("SHADOW_LOG", HERE / "shadow_log.md"))
for _d in (RAW, CACHE, PRED, SCORES, REPLAY):
    _d.mkdir(exist_ok=True)
STUDY_PI = PACK / "12_studies" / "msci_pi"                  # parallel study: polite fetcher + cached CH holdings
ISHCH_CACHE = STUDY_PI / "data" / "ishch"                   # {FUND}_{YYYYMMDD}.json around past reviews (read-only)
TSX_SHADOW = PACK / "13_shadow" / "tsx_dec2026"             # read-only: TMX MiG, TSX directory, yfinance cache
EU_PANEL = PACK / "12_studies" / "stoxx600_dax" / "data"     # read-only: European R2 panel (to 2026-06-30)
CA_PANEL = PACK / "12_studies" / "tsx_composite" / "data"    # read-only: Canadian R2 panel (to 2026-06-26)
ENGINE = PACK / "16_flow_engine"

# ------------------------------------------------------------------ calendar (gimi §5 table; ca §5; eu §0)
REVIEW = "2026-11"
UNIVERSE_CUTOFF = pd.Timestamp("2026-08-31")      # last business day of Aug (gimi §3.1 data dates)            V/I
LIQUIDITY_CUTOFF = pd.Timestamp("2026-09-30")     # last business day of Sep                                  V/I
CUTOFF_WINDOW = pd.bdate_range("2026-10-19", "2026-10-30")   # "any one of the last 10 business days" (gimi §3.1) V rule, I dates
CUTOFF_LIKELY = [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")]  # observed early-window pattern (gimi §5): I
ANNOUNCE = pd.Timestamp("2026-11-11")             # ~23:00 CET, after the US close (gimi §5; MSCI-IRD)          V
TRADE = pd.Timestamp("2026-11-30")                # trade at the close; effective Tue 1 Dec (MSCI-IRD)           V
EFFECTIVE = pd.Timestamp("2026-12-01")
T_PLUS5 = pd.Timestamp("2026-12-07")              # 5 sessions after the trade close (TSX/NYSE/LSE all open)
# previous reviews (for the cutoff bracket and the calibration replay)
# cutoff = MSCI data stamp of that review (gimi §5 "Observed Price Cutoff Dates"; Feb 2026 from the archived Feb 2026
# GIMI book p.25 "Data as of the close of January 16, 2026"); pre/post = holdings dates around the trade close;
# gmsr = DM Standard GMSR of that review (Aug: gimi §2.3.2.1 p.25; May: 2 x MSCI-MCF p.3; Feb: GIMI-FEB26 p.25)
PREV = {"2026-08": dict(cutoff="2026-07-20", trade="2026-08-31", pre="2026-08-28", post="2026-08-31", post2="2026-09-01",
                        gmsr=16_276e6),
        "2026-05": dict(cutoff="2026-04-20", trade="2026-05-29", pre="2026-05-28", post="2026-05-29", post2="2026-06-01",
                        gmsr=15_748e6),
        "2026-02": dict(cutoff="2026-01-16", trade="2026-02-27", pre="2026-02-26", post="2026-02-27", post2="2026-03-02",
                        gmsr=15_200e6)}

# ------------------------------------------------------------------ size rules (gimi §2.3, §3.2; params/gimi_size_segments.yaml)
GMSR_STD_AUG26 = 16_276e6         # DM Standard GMSR, USD full cap, data as of 20 Jul 2026 (gimi §2.3.2.1 p.25)  V
GMSR_STD_MAY26 = 15_748e6         # = 2 x USD 7,874m DM Standard company-size criterion, May 2026 (MSCI-MCF p.3)  I (x2)
GMSR_RANGE = (0.5, 1.15)          # cutoffs bounded to 0.5-1.15 x GMSR (gimi §2.3.2, §11 item 5)                V
BUF_LO, BUF_HI = 2 / 3, 1.5       # migration buffers on FULL company cap (gimi §3.1.5.1 pp.43-44)              V
FLOAT_NEW = 0.5                   # Standard float cap >= 0.5 x C for new / migrating-up names (gimi §2.3.6.1)  V
FLOAT_EXIST = 0.5 * 2 / 3         # existing members: 2/3 of that threshold (gimi §3.1.6.2)                     V
FIF_MIN = 0.15                    # FIF < 0.15 needs 1.8x the float-cap threshold (gimi §2.2.6, §2.3)            V
LOW_FIF_MULT = 1.8
EPI_FIF_EXEMPT = 0.75             # FIF >= 0.75 not subject to EPI (gimi §2.3.6.3, Aug 2026)                     V
EPI = [(5, 1.0), (10, 1.0), (15, 1.0), (20, 1.0), (25, 2.0), (30, 2.0), (35, 2.0), (40, 2.0), (45, 4.0), (50, 4.0),
       (55, 4.0), (60, 4.0), (90, 5.0), (120, 8.0), (150, 15.0), (180, 15.0), (250, 25.0)]   # (Mon-Fri days, excess return)
UMSR_AUG26 = 585e6                # Equity Universe Minimum Size Requirement (gimi §2.2 table)                    V
# liquidity, DM (gimi §2.2.5 p.19; §3.1.2.4 p.38): new entrants / existing IMI constituents
ATVR_NEW_12M, ATVR_NEW_3M, FOT_NEW = 0.20, 0.20, 0.90
ATVR_EX_12M, ATVR_EX_3M, FOT_EX = 0.20 * 2 / 3, 0.05, 0.80
SEASONING_MONTHS = 3              # IPO traded >= 3 months before implementation (gimi §2.2.7)                   V

# ------------------------------------------------------------------ probability model (I; as the TSX / Russell shadows)
SIGMA_DATA = 0.06                 # log full-cap error vs MSCI (shares, unlisted classes, FX)                     I
SIGMA_DATA_FLAG = 0.15            # names with share-count / class / mapping flags                               I
SIGMA_CUTOFF = {"CA": 0.08, "EU": 0.07}   # log error of our reconstructed and drifted market cutoff C             I
REL_VOL_WINDOW = 60               # trading days for the stock-vs-DM-market relative volatility
KNOWN_EXCL_FACTOR = 0.25          # non-member that already qualified on size at the previous review and was not added;
                                  # May 2026: Abivax and Var Energi were in this set and were added (blocks such as EPI
                                  # or seasoning lapse)                                                              I
CUTOFF_DRIFT = 1.07               # market cutoffs rose faster than the members' prices: bracket(next) / (bracket(prev)
                                  # x price scale) = CA 1.088 (Feb->May), 1.140 (May->Aug); EU ~1.08 (Feb->May, Grifols
                                  # excluded), 0.980 (May->Aug); geometric mean ~1.07 (NOC reductions as the DM GMSR
                                  # rises; Standard counts fell 395 -> 376 in Europe and 83 -> 80 in Canada in 2026)   I
FLOAT_EXIST_BITE = 0.85           # share of existing-member float-test fails that MSCI acts on (calibrated on Aug 2026)  I

# ------------------------------------------------------------------ sources
UA_NOTE = "all requests via 12_studies/msci_pi/polite.py (neutral UA, >= 1.6 s apart across processes)"
CH_JSON = "https://www.ishares.com/ch/individual/en/products/{pid}/x/1495092304805.ajax?tab=all&fileType=json"
CH_FUNDS = {   # iShares UCITS (Ireland), physical; CH site serves dated holdings with ISIN (probe 2026-10-12)
    "CSCA": dict(pid=253721, index="MSCI Canada (Standard)", seg="STD", mkt="CA"),
    "IMEA": dict(pid=251861, index="MSCI Europe (Standard)", seg="STD", mkt="EU"),
    "WSML": dict(pid=296576, index="MSCI World Small Cap", seg="SC", mkt="DM"),
    "SWDA": dict(pid=251882, index="MSCI World (Standard)", seg="STD", mkt="DM"),
}
US_CSV = "https://www.ishares.com/us/products/{pid}/{slug}/latest-holdings.csv"   # latest only; asOfDate ignored
US_FUNDS = {"EWC": (239615, "ishares-msci-canada-etf"), "IEUR": (264617, "ishares-core-msci-europe-etf"),
            "EFA": (239623, "ishares-msci-eafe-etf"), "EWU": (239690, "ishares-msci-united-kingdom-etf"),
            "EWG": (239650, "ishares-msci-germany-etf"), "EWQ": (239648, "ishares-msci-france-etf"),
            "EWL": (239685, "ishares-msci-switzerland-capped-etf"), "EWN": (239671, "ishares-msci-netherlands-etf"),
            "EWD": (239684, "ishares-msci-sweden-etf")}
MSCI_LIST = {"STD": "https://www.msci.com/eqb/gimi/stdindex/MSCI_{mon}{yy}_STPublicList.pdf",
             "SC": "https://www.msci.com/eqb/gimi/smallcap/MSCI_{mon}{yy}_SCPublicList.pdf"}   # msci/README, manifest §3
MSCI_PR = "https://www.msci.com/downloads/documents/press-releases/media-room/MSCI_{mon}{yy}_QIRPR.pdf"

# DM Europe countries (eu §1; App I) -> ISO2; MSCI list section names
EU_COUNTRIES = {"AT": "AUSTRIA", "BE": "BELGIUM", "DK": "DENMARK", "FI": "FINLAND", "FR": "FRANCE", "DE": "GERMANY",
                "IE": "IRELAND", "IT": "ITALY", "NL": "NETHERLANDS", "NO": "NORWAY", "PT": "PORTUGAL", "ES": "SPAIN",
                "SE": "SWEDEN", "CH": "SWITZERLAND", "GB": "UNITED KINGDOM"}
REPORT_FIRST = ["GB", "FR", "DE", "CH", "NL", "SE"]
