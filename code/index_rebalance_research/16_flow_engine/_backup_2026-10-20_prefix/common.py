"""Shared paths, the index registry, market constants and a polite HTTP helper for the net-flow engine.

Paper-only research. Every web request goes through `http_get`: neutral browser User-Agent, no identifiers,
>= 1.5 s between requests, and the response is cached on disk by the caller (data/...).

Sources of the constants (all in the research pack, ../):
  normal and event-day closing-auction shares  04_models/volume_forecasting.md §3.4, §4.3, §5.2
  passive shares pi and volume multipliers R   04_models/flow_sizing.md §4.4 (fallback only; see pi.py)
  close-pricing rows                           12_studies/close_dislocation/RESULTS.md §9, table_pricing.csv
  pricing layer v2 (2026-10-15)                + close_dislocation/table_pricing_post2021.csv (S&P/TSX Oct2021+),
                                               12_studies/msci_close/table_pricing_msci.csv (MSCI families),
                                               12_studies/expiry_basis/ (index-futures hedge gap), venue_rules.csv
                                               (03_microstructure close rules), SIX/HSG depth prior
                                               (03_microstructure/closing_auction_literature.md)
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                 # 16_flow_engine/
PACK = HERE.parent                                     # index_rebalance_research/
INPUTS = HERE / "inputs"
SHEETS = HERE / "sheets"
DATA = HERE / "data"
VALID = HERE / "validation"
R2DIR = DATA / "r2"
HOLD = DATA / "holdings"
MKT = DATA / "market"
for _d in (INPUTS, SHEETS, DATA, VALID, R2DIR, HOLD, MKT):
    _d.mkdir(parents=True, exist_ok=True)

STUDIES = PACK / "12_studies"
SHADOW_TSX = PACK / "13_shadow" / "tsx_dec2026"
SHADOW_RUSSELL = PACK / "13_shadow" / "russell_dec2026"
PI_CALIBRATION = STUDIES / "net_flow" / "pi_calibration.csv"     # produced by a parallel study (read-only here)
PI_OVERRIDES = HERE / "pi_overrides.csv"                         # optional, hand-edited
PRICING_TABLE = STUDIES / "close_dislocation" / "table_pricing.csv"            # v1: every family, 2016+ / all
PRICING_TSX_POST2021 = STUDIES / "close_dislocation" / "table_pricing_post2021.csv"   # CA S&P/TSX, Oct2021+
PRICING_MSCI = STUDIES / "msci_close" / "table_pricing_msci.csv"                     # CA/EU MSCI (+ UK split)
EXPIRY_BASIS = STUDIES / "expiry_basis"                                           # hedge gap (settle timing, sd)
VENUE_RULES = HERE / "venue_rules.csv"                                            # close rules per listing country
MOC_ARCHIVE = Path.home() / "index_archive" / "tsx"

# ------------------------------------------------------------------------------------------------ HTTP
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MIN_GAP_S = 2.0
_last_request = [0.0]


def http_get(url: str, timeout: int = 60, retries: int = 2) -> bytes:
    """GET with a neutral User-Agent and at least MIN_GAP_S seconds between requests (process-wide)."""
    last = None
    for k in range(retries + 1):
        wait = MIN_GAP_S - (time.time() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        try:
            _last_request[0] = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


# ------------------------------------------------------------------------------------------- registry
# One entry per index the engine knows. `market` is the listing market of the members for pricing/volume;
# pan-European indices take each member's own listing country. `pi_key` names the passive-share series.
# `etf` = the fully replicating iShares fund used for current weights and holdings diffs (None = no free file).
# `cap` = single-stock weight cap applied at the quarterly review (None = uncapped), per 11_methodology specs.
INDEX = {
    "SPTSX_COMP": dict(name="S&P/TSX Composite (Capped Composite for XIC)", region="CA", ccy="CAD", etf="XIC",
                       cap=0.10, pricing="CA S&P/TSX", note="XIC tracks the Capped Composite (10% cap)"),
    "SPTSX_60":   dict(name="S&P/TSX 60", region="CA", ccy="CAD", etf="XIU", cap=None, pricing="CA S&P/TSX"),
    "STOXX600":   dict(name="STOXX Europe 600", region="EU", ccy="EUR", etf="EXSA", cap=None, pricing="EU STOXX 600"),
    "FTSE100":    dict(name="FTSE 100", region="EU", ccy="GBP", etf="ISF", cap=None, pricing="EU STOXX 600"),
    "FTSE250":    dict(name="FTSE 250", region="EU", ccy="GBP", etf="MIDD", cap=None, pricing="EU STOXX 600",
                       note="MIDD uses optimised sampling: holdings diffs are noisy"),
    "DAX":        dict(name="DAX", region="EU", ccy="EUR", etf="EXS1", cap=0.15, pricing="EU DAX family"),
    "MDAX":       dict(name="MDAX", region="EU", ccy="EUR", etf="EXS3", cap=0.15, pricing="EU DAX family"),
    "TECDAX":     dict(name="TecDAX", region="EU", ccy="EUR", etf="EXS2", cap=0.15, pricing="EU DAX family"),
    "SDAX":       dict(name="SDAX", region="EU", ccy="EUR", etf=None, cap=0.15, pricing="EU DAX family"),
    "MSCI_EUROPE": dict(name="MSCI Europe (+ World/ACWI/EAFE trackers if pi supplied)", region="EU", ccy="EUR",
                        etf=None, cap=None, pricing="EU MSCI", pricing_proxy="EU STOXX 600"),
    "MSCI_CANADA": dict(name="MSCI Canada (+ World/ACWI trackers if pi supplied)", region="CA", ccy="CAD",
                        etf=None, cap=None, pricing="CA MSCI", pricing_proxy="CA S&P/TSX"),
    "FTSE_AW":    dict(name="FTSE All-World (GEIS)", region="GLOBAL", ccy="USD", etf=None, cap=None, pricing=None),
    # US indices: optional input (interlisted Canadian names on a shared close)
    "NDX":        dict(name="Nasdaq-100", region="US", ccy="USD", etf=None, cap=None, pricing="US Russell"),
    "SP500":      dict(name="S&P 500", region="US", ccy="USD", etf=None, cap=None, pricing="US Russell"),
    "R2000":      dict(name="Russell 2000", region="US", ccy="USD", etf="IWM", cap=None, pricing="US Russell"),
    "R1000":      dict(name="Russell 1000", region="US", ccy="USD", etf="IWB", cap=None, pricing="US Russell"),
}
US_INDICES = {k for k, v in INDEX.items() if v["region"] == "US"}

# iShares holdings endpoints (verified 2026-10-11). JSON carries ISIN and accepts asOfDate=YYYYMMDD.
# Canadian JSON has no share counts, so the CSV is fetched as well (same asOfDate parameter).
ETF = {
    "XIC":  dict(site="ca", pid=239837, index="SPTSX_COMP", ccy="CAD"),
    "XIU":  dict(site="ca", pid=239832, index="SPTSX_60", ccy="CAD"),
    "EXSA": dict(site="ch", pid=251931, index="STOXX600", ccy="EUR"),
    "ISF":  dict(site="ch", pid=251795, index="FTSE100", ccy="GBP"),
    "MIDD": dict(site="ch", pid=251796, index="FTSE250", ccy="GBP"),
    "EXS1": dict(site="ch", pid=251464, index="DAX", ccy="EUR"),
    "EXS3": dict(site="ch", pid=251845, index="MDAX", ccy="EUR"),
    "EXS2": dict(site="ch", pid=251975, index="TECDAX", ccy="EUR"),
}
ETF_URL = {
    "ca_json": "https://www.blackrock.com/ca/investors/en/products/{pid}/x/1464253357814.ajax?tab=all&fileType=json",
    "ca_csv": ("https://www.blackrock.com/ca/investors/en/products/{pid}/x/1464253357814.ajax?fileType=csv"
               "&fileName={etf}_holdings&dataType=fund"),
    "ch_json": "https://www.ishares.com/ch/individual/en/products/{pid}/x/1495092304805.ajax?tab=all&fileType=json",
}

# ------------------------------------------------------------------------------------ market constants
# Listing country -> (currency, normal-day closing-auction share of on-exchange value, label, source).
# Europe: Cboe on-exchange notional, May 2026 excl. 29 May (volume_forecasting.md §3.4, tag I).
# Canada: per-name share from the TSX MOC archive when available; this is the fallback (median per-name share of
# MOC/(MOC+continuous) on 8 Oct 2026, measured from ~/index_archive in marketdata.py; 0.10 if the archive is absent).
# US: about 9% of notional on a normal day (2024, V 3-0 medium).
COUNTRY = {
    "CA": ("CAD", 0.10, "TSX", "TSX MOC archive / fallback"),
    "GB": ("GBP", 0.246, "London", "Cboe May 2026"),
    "DE": ("EUR", 0.200, "Xetra", "Cboe May 2026"),
    "FR": ("EUR", 0.285, "Paris", "Cboe May 2026"),
    "NL": ("EUR", 0.215, "Amsterdam", "Cboe May 2026"),
    "BE": ("EUR", 0.25, "Brussels", "assumed = Euronext 25%"),
    "PT": ("EUR", 0.25, "Lisbon", "assumed = Euronext 25%"),
    "IE": ("EUR", 0.25, "Dublin", "assumed = Euronext 25%"),
    "IT": ("EUR", 0.162, "Milan", "Cboe May 2026"),
    "ES": ("EUR", 0.259, "Madrid", "Cboe May 2026"),
    "CH": ("CHF", 0.307, "Zurich", "Cboe May 2026"),
    "SE": ("SEK", 0.227, "Stockholm", "Cboe May 2026"),
    "DK": ("DKK", 0.227, "Copenhagen", "assumed = Stockholm"),
    "FI": ("EUR", 0.227, "Helsinki", "assumed = Stockholm"),
    "NO": ("NOK", 0.227, "Oslo", "assumed = Stockholm"),
    "AT": ("EUR", 0.20, "Vienna", "assumed = Xetra"),
    "PL": ("PLN", 0.20, "Warsaw", "assumed"),
    "LU": ("EUR", 0.25, "Luxembourg/Euronext", "assumed = Euronext 25%"),
    "GR": ("EUR", 0.20, "Athens", "assumed"),
    "US": ("USD", 0.09, "NYSE/Nasdaq", "volume_forecasting.md §3.4"),
}
ISO3_TO_2 = {"GBR": "GB", "DEU": "DE", "FRA": "FR", "NLD": "NL", "BEL": "BE", "PRT": "PT", "IRL": "IE", "ITA": "IT",
             "ESP": "ES", "CHE": "CH", "SWE": "SE", "DNK": "DK", "FIN": "FI", "NOR": "NO", "AUT": "AT", "POL": "PL",
             "LUX": "LU", "CAN": "CA", "USA": "US", "GRC": "GR"}
# iShares "Location" text -> ISO2 (the listing country used for volume and pricing is the exchange's country)
EXCH_COUNTRY = {"Toronto Stock Exchange": "CA", "London Stock Exchange": "GB", "Xetra": "DE",
                "Deutsche Boerse Xetra": "DE", "Euronext Paris": "FR", "Nyse Euronext - Euronext Paris": "FR",
                "Euronext Amsterdam": "NL", "Euronext Brussels": "BE", "Euronext Lisbon": "PT", "Euronext Dublin": "IE",
                "Irish Stock Exchange - All Market": "IE", "Borsa Italiana": "IT", "Bolsa De Madrid": "ES",
                "Six Swiss Exchange": "CH", "SIX Swiss Exchange": "CH", "Nasdaq Omx Nordic": "SE",
                "Omx Nordic Exchange Copenhagen A/S": "DK", "Nasdaq Omx Helsinki Ltd.": "FI", "Oslo Bors Asa": "NO",
                "Wiener Boerse Ag": "AT", "Warsaw Stock Exchange/Equities/Main Market": "PL",
                "Nasdaq Copenhagen": "DK", "Nasdaq Stockholm": "SE", "Nasdaq Helsinki": "FI", "Oslo Bors": "NO",
                "Wiener Boerse": "AT", "Euronext Oslo": "NO", "Nasdaq Omx Helsinki": "FI",
                "Athens Exchange S.A. Cash Market": "GR", "Boerse Berlin": "DE",
                "Nyse Euronext - Euronext Brussels": "BE", "Nyse Euronext - Euronext Lisbon": "PT"}

# Event-day closing-auction share of the day (phi in volume_forecasting.md §5.2): used to translate demand into
# an expected closing print. CA 0.71 is inferred for adds (canada.md §9.4, I); EU 0.55-0.60; US 0.5-0.7.
PHI = {"CA": 0.71, "EU": 0.58, "US": 0.60}

# ------------------------------------------------------------------------------- pricing layer v2 (2026-10-15)
# Which rows price a net imbalance (engine.pricing_row; README §4, §7b):
#   MSCI_CANADA / MSCI_EUROPE -> msci_close families 'CA MSCI' / 'EU MSCI', period MSCI_PERIOD; the event-type sub-row
#     (MSCI_EVENT_GROUP) when it has n >= MSCI_SUBROW_MIN_N, else 'net buy' / 'net sell' (flagged). UK split rows
#     ('EU MSCI UK') are shown in pricing_note only (msci_close RESULTS §6: n 16 / 36, no UK effect, t -0.4).
#   other Canadian rows -> 'CA S&P/TSX' TSX_PERIOD when that row has n >= TSX_MIN_N, else 2016+.
#   other families -> close_dislocation 2016+ rows, as in v1.
# Quotes: s09's lambda-only quote = d0 + PRICING_LAMBDA x SD(hold); quote_adj = quote + max(0, -E[reversal to exit])
# (msci_close RESULTS §6), applied to every row and hold.
PRICING_LAMBDA = 0.25
MSCI_PERIOD = "Oct2021+"
MSCI_SUBROW_MIN_N = 10
MSCI_EVENT_GROUP = {"migration_in": "up-migration", "migration_out": "down-migration", "add": "direct add",
                    "delete": "direct delete"}
TSX_PERIOD = "Oct2021+"
TSX_MIN_N = 30

# Index-futures hedge per listing country (expiry_basis RESULTS §1). GB names that are FTSE 250 members after the
# close (or not FTSE 100 members) use Y2; every other European country uses the STOXX 600 future.
HEDGE_FUTURE = {"CA": "SXF", "DE": "FDAX", "FR": "FCE", "NL": "FTI", "GB": "Z", "CH": "FSMI", "ES": "FIE",
                "IT": "FIB"}
HEDGE_FUTURE_EU_DEFAULT = "FXXP"
HEDGE_FUTURE_UK_MID = "Y2"
FUT_INDEX = {"SXF": "S&P/TSX 60", "FDAX": "DAX", "FCE": "CAC 40", "FTI": "AEX", "Z": "FTSE 100", "Y2": "FTSE 250",
             "FSMI": "SMI", "FIE": "IBEX 35", "FIB": "FTSE MIB", "FXXP": "STOXX 600"}
# Futures that settle on or near the cash close: the gap sd is measured from expiry_basis/data/auction_move_daily.csv
# since the settlement regime started (SXF 15:59-16:00 VWAP since 24 Sep 2018; FTSE MIB on the close since about
# 29 Apr 2024; CAC 40 / AEX on the close in all of 2015-2026), same stale filter (|a| < 150 bp) as the study.
HEDGE_REGIME_START = {"SXF": "2018-09-24", "FCE": "2015-01-01", "FTI": "2015-01-01", "FIB": "2024-04-29"}
# MSCI review closes for the hedge-gap day type (expiry_basis/common.py MSCI list + the Nov 2026 review)
MSCI_REVIEW_CLOSES = {
    "2018-02-28", "2018-05-31", "2018-08-31", "2018-11-30", "2019-02-28", "2019-05-28", "2019-08-27", "2019-11-26",
    "2020-02-28", "2020-05-29", "2020-08-31", "2020-11-30", "2021-02-26", "2021-05-27", "2021-08-31", "2021-11-30",
    "2022-02-28", "2022-05-31", "2022-08-31", "2022-11-30", "2023-02-28", "2023-05-31", "2023-08-31", "2023-11-30",
    "2024-02-29", "2024-05-31", "2024-08-30", "2024-11-25", "2025-02-28", "2025-05-30", "2025-08-26", "2025-11-24",
    "2026-02-27", "2026-05-29", "2026-11-30"}

# Depth stress (not in the quote): SIX / University of St. Gallen 2020, removing 10% of one side of a Swiss blue-chip
# closing auction moves the close 8-12 bp -> about 1 bp per 1% of the expected auction that is one-sided
# (03_microstructure/closing_auction_literature.md). depth_stress_bp = DEPTH_BP_PER_PCT x 100 x one-sided share.
DEPTH_BP_PER_PCT = 1.0
DEPTH_BP_PER_PCT_RANGE = (0.8, 1.2)


def region_of_country(c: str) -> str:
    return "CA" if c == "CA" else "US" if c == "US" else "EU"


# --------------------------------------------------------------------------------------------- calendar
_CAL: dict = {}


def sessions(exch: str = "XTSE") -> pd.DatetimeIndex:
    if exch not in _CAL:
        import exchange_calendars as xc
        c = xc.get_calendar(exch, start="2015-01-01", end="2028-12-31")
        s = c.sessions
        _CAL[exch] = pd.DatetimeIndex(s.tz_localize(None) if s.tz is not None else s)
    return _CAL[exch]


def prev_session(d, exch="XTSE") -> pd.Timestamp:
    cal = sessions(exch)
    return cal[cal.searchsorted(pd.Timestamp(d)) - 1]


def fmt_bn(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:,.2f}bn"
    if a >= 1e6:
        return f"{x / 1e6:,.1f}m"
    if a >= 1e3:
        return f"{x / 1e3:,.0f}k"
    return f"{x:,.0f}"
