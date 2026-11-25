"""Parse the cached iShares UCITS daily holdings (ishares.com/ch JSON) into a tidy frame (as 16_flow_engine/holdings.py)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common import ISHCH, weekdays_before, weekday_after

LOC_CTRY = {"Canada": "CA", "United Kingdom": "GB", "France": "FR", "Germany": "DE", "Switzerland": "CH",
            "Netherlands": "NL", "Sweden": "SE", "Spain": "ES", "Italy": "IT", "Denmark": "DK", "Belgium": "BE",
            "Finland": "FI", "Norway": "NO", "Ireland": "IE", "Austria": "AT", "Portugal": "PT", "Luxembourg": "LU",
            "United States": "US", "Japan": "JP", "Australia": "AU", "Hong Kong": "HK", "Singapore": "SG",
            "Israel": "IL", "New Zealand": "NZ", "Jersey": "GB", "Guernsey": "GB", "Isle Of Man": "GB",
            "Bermuda": "BM", "Cayman Islands": "KY", "Faroe Islands": "DK", "Macau": "MO", "Poland": "PL"}
JUNK = r"\bRTS\b|RIGHT|COUPON|SCRIP|NIL PAID|WARRANT|\bWTS?\b|ENTITLEMENT|BONUS|\bCVR\b|ESCROW|\bRT\b|\.RT\b"
EU_LOC = {"GB", "FR", "DE", "CH", "NL", "SE", "ES", "IT", "DK", "BE", "FI", "NO", "IE", "AT", "PT", "LU"}


def _raw(x):
    return x.get("raw") if isinstance(x, dict) else x


def _num(s):
    s = _raw(s)
    if s is None:
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace(",", "").replace("’", "").replace("'", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


@lru_cache(maxsize=4096)
def load(fund: str, d) -> pd.DataFrame | None:
    """Equity lines of one fund on one date (None if the file is missing or empty)."""
    p = ISHCH / f"{fund}_{pd.Timestamp(d):%Y%m%d}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8-sig")).get("aaData", [])
    recs = []
    for r in rows:
        if len(r) < 13:
            continue
        o = len(r) - 13
        recs.append(dict(ticker=str(r[0]).strip(), name=r[1], sector=r[2], asset_class=r[3 + o], mv=_num(r[4 + o]),
                         weight=_num(r[5 + o]) / 100.0, shares=_num(r[7 + o]), isin=r[8 + o], price=_num(r[9 + o]),
                         location=r[10 + o], exchange=r[11 + o], mkt_ccy=r[12 + o]))
    df = pd.DataFrame(recs)
    if df.empty:
        return None
    df = df[(df.asset_class == "Equity") & (df.shares > 0) & df["isin"].astype(str).str.len().eq(12)].copy()
    junk = df.name.astype(str).str.upper().str.contains(JUNK) | df.ticker.isin(["-", ""])
    df = df[~junk]
    df["ctry"] = df.location.map(LOC_CTRY).fillna(df["isin"].str[:2])
    df = df.groupby("isin", as_index=False).agg(ticker=("ticker", "first"), name=("name", "first"),
                                                 sector=("sector", "first"), shares=("shares", "sum"),
                                                 price=("price", "first"), mv=("mv", "sum"), weight=("weight", "sum"),
                                                 ctry=("ctry", "first"), exchange=("exchange", "first"),
                                                 mkt_ccy=("mkt_ccy", "first"))
    df["fund"], df["date"] = fund, pd.Timestamp(d)
    return df


def around(fund: str, T) -> dict:
    """Nearest non-empty files: 'pre' = last before T (T-1, else T-2, T-3), 'T', 'post' = T+1, 'pre2' = before pre."""
    T = pd.Timestamp(T)
    out = {}
    pre = [d for d in weekdays_before(T, 3) if load(fund, d) is not None]
    out["pre"] = pre[0] if pre else None
    out["pre2"] = pre[1] if len(pre) > 1 else None
    out["T"] = T if load(fund, T) is not None else None
    a = weekday_after(T)
    out["post"] = a if load(fund, a) is not None else None
    return out
