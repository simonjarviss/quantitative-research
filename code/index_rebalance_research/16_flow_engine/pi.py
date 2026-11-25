"""Passive share pi per index, as of a date (never after it).

Priority:
  1. pi_overrides.csv (this folder; hand-entered: index,date,pi_visible,R,pi_true,source)
  2. ../12_studies/net_flow/pi_calibration.csv (parallel study: family, date, visible pi, R, true pi, source)
     MSCI rows (MSCI_CANADA, MSCI_CANADA_SC, MSCI_EUROPE, MSCI_EUROPE_SC, MSCI_EUROPE_<CC>) are appended by
     ../12_studies/msci_pi (source 'msci_pi study', 2026-10-12); ALIASES maps the first four.
  3. FALLBACK, flagged in every output row (fallback=True):
     - visible pi = tracker AUM / index float cap, the method of flow_sizing.md §4.4, taken on the last date <= the
       event date: daily series rebuilt from the close_dislocation caches (S&P/TSX Composite, STOXX 600, DAX, MDAX,
       SDAX, TecDAX: Datastream wrds_ds2dsf x ds2indexdata) and from data/r2 (FTSE 100, FTSE 250, our pull).
       For dates on/after 2026-06-23 the §4.4 table value is used where §4.4 has one.
     - indices with no series (S&P/TSX 60, MSCI Europe, S&P 500, Russell 2000) take the §4.4 table value (dated
       2026-06-23); before that date the S&P/TSX 60 value is scaled by the Composite series (tag I).
     - R = excess trade-day volume / visible demand, flow_sizing.md §4.4 and volume_forecasting.md §1:
       S&P 6.3 (consolidated), S&P/TSX 4.2 (consolidated), STOXX 600 5.1 (primary). Indices without their own R borrow
       the closest one (flagged R_assumed=True). pi_true = R * pi_visible.
Units: fractions (0.01 = 1%).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from common import PI_CALIBRATION, PI_OVERRIDES, R2DIR, STUDIES

TABLE_DATE = pd.Timestamp("2026-06-23")
TABLE_PI = {"SPTSX_COMP": 0.0100, "SPTSX_60": 0.0072, "STOXX600": 0.0027, "MSCI_EUROPE": 0.0032,
            "SP500": 0.0411, "R2000": 0.0287}
R_OWN = {"SP500": 6.3, "SPTSX_COMP": 4.2, "STOXX600": 5.1}
R_BORROW = {"SPTSX_60": "SPTSX_COMP", "MSCI_CANADA": "SPTSX_COMP", "FTSE100": "STOXX600", "FTSE250": "STOXX600",
            "DAX": "STOXX600", "MDAX": "STOXX600", "SDAX": "STOXX600", "TECDAX": "STOXX600",
            "MSCI_EUROPE": "STOXX600", "NDX": "SP500", "FTSE_AW": "STOXX600",
            "MSCI_CANADA_SC": "SPTSX_COMP", "MSCI_EUROPE_SC": "STOXX600"}
# Russell: no R in the pack (flow_sizing §4.4 fits S&P 500, S&P/TSX and STOXX 600 only) -> R missing, pi_true = pi_visible
SERIES_FAMILY = {"TSX": "SPTSX_COMP", "STOXX": "STOXX600", "DAX": "DAX", "MDAX": "MDAX", "SDAX": "SDAX",
                 "TecDAX": "TECDAX", "FTSE100": "FTSE100", "FTSE250": "FTSE250"}
ALIASES = {"TSX": "SPTSX_COMP", "SPTSX": "SPTSX_COMP", "SPTSXCOMPOSITE": "SPTSX_COMP", "SPTSXCOMP": "SPTSX_COMP",
           "COMPOSITE": "SPTSX_COMP", "TSX60": "SPTSX_60", "SPTSX60": "SPTSX_60", "STOXX": "STOXX600",
           "STOXX600": "STOXX600", "STOXXEUROPE600": "STOXX600", "DAX": "DAX", "MDAX": "MDAX", "SDAX": "SDAX",
           "TECDAX": "TECDAX", "FTSE100": "FTSE100", "FTSE250": "FTSE250", "UKX": "FTSE100", "MCX": "FTSE250",
           "SP500": "SP500", "R2000": "R2000", "RUSSELL2000": "R2000", "R1000": "R1000", "RUSSELL1000": "R1000",
           "MSCIEUROPE": "MSCI_EUROPE", "MSCICANADA": "MSCI_CANADA", "NDX": "NDX", "NASDAQ100": "NDX",
           # MSCI families written by 12_studies/msci_pi (source 'msci_pi study'): Standard holders resolve to the
           # engine's MSCI_CANADA / MSCI_EUROPE; the Small Cap holders' families resolve for pi_for() (migration
           # netting in the input rows) but have no common.INDEX entry. Country splits (MSCI_EUROPE_GB, ...) stay
           # unmapped (reference only).
           "MSCICANADASC": "MSCI_CANADA_SC", "MSCIEUROPESC": "MSCI_EUROPE_SC"}


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def _to_index(fam: str) -> str | None:
    n = _norm(fam)
    return ALIASES.get(n) or (n if n in R_OWN or n in R_BORROW else None)


@lru_cache(maxsize=1)
def _series() -> pd.DataFrame:
    """Daily visible pi by index from cached R2 pulls (AUM and MV in the same currency, millions vs units)."""
    out = []
    d = STUDIES / "close_dislocation" / "data"
    if (d / "etf_aum.parquet").exists():
        a = pd.read_parquet(d / "etf_aum.parquet")
        a["marketdate"] = pd.to_datetime(a.marketdate)
        eur_usd = a[a.currency == "EUR"].assign(r=lambda x: x.mktcap / x.mktcap_usd).groupby("marketdate").r.median()
        a["aum"] = np.where(a.currency == "CHF", a.mktcap_usd * a.marketdate.map(eur_usd), a.mktcap)
        aum = a.groupby(["family", "marketdate"]).aum.sum().reset_index()
        m = pd.read_parquet(d / "index_mv.parquet")
        m["marketdate"] = pd.to_datetime(m.valuedate)
        x = aum.merge(m[["family", "marketdate", "mv"]], on=["family", "marketdate"])
        x["pi"] = x.aum / (x.mv * 1e6)
        x["index"] = x.family.map(SERIES_FAMILY)
        x["src"] = "R2 Datastream tracker AUM / index MV (close_dislocation caches)"
        out.append(x[["index", "marketdate", "pi", "src", "aum", "mv"]])
    if (R2DIR / "ftse_etf_aum.parquet").exists():
        a = pd.read_parquet(R2DIR / "ftse_etf_aum.parquet")
        a["marketdate"] = pd.to_datetime(a.marketdate)
        gbp_usd = a[a.fund == "ISF"].assign(r=lambda x: x.mktcap / x.mktcap_usd).set_index("marketdate").r
        # hedged USD/EUR share classes: convert through their own USD value
        a["aum"] = np.where(a.fund.str.contains("hdg"), a.mktcap_usd * a.marketdate.map(gbp_usd), a.mktcap)
        aum = a.groupby(["family", "marketdate"]).aum.sum().reset_index()
        m = pd.read_parquet(R2DIR / "ftse_index_mv.parquet")
        m["marketdate"] = pd.to_datetime(m.valuedate)
        x = aum.merge(m[["family", "marketdate", "mv"]], on=["family", "marketdate"])
        x["pi"] = x.aum / (x.mv * 1e6)
        x["index"] = x.family
        x["src"] = "R2 Datastream FTSE tracker AUM / index MV (data/r2, this engine)"
        out.append(x[["index", "marketdate", "pi", "src", "aum", "mv"]])
    if not out:
        return pd.DataFrame(columns=["index", "marketdate", "pi", "src", "aum", "mv"])
    s = pd.concat(out, ignore_index=True).dropna(subset=["index", "pi"])
    return s[(s.pi > 0) & (s.pi < 0.5)].sort_values(["index", "marketdate"])


def series_value(index: str, date) -> tuple[float, pd.Timestamp | None, str]:
    s = _series()
    x = s[(s["index"] == index) & (s.marketdate <= pd.Timestamp(date))]
    if x.empty:
        return np.nan, None, ""
    # median of the last 5 sessions on/before the date (one-day AUM prints are noisy)
    t = x.tail(5)
    return float(t.pi.median()), t.marketdate.iloc[-1], t.src.iloc[-1]


def index_mv(index: str, date) -> tuple[float, pd.Timestamp | None]:
    """Index float cap (units of index currency) on the last date <= date, from the same caches."""
    s = _series()
    x = s[(s["index"] == index) & (s.marketdate <= pd.Timestamp(date))]
    if x.empty:
        return np.nan, None
    return float(x.mv.iloc[-1] * 1e6), x.marketdate.iloc[-1]


def _read_table(path) -> pd.DataFrame | None:
    """Read pi_overrides.csv / pi_calibration.csv. Expected columns (case-insensitive): family (or index), date,
    pi_visible, R, pi_true, source; optional R_se (standard error of R), R_n, R_window, R_basis."""
    if not path.exists():
        return None
    t = pd.read_csv(path)
    low = {c: c.strip().lower() for c in t.columns}
    exact = {"family": "family", "index": "family", "date": "date", "asof": "date", "pi_visible": "pi_visible",
             "visible_pi": "pi_visible", "r": "R", "pi_true": "pi_true", "true_pi": "pi_true", "source": "source",
             "r_se": "R_se", "r_n": "R_n", "r_window": "R_window", "r_basis": "R_basis"}
    t = t.rename(columns={c: exact[l] for c, l in low.items() if l in exact})
    t = t.loc[:, ~t.columns.duplicated()]
    if "family" not in t or "date" not in t:
        return None
    t["index"] = t.family.map(_to_index)
    t["date"] = pd.to_datetime(t.date, errors="coerce")
    for c in ("pi_visible", "pi_true", "R", "R_se"):
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    for c in ("pi_visible", "pi_true"):
        if c in t:
            t.loc[t[c] > 0.5, c] = t.loc[t[c] > 0.5, c] / 100.0    # given in percent
    return t.dropna(subset=["index", "date"])


def pi_for(index: str, date) -> dict:
    """pi_visible, R, pi_true, source, fallback flags for one index as of `date` (never later)."""
    date = pd.Timestamp(date)
    for path, label in ((PI_OVERRIDES, "override"), (PI_CALIBRATION, "pi_calibration.csv")):
        t = _read_table(path)
        if t is None:
            continue
        x = t[(t["index"] == index) & (t.date <= date)].sort_values("date")
        if x.empty:
            continue
        r = x.iloc[-1]
        pv = float(r.get("pi_visible", np.nan))
        R = float(r.get("R", np.nan))
        pt = float(r.get("pi_true", np.nan))
        R_se = float(r.get("R_se", np.nan))
        if not np.isfinite(pv) and np.isfinite(pt) and np.isfinite(R) and R > 0:
            pv = pt / R
        if not np.isfinite(pv):
            continue
        note, R_assumed = "", False
        # Guard: an R that is not significant (t < 2) or missing is replaced by the pack's R (flagged)
        if not np.isfinite(R) or (np.isfinite(R_se) and R_se > 0 and R / R_se < 2):
            fb = R_OWN.get(index, R_OWN.get(R_BORROW.get(index, ""), np.nan))
            note = (f"; R {R:.2f} (t {R / R_se:.1f}) not significant -> pack R {fb}" if np.isfinite(R)
                    else f"; no R -> pack R {fb}")
            R, R_assumed = fb, True
            pt = pv * R if np.isfinite(R) else pv
        elif not np.isfinite(pt):
            pt = pv * R
        return dict(index=index, pi_visible=pv, R=R, pi_true=pt, pi_date=r.date,
                    pi_source=f"{label} ({r.date.date()}): {str(r.get('source', ''))[:110]}{note}",
                    fallback=False, R_assumed=R_assumed)
    # ---- fallback
    pv, pdate, src = np.nan, None, ""
    if index in TABLE_PI and date >= TABLE_DATE:
        pv, pdate, src = TABLE_PI[index], TABLE_DATE, "flow_sizing.md §4.4 table (2026-06-23)"
    else:
        v, d, s = series_value(index, date)
        if np.isfinite(v):
            pv, pdate, src = v, d, s
        elif index == "SPTSX_60":
            v, d, s = series_value("SPTSX_COMP", date)
            if np.isfinite(v):
                pv, pdate = TABLE_PI["SPTSX_60"] * v / TABLE_PI["SPTSX_COMP"], d
                src = "flow_sizing.md §4.4 TSX 60 value scaled by the Composite R2 series (I)"
        elif index in TABLE_PI:
            pv, pdate, src = TABLE_PI[index], TABLE_DATE, "flow_sizing.md §4.4 table (2026-06-23, AFTER event date)"
    R = R_OWN.get(index, np.nan)
    R_assumed = False
    if not np.isfinite(R) and index in R_BORROW:
        R, R_assumed = R_OWN[R_BORROW[index]], True
    pt = pv * R if np.isfinite(pv) and np.isfinite(R) else pv
    return dict(index=index, pi_visible=pv, R=R, pi_true=pt, pi_date=pdate, pi_source=src or "MISSING",
                fallback=True, R_assumed=R_assumed)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "2026-06-19"
    rows = [pi_for(i, d) for i in ["SPTSX_COMP", "SPTSX_60", "STOXX600", "FTSE100", "FTSE250", "DAX", "MDAX",
                                   "TECDAX", "SDAX", "MSCI_EUROPE", "MSCI_CANADA", "SP500", "R2000", "NDX"]]
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
