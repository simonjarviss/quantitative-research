"""Helpers shared by the analysis steps (local data only; no R2 access)."""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from common import CACHE, CAL

_con = None


def con():
    global _con
    if _con is None:
        _con = duckdb.connect()
        _con.execute(f"CREATE VIEW dsf AS SELECT * FROM read_parquet('{CACHE}/dsf/*.parquet')")
        _con.execute(f"CREATE VIEW msf AS SELECT * FROM read_parquet('{CACHE}/msf.parquet')")
    return _con


def daily(start, end, cols="permno, permco, dlycaldt, ret, prc, cap, vol, shrout, dlydelflg", where=""):
    q = f"SELECT {cols} FROM dsf WHERE dlycaldt BETWEEN DATE '{pd.Timestamp(start).date()}' AND DATE '{pd.Timestamp(end).date()}' {where}"
    return con().execute(q).df()


_tdays = None


def tdays() -> pd.DatetimeIndex:
    """Trading days = dates with at least 1,000 CRSP equity rows (delisting rows sit on trading days too)."""
    global _tdays
    if _tdays is None:
        d = con().execute("SELECT dlycaldt, count(*) n FROM dsf GROUP BY 1 HAVING count(*) > 1000 ORDER BY 1").df()
        _tdays = pd.DatetimeIndex(pd.to_datetime(d.dlycaldt))
    return _tdays


def tshift(d, k: int) -> pd.Timestamp:
    """Trading day k days after (k<0: before) the trading day on or before d."""
    td = tdays()
    i = td.searchsorted(pd.Timestamp(d), side="right") - 1
    return td[i + k]


def last_tday_on_or_before(d) -> pd.Timestamp:
    return tshift(d, 0)


def horizon_dates(year: int) -> dict:
    T = CAL.loc[year, "T"]
    return {"H6": last_tday_on_or_before(pd.Timestamp(year - 1, 12, 31)),
            "H3": last_tday_on_or_before(pd.Timestamp(year, 3, 31)),
            "HR": last_tday_on_or_before(CAL.loc[year, "rank_day"]),
            "P": last_tday_on_or_before(CAL.loc[year, "prelim"]),
            "Tm1": tshift(T, -1), "T": T, "Tp1": tshift(T, 1), "Tp20": tshift(T, 20), "Tp60": tshift(T, 60)}


def cluster_t(x: pd.Series, g: pd.Series):
    """Mean, cluster-robust (by g) SE and t for the mean of x (CR1, intercept-only OLS)."""
    x = pd.Series(np.asarray(x, float)); g = pd.Series(np.asarray(g))
    ok = x.notna(); x, g = x[ok], g[ok]
    n, G = len(x), g.nunique()
    if n < 2 or G < 2:
        return x.mean() if n else np.nan, np.nan, np.nan
    m = x.mean()
    s = (x - m).groupby(g.values).sum()
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return m, se, m / se if se > 0 else np.nan
