"""Shared paths, calendars and statistics for the net-flow study (12_studies/net_flow/).

Run every script from <repo>/code with
PYTHONPATH=<repo>/code so `import r2` resolves (run_all.sh sets this).

Reads the sister studies read-only:
  ../close_dislocation/ (events_legs.csv, legs.py, common.py statistics, data/etf_aum.parquet, data/index_mv.parquet)
  ../tsx_composite/     (reviews.csv, events.csv, data/panel.npz, data/secd.parquet, data/ds_consvol.parquet)
  ../russell_june/      (output/events_actual.csv, _cache/dsf)
  ../stoxx600_dax/      (events_*.csv, reviews_*.csv, data/panel.npz)
Writes only to this folder: data/ (downloads and intermediates, regenerable) and the deliverables.
"""
from __future__ import annotations
import importlib.util
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE
DATA.mkdir(exist_ok=True)
STUDIES = HERE.parent
CD = STUDIES / "close_dislocation"
RUS = STUDIES / "russell_june"
TSX = STUDIES / "tsx_composite"
EU = STUDIES / "stoxx600_dax"
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_cd = _load("cd_common", CD / "common.py")          # statistics of the sister study (no side effects but mkdir data/)
clt, summ, ols_cluster, trimmed = _cd.clt, _cd.summ, _cd.ols_cluster, _cd.trimmed
legs_mod = _load("cd_legs", CD / "legs.py")


def period(y) -> str:
    return "pre-2016" if int(y) < 2016 else "2016+"


# ------------------------------------------------------------------------------------------------ calendars
_CAL = {}


def sessions(exch: str) -> pd.DatetimeIndex:
    if exch not in _CAL:
        import exchange_calendars as xc
        c = xc.get_calendar(exch, start="2005-01-03", end="2027-06-30")
        s = c.sessions
        _CAL[exch] = pd.DatetimeIndex(s.tz_localize(None) if s.tz is not None else s)
    return _CAL[exch]


def tsx_sessions() -> pd.DatetimeIndex:
    return sessions("XTSE")


def nth_weekday(y, m, weekday, n):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def last_on_or_before(cal, d):
    return cal[cal.searchsorted(pd.Timestamp(d), side="right") - 1]


def tsx_reviews() -> pd.DataFrame:
    """Quarterly S&P/TSX reviews 2006-03 .. 2026-09 with T (3rd-Friday close) and eff (next session).
    T matches tsx_composite/reviews.csv for 2006-03 .. 2026-06 (asserted)."""
    cal = tsx_sessions()
    rows = []
    for y in range(2006, 2027):
        for m in (3, 6, 9, 12):
            if (y, m) > (2026, 9):
                continue
            T = last_on_or_before(cal, nth_weekday(y, m, 4, 3))
            eff = cal[cal.get_loc(T) + 1]
            rows.append(dict(review=f"{y}-{m:02d}", T=T, eff=eff))
    rv = pd.DataFrame(rows)
    sis = pd.read_csv(TSX / "reviews.csv", parse_dates=["T"])
    chk = rv.merge(sis[["review", "T"]], on="review", suffixes=("", "_s"))
    assert (chk["T"] == chk["T_s"]).all(), chk[chk["T"] != chk["T_s"]]
    return rv


# ------------------------------------------------------------------------------------------------ statistics
def winsor(x, p=0.01):
    x = pd.Series(np.asarray(x, float))
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lo, hi).values


def demean_by(x, g):
    """Within-group demeaning (absorbs fixed effects g) for a 1-d or 2-d array."""
    x = np.asarray(x, float)
    df = pd.DataFrame(x if x.ndim == 2 else x[:, None])
    out = df - df.groupby(np.asarray(g)).transform("mean")
    return out.values if x.ndim == 2 else out.values[:, 0]


def fe_ols(y, X, fe, cl, names):
    """OLS of y on X absorbing fixed effects `fe` (by demeaning), CR1 SEs clustered by `cl`.
    Rows with any NaN are dropped before demeaning. Returns (table, (n, clusters, within r2))."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    fe = np.asarray(fe); cl = np.asarray(cl)
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    y, X, fe, cl = y[ok], X[ok], fe[ok], cl[ok]
    if len(y) < X.shape[1] + 5:
        return None, (len(y), 0, np.nan)
    yd = demean_by(y, fe); Xd = demean_by(X, fe)
    return ols_cluster(yd, Xd, cl, names)
