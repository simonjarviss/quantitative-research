"""Shared paths, event-date helpers and statistics for the close-dislocation study.

Run every script from <repo>/code with
PYTHONPATH=<repo>/code so `import r2` resolves.

The study reads the sister studies' event files, panels and caches read-only:
  ../russell_june/   (output/events_actual.csv, _cache/dsf, msf, cells_<y>.parquet, pods13f, shortint, ccm)
  ../tsx_composite/  (events.csv, reviews.csv, data/panel.npz, predictions_final.csv, car_events.csv, pods_13f.csv)
  ../stoxx600_dax/   (events_*.csv, reviews_*.csv, data/panel.npz, predictions_*.csv, car_events_*.csv,
                      shorts_footprint.csv)
and writes only to this folder (data/ = regenerable intermediates; *.csv = deliverables).
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE
DATA.mkdir(exist_ok=True)
STUDIES = HERE.parent
RUS = STUDIES / "russell_june"
TSX = STUDIES / "tsx_composite"
EU = STUDIES / "stoxx600_dax"
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

IWM_PERMNO = 88222
CLIP = (-0.5, 1.0)                      # daily return clip, as in the TSX/EU sister studies

# Legs (all signed so that + = in the direction of index demand). Day labels are sessions relative to T,
# the last session before the effective date; "o" = open of that session.
LEGS = [("pre5", "T-5 -> T-1 close"),
        ("d0", "T-1 close -> T close (trade day, incl. auction)"),
        ("d0_on", "T-1 close -> T open"),
        ("d0_id", "T open -> T close"),
        ("on1", "T close -> T+1 open (overnight)"),
        ("id1", "T+1 open -> T+1 close"),
        ("r1", "T close -> T+1 close"),
        ("r2_5", "T+1 close -> T+5 close"),
        ("r6_20", "T+5 close -> T+20 close"),
        ("r1_5", "T close -> T+5 close"),
        ("r1_20", "T close -> T+20 close")]
LEG_LABEL = dict(LEGS)

# Passive shares (pi). Time-varying where R2 allows (s02_pi.py); constants are the flow_sizing.md §4.4 values.
PI_CONST = {"R2000": 0.0287, "TSX": 0.0100, "STOXX": 0.0027}

# Tax regime of a listing country (STOXX study common.buy_tax, 03_microstructure/transaction_taxes.md)
FTT = {"FRA", "ITA", "ESP"}


def regime(cty: str) -> str:
    if cty == "GBR":
        return "UK"
    if cty in FTT:
        return "FTT (FR/IT/ES)"
    if cty in ("IRL", "BEL"):
        return "other taxed (IE/BE)"
    return "untaxed"


def period(y) -> str:
    return "pre-2016" if int(y) < 2016 else "2016+"


# ---------------------------------------------------------------- statistics
def clt(x, g):
    """Mean, cluster-robust SE (CR1, clusters g) and t of the mean of x."""
    x = np.asarray(x, float); g = np.asarray(g)
    ok = ~np.isnan(x); x, g = x[ok], g[ok]
    n = len(x); G = len(np.unique(g)) if n else 0
    if n < 3 or G < 2:
        return (x.mean() if n else np.nan), np.nan, np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum()
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return m, se, (m / se if se > 0 else np.nan)


def trimmed(x, p=0.025):
    """Mean after dropping the lowest and highest p share (default 2.5% each tail)."""
    x = np.sort(np.asarray(x, float)[~np.isnan(np.asarray(x, float))])
    if len(x) < 10:
        return np.nan
    k = int(np.floor(p * len(x)))
    return x[k:len(x) - k].mean() if k > 0 else x.mean()


def summ(x, g, scale=1e4):
    """dict of n, events (clusters), mean, t, median, trimmed mean, share > 0 (x in fractions; output in bp)."""
    x = np.asarray(x, float); g = np.asarray(g)
    ok = ~np.isnan(x)
    m, se, t = clt(x[ok], g[ok])
    return dict(n=int(ok.sum()), n_dates=int(len(np.unique(g[ok]))), mean_bp=m * scale, t=t,
                median_bp=np.median(x[ok]) * scale if ok.any() else np.nan,
                trim_bp=trimmed(x[ok]) * scale, pos_share=float((x[ok] > 0).mean()) if ok.any() else np.nan,
                sd_bp=np.std(x[ok], ddof=1) * scale if ok.sum() > 1 else np.nan)


def ols_cluster(y, X, g, names):
    """OLS with CR1 cluster-robust SEs. Returns DataFrame coef / se / t and (n, clusters, r2)."""
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(g)
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    y, X, g = y[ok], X[ok], g[ok]
    n, k = X.shape
    G = len(np.unique(g))
    if n <= k + 2 or G < 3:
        return None, (n, G, np.nan)
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    meat = np.zeros((k, k))
    for gg in np.unique(g):
        idx = g == gg
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX @ meat @ XtX
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"term": names, "coef": b, "se": se, "t": b / se}), (n, G, r2)
