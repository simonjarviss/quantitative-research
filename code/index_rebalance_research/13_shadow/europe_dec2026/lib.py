"""Shared helpers: prices in index currency, relative volatility, the rank -> free-float-cap curve, probability
model, file archiving and hashing (paper only)."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C


# ------------------------------------------------------------------------------------------ files
def archive_old(p: Path) -> Path:
    """Never overwrite a logged prediction: move an existing file to <dir>/superseded/ first."""
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        d = p.parent / "superseded"
        d.mkdir(exist_ok=True)
        ts = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
        p.rename(d / f"{p.stem}__{ts}{p.suffix}")
    return p


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def append_log(lines: list[str]):
    new = not C.LOG.exists() or C.LOG.stat().st_size == 0
    with open(C.LOG, "a") as fh:
        if new:
            fh.write("# Shadow log — December 2026 European reviews: STOXX Europe 600, DAX family, FTSE 100/250 "
                     "(paper only)\n\nAppend-only. Each predict.py / score.py / validate run adds a block; never edit "
                     "earlier blocks.\n")
        fh.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------ prices
def price_panel(P: pd.DataFrame, field: str = "close") -> tuple[pd.DataFrame, pd.Series]:
    """Wide panel date x symbol of `field`, pence converted to pounds (Yahoo 'GBp'/'GBX'); returns (panel, ccy)."""
    x = P.copy()
    ccy = x.groupby("symbol").currency.first().fillna("")
    pence = ccy.isin(["GBp", "GBX", "ZAc", "ILA"])
    W = x.pivot_table(index="date", columns="symbol", values=field, aggfunc="last").sort_index()
    for s in ccy[pence].index:
        if s in W:
            W[s] = W[s] / 100.0
    ccy = ccy.replace({"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"})
    return W, ccy


def fx_to(P: pd.DataFrame, target: str = "EUR") -> pd.DataFrame:
    """Daily FX: units of `target` per unit of currency (columns = currency codes)."""
    W, _ = price_panel(P)
    out = {}
    eur = {k: W[v] for k, v in C.FX_TICKERS.items() if v in W}
    eur["EUR"] = pd.Series(1.0, index=W.index)
    if target == "EUR":
        out = eur
    else:   # e.g. GBP: x_GBP = x_EUR / (GBP in EUR)
        g = eur[target]
        out = {k: v / g for k, v in eur.items()}
    F = pd.DataFrame(out).sort_index().ffill()
    return F


def asof_value(W: pd.DataFrame, d: pd.Timestamp, max_stale: int = 5) -> pd.Series:
    """Last value on or before d per column (NaN if older than max_stale sessions)."""
    x = W[W.index <= d]
    if x.empty:
        return pd.Series(np.nan, index=W.columns)
    last = x.ffill().iloc[-1]
    age = x.notna()[::-1].cumsum().eq(0).sum()     # sessions since last value
    return last.where(age <= max_stale)


def stale(W: pd.DataFrame, d: pd.Timestamp, n: int = 6, P: pd.DataFrame | None = None) -> pd.Series:
    """True where the last n closes up to d contain 3+ consecutive identical values (Yahoo repeats a stale price).
    With the long price frame P, a line that traded (volume > 0) in the last bar on or before d is not stale: Yahoo
    gap-fills missing Warsaw sessions (e.g. 5-9 Oct 2026) with the previous close and zero volume, then resumes."""
    x = W[W.index <= d].tail(n)
    same = x.diff().eq(0)
    run = (same & same.shift(1, fill_value=False)).any()
    if P is not None and len(P):
        v = P[P.date <= d].sort_values("date").groupby("symbol").volume.last()
        traded = v.reindex(run.index).fillna(0) > 0
        run = run & ~traded
    return run


def vwap20(P: pd.DataFrame, d: pd.Timestamp, n: int = C.VWAP_DAYS) -> pd.Series:
    """20-session mean of the daily typical price (H+L+C)/3 ending at d: proxy for the 20-day Xetra VWAP."""
    Wc, _ = price_panel(P, "close")
    Wh, _ = price_panel(P, "high")
    Wl, _ = price_panel(P, "low")
    tp = (Wc + Wh.reindex_like(Wc) + Wl.reindex_like(Wc)) / 3.0
    tp = tp.where(tp.notna(), Wc)
    x = tp[tp.index <= d].tail(n)
    return x.mean().where(x.notna().sum() >= n // 2)


def rel_sigma(W_eur: pd.DataFrame, d: pd.Timestamp, cols: list[str], window: int = C.VOL_WINDOW) -> pd.Series:
    """Daily volatility of each column's log return minus the equal-weighted mean log return of `cols`."""
    x = np.log(W_eur[W_eur.index <= d].tail(window + 1)).diff().iloc[1:]
    x = x.clip(-0.5, 0.5)
    mk = x[[c for c in cols if c in x]].mean(axis=1)
    rel = x.sub(mk, axis=0)
    s = rel.std().where(rel.notna().sum() >= 20)
    med = s.median()
    return s.fillna(med).clip(0.005, 0.10)


def h_eff_vwap(h: int, n: int = C.VWAP_DAYS) -> float:
    """Variance multiplier (in days) of the log of an n-day average ending h days ahead (random walk)."""
    if h <= 0:
        return 0.0
    if h >= n:
        return (h - n) + (n + 1) * (2 * n + 1) / (6.0 * n)              # a + Var(mean of n random-walk steps)
    # inside the window: only the last h of n days are unknown
    k = np.arange(1, h + 1)
    return float(((h - k + 1) ** 2).sum()) / n ** 2


# ------------------------------------------------------------------------------------------ rank curve
def rank_curve(rank: np.ndarray, logv: np.ndarray, w: np.ndarray | None = None):
    """Monotone (decreasing in rank) curve log FFMC = g(rank) through anchor points (isotonic fit), linear between
    block centres, linear extrapolation at the ends. Returns g (callable)."""
    from sklearn.isotonic import IsotonicRegression
    ok = np.isfinite(rank) & np.isfinite(logv)
    r, v = rank[ok], logv[ok]
    w = np.ones_like(r) if w is None else np.asarray(w)[ok]
    ir = IsotonicRegression(increasing=False)
    fit = ir.fit_transform(r, v, sample_weight=w)
    o = np.argsort(r)
    r, fit, w = r[o], fit[o], w[o]
    blk = np.r_[0, np.cumsum(np.abs(np.diff(fit)) > 1e-12)]
    xs = np.array([np.average(r[blk == b], weights=w[blk == b]) for b in np.unique(blk)])
    ys = np.array([fit[blk == b][0] for b in np.unique(blk)])
    # enforce strictly decreasing
    for i in range(1, len(ys)):
        if ys[i] >= ys[i - 1]:
            ys[i] = ys[i - 1] - 1e-6
    lo_slope = (ys[min(5, len(ys) - 1)] - ys[0]) / max(xs[min(5, len(xs) - 1)] - xs[0], 1)
    k = min(10, len(ys) - 1)
    hi_slope = (ys[-1] - ys[-1 - k]) / max(xs[-1] - xs[-1 - k], 1)

    def g(q):
        q = np.asarray(q, float)
        out = np.interp(q, xs, ys)
        out = np.where(q < xs[0], ys[0] + lo_slope * (q - xs[0]), out)
        out = np.where(q > xs[-1], ys[-1] + hi_slope * (q - xs[-1]), out)
        return out
    return g


# ------------------------------------------------------------------------------------------ probability
def p_cross(margin_log, sigma_tot):
    """P(the name ends on the same side of the line as now) for a signed log margin (+ = safely on its side)."""
    m = np.asarray(margin_log, float)
    s = np.asarray(sigma_tot, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(s > 0, m / s, np.sign(m) * np.inf)
    return norm.cdf(z)


def sigma_total(sig_rel, h_days, sig_data):
    return np.sqrt(np.asarray(sig_rel, float) ** 2 * h_days + np.asarray(sig_data, float) ** 2)


def fmt_pct(x):
    return "" if not np.isfinite(x) else f"{x:.0%}"
