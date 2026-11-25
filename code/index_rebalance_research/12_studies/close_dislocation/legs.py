"""Leg arithmetic shared by the market builders (s03_us.py, s04_ca.py, s05_eu.py).

For one event we hold, for sessions k = -5..+20 around T (k = 0 is T):
  r[k]  stock daily close-to-close total return (clipped to [-50%, +100%]; NaN when not traded)
  c[k]  equal-weighted mean of the matched control cell on day k
  x[k]  index (benchmark) return on day k
and the open-based pieces on T and T+1 (stock, control-cell mean, index):
  on0 = T-1 close -> T open,   id0 = T open -> T close,   on1 = T close -> T+1 open,   id1 = T+1 open -> T+1 close
where id = close/open - 1 on the same session (no distribution can fall inside a session) and
on = (1 + total return of the session) / (1 + id) - 1, so the overnight piece carries any dividend or split.
Abnormal multi-day legs are sums of daily abnormal returns (the TSX/STOXX sister-study CAR convention); a missing
stock return contributes 0 (as there). Sub-day legs are NaN when the stock's open is missing or flagged.
"""
import numpy as np

K0 = 5            # index of k = 0 in arrays that run k = -5..+20
KS = np.arange(-5, 21)

SUMS = {"pre5": range(-4, 0), "d0": [0], "r1": [1], "r2_5": range(2, 6), "r6_20": range(6, 21),
        "r1_5": range(1, 6), "r1_20": range(1, 21)}


def daily_legs(r, c, x):
    """r, c, x: arrays of length 26 (k=-5..20). Returns dict leg -> (raw, abnormal, index-adjusted)."""
    out = {}
    for leg, ks in SUMS.items():
        idx = [k + K0 for k in ks]
        rr = r[idx]
        if np.all(np.isnan(rr)):
            out[leg] = (np.nan, np.nan, np.nan)
            continue
        raw = np.nansum(rr)
        ab = np.nansum(np.where(np.isnan(rr), 0.0, rr - c[idx]))
        ix = np.nansum(np.where(np.isnan(rr), 0.0, rr - x[idx]))
        out[leg] = (raw, ab, ix)
    return out


def split_session(ret, opn, cls):
    """(overnight, intraday) pieces of one session from its total return, open and close (raw prices)."""
    ok = np.isfinite(opn) & np.isfinite(cls) & (opn > 0) & (cls > 0) & np.isfinite(ret)
    idr = np.where(ok, cls / np.where(ok, opn, 1.0) - 1, np.nan)
    on = np.where(ok, (1 + ret) / (1 + idr) - 1, np.nan)
    return on, idr


def subday_legs(s, cm, ix):
    """s, cm, ix: dicts with keys on0, id0, on1, id1 (stock, control mean, index). Returns leg -> (raw, ab, idx)."""
    out = {}
    for leg, key in [("d0_on", "on0"), ("d0_id", "id0"), ("on1", "on1"), ("id1", "id1")]:
        v = s[key]
        out[leg] = (v, v - cm[key] if np.isfinite(cm[key]) else np.nan, v - ix[key] if np.isfinite(ix[key]) else np.nan)
    return out


def flatten(legs: dict) -> dict:
    row = {}
    for leg, (raw, ab, ixa) in legs.items():
        row[f"raw_{leg}"] = raw
        row[f"ab_{leg}"] = ab
        row[f"ix_{leg}"] = ixa
    return row
