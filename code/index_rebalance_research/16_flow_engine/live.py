"""Live helpers: index float cap today, current constituents' index shares, TSX share updates from MiG files,
and estimated float for a new name.

Index cap today (units of index currency):
  SPTSX_COMP  the TSX shadow's latest logged estimate (run_meta.json M_index_float_cap_cad), else as below
  others      last R2 index MV (pi.index_mv, to 2026-06-23) x (index price level today / level then), levels from
              yfinance; for TSX 60 the XIC-overlap method of the validation
Constituent index shares F_i = H_i / k with k = fund AUM / index cap (holdings.py docstring).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import holdings as H
from common import ETF, INDEX, MKT, MOC_ARCHIVE, SHADOW_TSX
from pi import index_mv

LEVEL_TICKER = {"SPTSX_COMP": "^GSPTSE", "STOXX600": "^STOXX", "FTSE100": "^FTSE", "FTSE250": "^FTMC",
                "DAX": "^GDAXI", "MDAX": "^MDAXI", "TECDAX": "^TECDAX"}
R2_LAST = pd.Timestamp("2026-06-23")


def _levels() -> pd.DataFrame:
    p = MKT / f"index_levels_{pd.Timestamp.today():%Y%m%d}.parquet"
    if not p.exists():
        import yfinance as yf
        x = yf.download(list(LEVEL_TICKER.values()), start="2026-06-01", auto_adjust=False, progress=False)["Close"]
        x.to_parquet(p)
    return pd.read_parquet(p)


def index_cap_today(index: str) -> tuple[float, str]:
    if index == "SPTSX_COMP":
        runs = sorted((SHADOW_TSX / "runs").glob("*/*/run_meta.json"))
        live = [r for r in runs if not json.loads(r.read_text()).get("backfill", True)]
        if live:
            m = json.loads(live[-1].read_text())
            return float(m["M_index_float_cap_cad"]), f"TSX shadow {live[-1].parent.name} (as of {m['asof']})"
    if index == "SPTSX_60":
        xic = H.weights_now("XIC")
        xiu = H.weights_now("XIU")
        M, src = index_cap_today("SPTSX_COMP")
        k = xic.mv.sum() / M
        j = xiu.merge(xic[["isin", "shares"]], on="isin", suffixes=("", "_xic"))
        k60 = float(np.median(j.shares / (j.shares_xic / k)))
        return xiu.mv.sum() / k60, "XIU / XIC overlap on today's holdings"
    mv, d = index_mv(index, R2_LAST)
    lv = _levels()
    t = LEVEL_TICKER.get(index)
    if not np.isfinite(mv) or t is None or t not in lv:
        return np.nan, "no index cap"
    s = lv[t].dropna()
    then = s[s.index <= d].iloc[-1]
    now = s.iloc[-1]
    return mv * now / then, f"R2 MV {d.date()} x yfinance {t} {now:.1f}/{then:.1f} ({s.index[-1].date()})"


def members_today(index: str) -> tuple[pd.DataFrame, float, str]:
    """Today's members with index shares F (= H/k) from the index's iShares fund."""
    etf = INDEX[index]["etf"]
    h = H.weights_now(etf)
    M, src = index_cap_today(index)
    k = h.mv.sum() / M
    h = h.assign(F=h.shares / k, k=k)
    return h, M, src


# --------------------------------------------------------------------------------- TSX share updates (MiG)
def _mig_files() -> list[tuple[str, Path]]:
    """(data label, path) of every distinct TMX MiG workbook we hold (TSX shadow raw + ~/index_archive)."""
    import sys
    sys.dont_write_bytecode = True    # read-only import: leave no __pycache__ in the shadow folder
    sys.path.insert(0, str(SHADOW_TSX))
    from live_common import parse_mig   # read-only import of the shadow's parser
    seen = {}
    for p in sorted(list((SHADOW_TSX / "raw" / "mig").glob("*.xlsx")) + list((MOC_ARCHIVE / "mig").glob("*.xlsx"))):
        label, _ = parse_mig(p)
        seen.setdefault(label, p)
    return sorted(seen.items(), key=lambda kv: pd.to_datetime(kv[0].replace("-", " "), errors="coerce"))


TSX_SHARE_REF = "31-August-2026"     # MiG month-end before the last (Sep 2026) review's share cut-off, 14 Aug
TSX_SHARE_NEW = "31-October-2026"    # MiG month-end before the Dec 2026 cut-off, 13 Nov; earlier months are interim
# XIC holdings windows since the reference that contain only off-cycle changes; the 17 -> 18 Sep step is the Sep
# review itself (the 18 Sep file already shows the rebalanced fund) and is left out. None = today's file.
TSX_XIC_WINDOWS = (("2026-08-31", "2026-09-17"), ("2026-09-18", None))
TSX_OFFCYCLE_MIN = 0.02              # XIC drift below this is rounding/capping noise (90% of names < 0.85%, 20 Oct)


def _xic_offcycle_drift() -> pd.Series:
    """Per-ISIN change in S&P/TSX index shares already applied off-cycle (accelerated rule, _shared.md §4) since the
    reference MiG date: XIC shares held across TSX_XIC_WINDOWS, each window divided by its median ratio (fund flows),
    chained. The quarterly update then only has to cover what is left of the MiG O/S change."""
    out = None
    for d0, d1 in TSX_XIC_WINDOWS:
        a = H.parse("XIC", d0, offline=False)
        b = H.parse("XIC", d1, offline=False) if d1 else H.weights_now("XIC")
        m = a.merge(b, on="isin", suffixes=("0", "1"))
        r = m.shares1 / m.shares0
        r = pd.Series((r / np.median(r)).values, index=m["isin"].values)
        out = r if out is None else r * out.reindex(r.index).fillna(1.0)
    return out - 1


def tsx_share_updates(min_rel: float = 0.001) -> tuple[pd.DataFrame, str]:
    """Predicted S&P/TSX Composite quarterly share updates: F_new = F_today x O/S_new / O/S_ref, where O/S_ref is
    the MiG month-end closest before the last review's share cut-off (TSX_SHARE_REF) and O/S_new the latest MiG
    (the Dec cut-off is 13 Nov -> MiG 31 Oct, published early Nov). The reference stays fixed, so the change covers
    the whole quarter, not just the last month. Rows from a MiG month before TSX_SHARE_NEW are labelled interim.
    IWF changes are not predicted. Returns no rows until a MiG month after the reference is held."""
    import sys
    sys.dont_write_bytecode = True    # read-only import: leave no __pycache__ in the shadow folder
    sys.path.insert(0, str(SHADOW_TSX))
    from live_common import parse_mig
    files = _mig_files()
    ref = [f for f in files if f[0] == TSX_SHARE_REF]
    if not ref:
        return pd.DataFrame(), f"reference MiG {TSX_SHARE_REF} not held: no share updates"
    (lab0, p0), (lab1, p1) = ref[0], files[-1]
    if lab1 == lab0:
        return pd.DataFrame(), f"only the reference MiG month held ({lab0}): no share updates yet"
    interim = pd.to_datetime(lab1.replace("-", " ")) < pd.to_datetime(TSX_SHARE_NEW.replace("-", " "))
    tag = f" (interim: the {TSX_SHARE_NEW} MiG replaces it)" if interim else ""
    _, a = parse_mig(p0)
    _, b = parse_mig(p1)
    m = a[["root", "mig_os"]].merge(b[["root", "mig_os"]], on="root", suffixes=("0", "1"))
    m["rel"] = m.mig_os1 / m.mig_os0 - 1
    h, M, src = members_today("SPTSX_COMP")
    h["root"] = h.ticker.str.split(".").str[0]
    x = h.merge(m, on="root")
    drift = _xic_offcycle_drift().reindex(x["isin"]).fillna(0.0).values
    # net only drift that explains the MiG change (same sign); other drift is an IWF change or an event after the
    # latest MiG month, already in the index and outside this O/S comparison
    x["offcycle"] = np.where((np.abs(drift) >= TSX_OFFCYCLE_MIN) & (np.sign(drift) == np.sign(x.rel))
                             & (x.rel.abs() >= min_rel), drift, 0.0)
    x["rel_q"] = (1 + x.rel) / (1 + x.offcycle) - 1          # what the quarterly update still has to apply
    netted = x[(x.offcycle != 0) & (x.rel.abs() >= min_rel)]
    x = x[np.where(x.offcycle != 0, x.rel_q.abs() >= TSX_OFFCYCLE_MIN, x.rel_q.abs() >= min_rel)]
    out = pd.DataFrame(dict(change_type="share_update", isin=x["isin"], ticker=x.ticker, country="CA", name=x.name,
                            currency="CAD", old_index_shares=x.F, new_index_shares=x.F * (1 + x.rel_q), price=x.price,
                            prob=0.8, status="predicted",
                            source=f"TMX MiG O/S {lab0} -> {lab1}{tag}; F from XIC ({src})",
                            source_ts=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                            notes=[f"{'INTERIM ' if interim else ''}O/S {r:+.2%}"
                                   + (f", {o:+.2%} already applied off-cycle (XIC)" if o else "")
                                   + " (multi-class issuers: root-level O/S, check)" for r, o in zip(x.rel, x.offcycle)]))
    net_note = "; ".join(f"{t} O/S {r:+.1%} vs off-cycle {o:+.1%}" for t, r, o in
                         zip(netted.ticker, netted.rel, netted.offcycle))
    return out, (f"MiG {lab0} -> {lab1}{tag}; {len(out)} rows; net of off-cycle changes already in XIC"
                 + (f" ({net_note})" if net_note else ""))


def estimate_float(ticker: str, country: str) -> tuple[float, str]:
    """Float shares for a new name with no provider file: TSX shadow screen (S&P/TSX) else yfinance floatShares."""
    if country == "CA":
        runs = sorted((SHADOW_TSX / "runs").glob("*/*/screen_all.csv"))
        s = pd.read_csv(runs[-1]).drop_duplicates("symbol").set_index("symbol")
        if ticker in s.index and np.isfinite(s.float_sh.get(ticker, np.nan)):
            return float(s.float_sh[ticker]), f"TSX shadow screen float_sh ({runs[-1].parent.name})"
    from marketdata import yahoo_symbol
    import yfinance as yf
    ys = yahoo_symbol(ticker, country)
    info = yf.Ticker(ys).info or {}
    f = info.get("floatShares") or info.get("sharesOutstanding")
    return (float(f), f"yfinance {ys} floatShares (not the provider's float)") if f else (np.nan, "no float")


def tsx_screen_info(ticker: str) -> dict:
    """Price, issuer and ISIN of a TSX line from the latest shadow screen and its R2 link table (for new adds)."""
    runs = sorted((SHADOW_TSX / "runs").glob("*/*/screen_all.csv"))
    s = pd.read_csv(runs[-1]).drop_duplicates("symbol").set_index("symbol")
    out = {}
    if ticker in s.index:
        out = dict(price=float(s.close_asof[ticker]), name=s.issuer[ticker], currency="CAD")
    try:
        lk = pd.read_parquet(SHADOW_TSX / "cache" / "r2_links.parquet").sort_values("last_px")
        lk = lk.drop_duplicates("symbol", keep="last").set_index("symbol")
        if ticker in lk.index and isinstance(lk["isin"][ticker], str):
            out["isin"] = lk["isin"][ticker]
    except Exception:  # noqa: BLE001
        pass
    return out
