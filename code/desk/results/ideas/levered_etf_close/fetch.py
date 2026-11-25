"""Fetch the Alpaca SIP data the levered_etf_close test needs. Read-only market data.

    QUANTT_ENV_FILE=<path to an .env holding Alpaca data keys> python3 desk/results/ideas/levered_etf_close/fetch.py symbols
    QUANTT_ENV_FILE=... python3 desk/results/ideas/levered_etf_close/fetch.py reference
    QUANTT_ENV_FILE=... python3 desk/results/ideas/levered_etf_close/fetch.py auctions
    QUANTT_ENV_FILE=... python3 desk/results/ideas/levered_etf_close/fetch.py intraday     # ~2 hours, resumable

WHY. The idea (card `results/ideas/2026-11-24_levered_etf_close.md`) enters the underlying
of a leveraged fund at a fixed time before the close and exits at that session's closing
auction print, and (second leg) holds from that print to the next opening print. That needs
four things no desk panel holds: the official auction prints, a quote mid at the entry
minute, the last trade before the decision minute, and the session calendar with its early
closes. CRSP (R2) supplies the fund assets to 2025-12-31 (`stage` in
`results/ideas/levered_etf_reversal/stage.py`); this module fetches only market data.

WHERE IT WRITES. ~/quantt_research/levered_etf_close/ (outside the repository: vendor rows are
not committed, and nothing here writes under data/, which the live sleeve decides on).

WHAT IS FETCHED
  symbols    the underlyings, from the levered_etf_reversal mapping (pairs.parquet) and the
             fixed index-candidate list in its mapping.py, plus a single-stock placebo set
             (below). Each CRSP ticker is checked against Alpaca: the symbol's raw SIP daily
             close on 2025-12-31 must be within 1% of CRSP's price that day, or the symbol is
             refused and listed (no silent substitution).
  reference  /v2/calendar 2016-01-01..LAST; 1Day bars, raw and adjustment=all, every symbol
             plus IEF, TLT, HYG, LQD (exposures), 2015-10-01..LAST.
  auctions   /v2/stocks/auctions, every print, both sides, every exchange (the choice of the
             primary print is the analysis's, `explore.cross_print`).
  intraday   per session: 1Min bars from close-76m to close-15m (the decision inputs), and the
             last valid NBBO strictly before each entry minute close-60m, close-30m and
             close-15m (`scripts/cef/exec_capacity_open_cross.last_valid_quote`, unchanged), first
             in the second before the minute, then 10 s (120 s for the sessions fetched first),
             then 15 min for names with none.
             Index group from 2016-01-04; single stocks and their placebo from 2022-01-03.
             From 2026-11-24 20:59 ET the headline entry (close-30m) was fetched alone for every
             session not yet cached (`--entries 30`), and the neighbours (close-60m and
             close-15m) on alternate sessions (`--entries 60,15 --shard 0/2`), because another
             job was sharing the account's rate budget.

SINGLE-STOCK PLACEBO, FIXED BEFORE ANY RETURN WAS READ: the 100 common stocks of the largest
1,000 (crsp_top4500_daily, cap_rank <= 1000 on any day of 2022-2025) with no mapped fund in
any year, ranked by median dollar volume over 2022-2025, alive on 2025-12-31.

THE PROD CLOCK. The cef key is the core book's paper account. The fetch sleeps through
15:00-16:30 and 18:30-19:45 New York time on weekdays, when the VM's session, collector and
shadow runner use the account, and paces at REQ_SLEEP_S, so a research pull never takes the
rate budget from an order send. The client is GET-only (`alpaca_get.DataOnly`).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # ct_root, beside this file
from ct_root import credittrading_root  # noqa: E402
REPO = credittrading_root()   # CreditTrading, named on PYTHONPATH (ct_root.py says why)
import alpaca_get as A  # noqa: E402

NY = ZoneInfo("America/New_York")
RES = Path.home() / "quantt_research"
PRIOR = RES / "levered_etf_reversal"
OUT = RES / "levered_etf_close"
FIRST = dt.date(2016, 1, 4)              # Alpaca's SIP history starts in 2016
SINGLE_FIRST = dt.date(2022, 1, 3)       # the first single-stock leveraged fund in the mapping is 2022
LAST = dt.date(2026, 11, 24)             # the last complete session when this was written (20:20 ET)
ENTRY_MIN = (60, 30, 15)                 # entry minutes before the close; 30 is the headline (card)
BAR_FROM_MIN, BAR_TO_MIN = 76, 15        # 1Min bars fetched over [close-76m, close-15m)
EXPOSURE = ("IEF", "TLT", "HYG", "LQD")
N_PLACEBO = 100
QUIET = ((dt.time(15, 0), dt.time(16, 30)), (dt.time(18, 30), dt.time(19, 45)))


def _iso(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def et(d: dt.date, hhmm: str | dt.time) -> pd.Timestamp:
    t = hhmm if isinstance(hhmm, dt.time) else dt.time(*map(int, hhmm.split(":")))
    return pd.Timestamp(dt.datetime.combine(d, t, NY)).tz_convert("UTC")


def wait_out_prod_windows() -> None:
    """Sleep while the VM's prod jobs may be using the account (weekdays, New York time)."""
    while True:
        now = dt.datetime.now(NY)
        if now.weekday() < 5 and any(a <= now.time() < b for a, b in QUIET):
            print(f"{now:%H:%M} ET: inside a prod window; sleeping 5 minutes", flush=True)
            time.sleep(300)
            continue
        return


# ------------------------------------------------------------------ symbols

def symbols(c) -> dict:
    sys.path.insert(0, str(REPO / "results" / "ideas" / "levered_etf_reversal"))
    from mapping import INDEX_UNDERLYINGS
    pairs = pd.read_parquet(PRIOR / "pairs.parquet")
    etf = pd.read_parquet(PRIOR / "etf_daily.parquet", columns=["permno", "ticker", "date", "prc"])
    etf["date"] = pd.to_datetime(etf["date"])
    last = etf[etf["date"] == etf["date"].max()].set_index("ticker")
    idx = [t for t in INDEX_UNDERLYINGS if t in last.index]
    top = pd.read_parquet(RES / "crsp_top4500_daily.parquet", columns=["permno", "ticker", "date", "prc", "dollar_volume", "cap_rank"])
    top["date"] = pd.to_datetime(top["date"])
    top = top[top["date"] >= "2022-01-01"]
    big = set(top.loc[top["cap_rank"] <= 1000, "permno"])
    mapped = set(pairs.loc[pairs["kind"] == "single", "under"])
    end = top[top["date"] == top["date"].max()].set_index("permno")
    med = top[top["permno"].isin(big - mapped)].groupby("permno")["dollar_volume"].median().sort_values(ascending=False)
    placebo = [int(p) for p in med.index if p in end.index][: N_PLACEBO + 20]     # 20 spare for refusals
    rows = [{"group": "index", "permno": int(last.loc[t, "permno"]), "crsp_ticker": t, "crsp_px": float(abs(last.loc[t, "prc"]))} for t in idx]
    rows += [{"group": "single", "permno": int(p), "crsp_ticker": str(end.loc[p, "ticker"]), "crsp_px": float(abs(end.loc[p, "prc"]))} for p in sorted(mapped)]
    rows += [{"group": "single_placebo", "permno": int(p), "crsp_ticker": str(end.loc[p, "ticker"]), "crsp_px": float(abs(end.loc[p, "prc"]))} for p in placebo]
    S = pd.DataFrame(rows)
    # Alpaca spells a share class with a dot; CRSP's ticker field drops the class. Only
    # Berkshire's B shares are in the mapped set; any other mismatch is refused below.
    S["symbol"] = S["crsp_ticker"].replace({"BRK": "BRK.B"})
    d0 = dt.date(2025, 12, 31)
    lo, hi = et(d0, "00:00"), et(d0, "20:00")
    got = {}
    for i in range(0, len(S), 100):
        chunk = S["symbol"].iloc[i:i + 100].tolist()
        b = A.paged(c, "/v2/stocks/bars", {"symbols": ",".join(chunk), "timeframe": "1Day", "start": _iso(lo), "end": _iso(hi),
                                           "feed": "sip", "adjustment": "raw", "limit": 10000}, "bars")
        got.update({s: v[0]["c"] for s, v in b.items() if v})
    S["alpaca_close_2025_12_31"] = S["symbol"].map(got)
    S["gap"] = (S["alpaca_close_2025_12_31"] / S["crsp_px"] - 1).abs()
    S["ok"] = S["gap"] <= 0.01
    bad = S[~S["ok"]]
    print("refused (no Alpaca bar on 2025-12-31 or price more than 1% from CRSP):")
    print(bad[["group", "crsp_ticker", "symbol", "crsp_px", "alpaca_close_2025_12_31"]].to_string())
    S = S[S["ok"]].copy()
    keep_p = S[S["group"] == "single_placebo"].head(N_PLACEBO).index
    S = S[(S["group"] != "single_placebo") | S.index.isin(keep_p)]
    S.to_parquet(OUT / "symbols.parquet", index=False)
    print(S.groupby("group").size())
    return {"n": int(len(S))}


def load_symbols() -> pd.DataFrame:
    f = OUT / "symbols.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f}: run `fetch.py symbols` first")
    return pd.read_parquet(f)


# ------------------------------------------------------------------ reference

def reference(c) -> None:
    from quantt.broker.alpaca import PAPER_BASE
    cal = c._get(PAPER_BASE, "/v2/calendar", {"start": "2015-10-01", "end": LAST.isoformat()})
    if not isinstance(cal, list) or not cal:
        raise A.FetchError("GET /v2/calendar returned no sessions")
    C = pd.DataFrame([{"date": r["date"], "open": r["open"], "close": r["close"]} for r in cal])
    C.to_parquet(OUT / "calendar.parquet", index=False)
    print("calendar", len(C), C["date"].min(), C["date"].max(), "early closes:", int((C["close"] != "16:00").sum()))
    syms = sorted(set(load_symbols()["symbol"]) | set(EXPOSURE))
    lo, hi = et(dt.date(2015, 10, 1), "00:00"), et(LAST, "20:00")
    for adj in ("raw", "all"):
        rows = []
        for i in range(0, len(syms), 50):
            chunk = syms[i:i + 50]
            b = A.paged(c, "/v2/stocks/bars", {"symbols": ",".join(chunk), "timeframe": "1Day", "start": _iso(lo), "end": _iso(hi),
                                               "feed": "sip", "adjustment": adj, "limit": 10000}, "bars")
            for s, v in b.items():
                rows += [{"symbol": s, "t": r["t"], "o": r["o"], "h": r["h"], "l": r["l"], "c": r["c"], "v": r["v"], "vw": r.get("vw")} for r in v]
        D = pd.DataFrame(rows)
        D.to_parquet(OUT / f"daily_{adj}.parquet", index=False)
        print("daily", adj, len(D), D["symbol"].nunique(), D["t"].min(), D["t"].max(), "requests so far", c.requests, flush=True)


# ------------------------------------------------------------------ auctions

def auctions(c) -> None:
    S = load_symbols()
    out = OUT / "auctions"
    out.mkdir(exist_ok=True)
    for yr in range(FIRST.year, LAST.year + 1):
        f = out / f"{yr}.parquet"
        if f.exists():
            continue
        wait_out_prod_windows()
        d0 = max(dt.date(yr, 1, 1), FIRST)
        d1 = min(dt.date(yr, 12, 31), LAST)
        syms = sorted(S.loc[(S["group"] == "index") | (yr >= SINGLE_FIRST.year), "symbol"])
        rows = []
        for i in range(0, len(syms), 50):
            chunk = syms[i:i + 50]
            got = A.paged(c, "/v2/stocks/auctions", {"symbols": ",".join(chunk), "start": _iso(et(d0, "00:00")), "end": _iso(et(d1, "20:00")),
                                                     "feed": "sip", "limit": 10000}, "auctions")
            for sym, days in got.items():
                if sym not in chunk:
                    raise A.FetchError(f"auctions returned unrequested symbol {sym}")
                for day in days:
                    for side, key in (("open", "o"), ("close", "c")):
                        for p in day.get(key) or []:
                            for k in ("x", "c", "p", "s", "t"):
                                if k not in p:
                                    raise A.FetchError(f"{sym} {day['d'][:10]} {side}: print has no {k!r}")
                            rows.append({"date": day["d"][:10], "symbol": sym, "side": side, "x": p["x"], "cond": p["c"],
                                         "p": float(p["p"]), "s": float(p["s"]), "t": p["t"]})
        D = pd.DataFrame(rows)
        D.to_parquet(f, index=False)
        print("auctions", yr, len(D), D["symbol"].nunique() if len(D) else 0, "requests", c.requests, flush=True)


# ------------------------------------------------------------------ intraday

def _last_bar_before(rows: list, cut: pd.Timestamp, oldest: pd.Timestamp):
    """The last 1Min bar whose END (start + 60 s) is at or before `cut`, and not older than
    `oldest`: its close is the last trade before `cut`. None if there is none."""
    best = None
    for r in rows:
        t0 = pd.Timestamp(r["t"])
        if t0 + pd.Timedelta(seconds=60) <= cut and t0 >= oldest:
            if best is None or t0 > pd.Timestamp(best["t"]):
                best = r
    return best


def intraday_day(c, d: dt.date, close_hhmm: str, syms: list, entries=ENTRY_MIN) -> dict:
    from scripts.cef.exec_capacity_open_cross import last_valid_quote
    close = et(d, close_hhmm)
    mn = lambda k: pd.Timedelta(minutes=k)  # noqa: E731
    bars = A.paged(c, "/v2/stocks/bars", {"symbols": ",".join(syms), "timeframe": "1Min", "start": _iso(close - mn(BAR_FROM_MIN)),
                                         "end": _iso(close - mn(BAR_TO_MIN)), "feed": "sip", "adjustment": "raw", "limit": 10000}, "bars")
    rec = {"date": d.isoformat(), "close": close_hhmm, "entries": {}}
    for k in entries:
        tau = close - mn(k)
        cut = tau - pd.Timedelta(seconds=60)                    # the decision uses trades strictly before tau - 60 s
        sig = {}
        for s in syms:
            b = _last_bar_before(bars.get(s, []), cut, close - mn(BAR_FROM_MIN))
            vol = float(sum(r["v"] for r in bars.get(s, []) if pd.Timestamp(r["t"]) + pd.Timedelta(seconds=60) <= cut))
            sig[s] = None if b is None else {"bar_t": b["t"], "c": float(b["c"]), "vol_window": vol}
        # the entry: the last valid NBBO strictly before tau
        q = {}
        missing = list(syms)
        # Windows widen only for names with no valid quote yet. The answer does not depend on
        # them (the last valid NBBO before tau within 15 minutes); they only bound the pages.
        # Until 2026-11-24 20:55 ET the middle window was 120 s; 10 s pages less on thin ETFs.
        for back in (pd.Timedelta(seconds=1), pd.Timedelta(seconds=10), pd.Timedelta(minutes=15)):
            if not missing:
                break
            got = A.paged(c, "/v2/stocks/quotes", {"symbols": ",".join(missing), "start": _iso(tau - back), "end": _iso(tau),
                                                  "feed": "sip", "limit": 10000}, "quotes")
            for s in missing:
                lq = last_valid_quote(got.get(s, []), tau)
                if lq is not None:
                    q[s] = {"bp": lq["bp"], "ap": lq["ap"], "mid": lq["mid"], "hs_bp": lq["hs_bp"], "t": lq["t"].isoformat(), "lookback_s": back.total_seconds()}
            missing = [s for s in missing if s not in q]
        rec["entries"][str(k)] = {"tau": tau.isoformat(), "decision_cut": cut.isoformat(), "signal": sig, "quote": q, "no_quote": missing}
    return rec


def _with_network_retry(fn, d, tries: int = 20, wait_s: int = 60):
    """Re-run one session's fetch when the network itself failed (the client raises with no
    HTTP status after its own retries: a DNS or connection error; this laptop lost DNS at
    22:51 ET on 2026-11-24 and every shard died). The same requests are made again; nothing
    is filled or skipped. Any HTTP error from Alpaca, and a network that stays down for
    `tries` x `wait_s`, still raise."""
    from quantt.broker.alpaca import AlpacaHTTPError
    for i in range(tries):
        try:
            return fn()
        except AlpacaHTTPError as e:
            if getattr(e, "status", None) is not None or i == tries - 1:
                raise
            print(f"{d}: network error ({str(e)[:80]}...); retrying in {wait_s} s ({i + 1}/{tries})", flush=True)
            time.sleep(wait_s)


def traded_on() -> dict:
    """{date: set of symbols with a raw SIP daily bar and positive volume}. A symbol that did not
    trade that session (not yet listed, halted all day) is not asked for; it is not a gap."""
    D = pd.read_parquet(OUT / "daily_raw.parquet", columns=["symbol", "t", "v"])
    D = D[D["v"] > 0]
    D["date"] = pd.to_datetime(D["t"]).dt.tz_convert(NY).dt.date
    return D.groupby("date")["symbol"].agg(set).to_dict()


def intraday(c, only_first: int | None = None, shard: tuple[int, int] = (0, 1), entries=ENTRY_MIN) -> None:
    """One JSON per session under intraday/ (all three entries), or, for a pass over fewer
    entries, under intraday_e<k...>/ for sessions intraday/ does not hold. The headline entry was
    fetched first this way on 2026-11-24, because another job shared the account's rate budget;
    the neighbours follow. explore.py reads an entry from intraday/ if the session is there,
    else from any intraday_e*/ file holding that entry."""
    S = load_symbols()
    traded = traded_on()
    cal = pd.read_parquet(OUT / "calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    cal = cal[(cal["date"] >= FIRST) & (cal["date"] <= LAST)]
    full = OUT / "intraday"
    full.mkdir(exist_ok=True)
    out = full if tuple(entries) == tuple(ENTRY_MIN) else OUT / ("intraday_e" + "_".join(str(k) for k in entries))
    out.mkdir(exist_ok=True)
    idx_syms = sorted(S.loc[S["group"] == "index", "symbol"])
    all_syms = sorted(S["symbol"])
    todo = [r for j, r in enumerate(cal.itertuples()) if j % shard[1] == shard[0] and not (out / f"{r.date.isoformat()}.json").exists()
            and not (full / f"{r.date.isoformat()}.json").exists()]
    if only_first:
        todo = todo[:only_first]
    t0 = time.monotonic()
    for i, r in enumerate(todo):
        wait_out_prod_windows()
        syms = [x for x in (all_syms if r.date >= SINGLE_FIRST else idx_syms) if x in traded.get(r.date, set())]
        if not syms:
            raise A.FetchError(f"{r.date}: a calendar session with no daily bar for any symbol")
        rec = _with_network_retry(lambda: intraday_day(c, r.date, r.close, syms, entries), r.date)
        tmp = out / f".{r.date.isoformat()}.tmp"
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, out / f"{r.date.isoformat()}.json")
        if i % 25 == 0:
            el = time.monotonic() - t0
            print(f"{r.date} done {i + 1}/{len(todo)}; requests {c.requests}; {el / 60:.1f} min", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("symbols", "reference", "auctions", "intraday"))
    ap.add_argument("--first", type=int, default=None, help="intraday: only the first N sessions not yet cached (a probe)")
    ap.add_argument("--shard", default="0/1", help="intraday: i/n, this process takes sessions i, i+n, ... (two processes share the rate budget)")
    ap.add_argument("--entries", default="60,30,15", help="intraday: entry minutes before the close to fetch")
    ap.add_argument("--sleep", type=float, default=A.REQ_SLEEP_S, help="seconds between request starts in this process")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    c = A.make_client()
    c.min_interval = a.sleep
    if a.step == "symbols":
        symbols(c)
    elif a.step == "reference":
        reference(c)
    elif a.step == "auctions":
        auctions(c)
    else:
        i, n = (int(x) for x in a.shard.split("/"))
        intraday(c, a.first, (i, n), tuple(int(x) for x in a.entries.split(",")))
    print("requests", c.requests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
