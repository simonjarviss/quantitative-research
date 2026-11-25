"""Explore: ride the leveraged funds' closing trade inside the session (levered_etf_close).

    QUANTT_ENV_FILE=<env> python3 desk/results/ideas/levered_etf_close/fetch.py symbols|reference|auctions|intraday
    python3 desk/results/ideas/levered_etf_close/fetch_proshares.py
    python3 desk/results/ideas/levered_etf_close/explore.py            # writes results.json and equity.png here
    python3 desk/results/ideas/levered_etf_close/render.py             # the card's tables from results.json

THE RULE was fixed in the card (results/ideas/2026-11-24_levered_etf_close.md, "The rule,
fixed before any forward return was read") before this file read any return. In short:

CONVENTION (NOT shift(2); asserted by harness.evaluate_window on every book)
  decided[t]  tau - 30 s, using only data from before tau - 60 s
  entered[t]  tau, at the last valid NBBO mid strictly before tau (fetch.py)
  exited[t]   session t's close (16:00, 13:00 on a half day), at the primary exchange's
              closing-cross print (the largest condition-6 print of the day, kept only if it
              is on the symbol's modal exchange for that month)
  leg 2       entered at t's closing print, exited at t+1's opening-cross print (condition O,
              the same primary rule), split- and dividend-adjusted
  tau = close - 30 min (headline); close - 60 and close - 15 are the neighbours.

SIGNAL (all dated before tau - 60 s)
  r     return from t-1's closing print to the last trade before tau - 60 s (the close of the
        last 1-minute bar ending by then, within 76 minutes of the close), carried through
        t's distribution and split factor g: r = P g_t / C_t-1 - 1. g is CRSP's to 2025-12-31
        and Alpaca's adjusted-to-raw ratio after (`factor_ratio` says why)
  k     sum over the underlying's funds of L(L-1) x assets at t-1 (CRSP shares x price on the
        session before t; ProShares' own file in 2026, ProShares funds only)
  S     k |r| / ADV, ADV = median dollar volume (v x vwap) over the 20 sessions before t
  SA    k |r| / cross, cross = median closing-cross dollar size over the 20 sessions before t
BOOKS (calendar time, gross 1 on each day with a position, side = sign(r)): T (S >= S*),
  T-cont (weight ~ S), C-mom (all), C-big (|r| >= own trailing-252 90th percentile), C-open
  (sign of the move from t's opening print), P-low (C-big days with S < S*/4), P-none (single
  stocks: the 100-stock placebo set; index: C-big days with no fund alive). S* = 90th
  percentile of S over eligible name-days with a fund alive.

CORRECTED AFTER REVIEW (2026-11-25):
  * S*, SA* and S3* were full-sample quantiles (they read the future). They are now expanding
    quantiles over the sessions BEFORE the one they are used on (`pit_threshold`), used once
    MIN_PRIOR_OBS name-days exist, and every book in a sample starts on that session (index
    2016-05-05, single stocks 2023-06-09), so the controls cover the days T could trade.
    `threshold_comparison` shows the full-sample and trailing-252 versions beside it.
  * P&L is dated by the session each window ends on (harness.evaluate_window): leg 2 and leg 3
    are the next session's. The exposures and sleeve correlations of those legs were a day
    early before. Every session is passed to the harness, flat ones included.
  * the neighbour entries' big-move threshold is each name's last 126 eligible observations
    (`big_moves`, "eligible126"): they were fetched on alternate sessions only.
  * the exclusion counters are per sample label (they summed seven samples).
  * added checks: the leg-2 decomposition into the overnight gap and the US session
    (`leg2_gap_us_decomposition`), the distribution factor rebuilt point in time
    (`g_t_check`), and the HAC alpha t at newey_west_t's own lag rule beside 5 lags.

ADDED AFTER THE RULE WAS FIXED, each labelled where it is built and in results.json:
  before the full run, after a development run on 2016-01..2016-09: T-SA, T-BIG, V5 (ProShares'
  own assets for its funds), V6 (ProShares-only assets), V7 (foreign-market ETFs), the t+1
  open-to-close diagnostic; after the interim read of 2016-01..2019-07: C-mom-Tnames, the
  within-ETF regression, and leg 3 (`run_leg3`, the next-session fade, decided after t's close).
  The neighbour entries were fetched on alternate sessions (`main`, `fetch.py`).

NO LOOKAHEAD, AND HOW IT IS CHECKED
  * every rolling statistic (ADV, cross size, the big-move quantile, the regime labels) is
    computed on rows before t (`.shift(1)` after the window);
  * k uses assets at t-1: CRSP's row for the session before t, ProShares' row dated t-1;
  * evaluate_window refuses a held weight whose newest datum (the signal bar's END) is not
    before the decision time, and an entry at or before the decision;
  * `truncation_check` re-builds the headline book with every input cut at probe dates and
    requires the probe rows unchanged (`check_truncation`).
NO SILENT FALLBACKS: a name-day missing any input is not eligible, and every such exclusion is
counted in results.json ("exclusions"). A fund alive on an underlying whose assets are unknown
that day makes the underlying's k unknown, not smaller.
Gross; no cost is modelled (desk research convention). The quoted half-spread at entry is shown
beside the edge per trade as a feasibility readout, not a deduction.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # ct_root, beside this file
from ct_root import credittrading_root  # noqa: E402
REPO = credittrading_root()   # CreditTrading, named on PYTHONPATH (ct_root.py says why)
from scripts.research import explore, harness, inference  # noqa: E402

assert pd.get_option("compute.use_numexpr") is False
assert harness.EXEC_LAG == 2          # untouched: this sleeve uses evaluate_window instead

NY = "America/New_York"
RES = Path.home() / "quantt_research"
PRIOR = RES / "levered_etf_reversal"
CACHE = RES / "levered_etf_close"
ENTRIES = (60, 30, 15)
HEAD = 30
MIN_ADV, MIN_PRICE = 20e6, 5.0
BROAD = ("SPY", "QQQ", "DIA")
# V7, ADDED AFTER THE DEVELOPMENT RUN'S EVENT LIST WAS SEEN (2016 only), BEFORE ANY SUBGROUP RETURN
# WAS READ: ETFs on foreign markets that are shut at the New York close. A leveraged fund on
# such an index cannot trade the index's own stocks at its NAV time, so its hedge goes through
# the US-listed ETF or swaps written on it; the flow share against the ETF's own volume is then
# the right measure, as it is not for a US sector ETF (whose funds' flow lands in the stocks).
FOREIGN = ("EEM", "EFA", "FXI", "EWZ", "EWJ", "EWY", "EWW", "EWT", "INDA", "MCHI", "ASHR", "KWEB")
INDEX_MULTIPLES = {-3.0, -2.0, -1.0, 2.0, 3.0}
Q_EVENT, Q_BIG = 0.90, 0.90
INDEX_FIRST, SINGLE_FIRST = pd.Timestamp("2016-01-04"), pd.Timestamp("2022-01-03")
CRSP_LAST = pd.Timestamp("2025-12-31")
SPLIT = harness.HOLDOUT_START
ERAS = (("2016-19", "2016-01-01", "2019-12-31"), ("2020-22", "2020-01-01", "2022-12-31"),
        ("2023-25", "2023-01-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31"))
CAPITAL = (1e6, 5e6, 10e6)
# POINT-IN-TIME THRESHOLDS (review, 2026-11-25). S*, SA* and S3* were the 90th percentile over the
# whole 2016-2025 sample: a threshold read from the future. Each is now the 90th percentile of
# the pool's values on the sessions BEFORE the one it is used on ("expanding"), used only once
# that pool holds MIN_PRIOR_OBS name-days; "trailing252" (the 252 sessions before) and the old
# full-sample value are reported beside it (`threshold_comparison`).
MIN_PRIOR_OBS = 2000
THRESHOLD_MODE = "expanding"
SANITY_MID_GAP = 0.10   # |entry mid / signal trade - 1| above this is a data error, counted, not traded
SANITY_MOVE = 0.50      # |r| above this is a data error (an unadjusted split), counted, not traded
EXCL: dict = {}
SESSION_NO: dict = {}   # session -> its number from 2016-01-04, as fetch.py numbers them for its shards


def count(name: str, n) -> None:
    EXCL[name] = EXCL.get(name, 0) + int(n)


# ===================================================================== inputs

def calendar() -> pd.DataFrame:
    C = pd.read_parquet(CACHE / "calendar.parquet")
    C["date"] = pd.to_datetime(C["date"])
    C = C[(C["date"] >= INDEX_FIRST)].set_index("date").sort_index()
    C["open_ts"] = [pd.Timestamp(f"{d.date()} {o}", tz=NY) for d, o in zip(C.index, C["open"])]
    C["close_ts"] = [pd.Timestamp(f"{d.date()} {c}", tz=NY) for d, c in zip(C.index, C["close"])]
    return C


def daily(sessions: pd.DatetimeIndex, syms: list) -> dict:
    """Wide frames from the SIP daily bars: raw close, adjustment factor, dollar volume."""
    out = {}
    for adj in ("raw", "all"):
        D = pd.read_parquet(CACHE / f"daily_{adj}.parquet")
        D["date"] = pd.to_datetime(D["t"]).dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
        if D.duplicated(["date", "symbol"]).any():
            raise ValueError(f"daily_{adj}: two bars for one symbol-day")
        out[adj] = D
    raw = out["raw"].pivot(index="date", columns="symbol", values="c").reindex(index=sessions, columns=syms)
    allc = out["all"].pivot(index="date", columns="symbol", values="c").reindex(index=sessions, columns=syms)
    dv = (out["raw"]["v"] * out["raw"]["vw"]).groupby([out["raw"]["date"], out["raw"]["symbol"]]).sum().unstack()
    dv = dv.reindex(index=sessions, columns=syms)
    f = allc / raw
    return {"raw_close": raw, "f": f, "dv": dv, "all_close": allc}


def factor_ratio(sessions: pd.DatetimeIndex, S_: pd.DataFrame, DLY: dict) -> tuple[pd.DataFrame, dict]:
    """g[t, i]: the factor that carries a raw price on t-1 to a total-return basis on t, i.e.
    (1 + total return) = (raw price on t / raw price on t-1) x g. 1 on most days; on an
    ex-date it adds the distribution, on a split date it undoes the split.

    WHY CRSP AND NOT ALPACA, THROUGH 2025. Alpaca's adjustment=all daily bars missed SPY's
    2016-03-18 distribution (its factor did not move that day; CRSP's return is +0.39% against a
    -0.12% price change), found while hand-checking the panel, before any book was scored. So
    2016-2025 uses CRSP: g = (1 + ret) x prc[t-1] / prc[t], from the ETF file for the index group
    and crsp_top4500_daily for stocks. 2026 has no CRSP, and uses Alpaca's factor
    f[t] / f[t-1], f = adjusted close / raw close, labelled; how often the two disagree on the
    years both exist is reported."""
    syms = list(S_["symbol"])
    e = pd.read_parquet(PRIOR / "etf_daily.parquet", columns=["permno", "date", "ret", "prc"])
    st = pd.read_parquet(RES / "crsp_top4500_daily.parquet", columns=["permno", "date", "ret", "prc"])
    rows = []
    for grp, src in (("index", e), ("stock", st)):
        ids = S_.loc[(S_["group"] == "index") == (grp == "index"), ["permno", "symbol"]]
        x = src[src["permno"].isin(ids["permno"])].copy()
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"])
        x["prc"] = x["prc"].abs()
        x["gx"] = (1.0 + x["ret"]) * x.groupby("permno")["prc"].shift(1) / x["prc"]
        x = x.merge(ids, on="permno")
        rows.append(x[["date", "symbol", "gx"]])
    G = pd.concat(rows).pivot(index="date", columns="symbol", values="gx").reindex(index=sessions, columns=syms)
    fA = DLY["f"].reindex(columns=syms)
    gA = fA / fA.shift(1)
    both = G.notna() & gA.notna()
    gap = (G - gA).abs().where(both)
    info = {"crsp_alpaca_name_days_compared": int(both.sum().sum()),
            "share_with_factor_gap_over_5bp": float((gap > 5e-4).sum().sum() / both.sum().sum()),
            "crsp_ex_or_split_days_(g_off_1_by_5bp)": int(((G - 1).abs() > 5e-4).sum().sum()),
            "of_which_alpaca_factor_also_moved_5bp": int((((G - 1).abs() > 5e-4) & ((gA - 1).abs() > 5e-4)).sum().sum())}
    out = G.where(pd.DataFrame(np.repeat((G.index <= CRSP_LAST)[:, None], G.shape[1], axis=1), index=G.index, columns=G.columns), gA)
    return out, info


def prints(sessions: pd.DatetimeIndex, syms: list) -> dict:
    """Primary closing and opening crosses: price and size, sessions x symbols.

    The cross is the largest condition-6 (close) or condition-O (open) print of the day; the
    CTS and UTP specifications code the primary's auction trade so, and in 2016-2020 Alpaca
    lists no M/Q twin (exec_capacity_open_cross.py documents both). It is kept only when it
    is on the exchange that carries the symbol's largest cross on most days of that month,
    so a day the primary ran no cross is missing, never another venue's print."""
    rows = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(CACHE / "auctions" / "*.parquet")))])
    rows["date"] = pd.to_datetime(rows["date"])
    # Alpaca's /v2/stocks/meta/exchanges names Q "NASDAQ OMX" and T "NASDAQ Int" [V, read
    # 2026-11-24]: one venue under two tape codes. MSFT's Nasdaq cross of 2023-05-08 is listed
    # twice, as Q and as T, same size and price; on 2023-04-19 only as T. Without this, 14% of
    # Nasdaq-listed name-days read as "no primary cross". Found on the data, before any return.
    rows["x"] = rows["x"].replace({"T": "Q"})
    out, info = {}, {}
    for side, cond in (("close", "6"), ("open", "O")):
        r = rows[(rows["side"] == side) & (rows["cond"] == cond)]
        top = r.sort_values("s").groupby(["date", "symbol"]).tail(1).copy()
        top["month"] = top["date"].dt.to_period("M")
        modal = top.groupby(["symbol", "month"])["x"].agg(lambda s: s.value_counts().index[0]).rename("primary")
        top = top.join(modal, on=["symbol", "month"])
        off = top["x"] != top["primary"]
        info[f"{side}_days_largest_cross_not_on_primary"] = int(off.sum())
        top = top[~off]
        out[f"{side}_px"] = top.pivot(index="date", columns="symbol", values="p").reindex(index=sessions, columns=syms)
        out[f"{side}_sz"] = top.pivot(index="date", columns="symbol", values="s").reindex(index=sessions, columns=syms)
    # check: where Alpaca also lists an M (official close) from the same exchange, does it agree?
    m = rows[(rows["side"] == "close") & (rows["cond"] == "M")]
    six = out["close_px"].stack().rename("p6").reset_index().rename(columns={"level_0": "date"})
    mm = m.groupby(["date", "symbol"])["p"].apply(list).rename("pm").reset_index()
    j = six.merge(mm, on=["date", "symbol"])
    j["agree"] = [any(abs(p - a) < 1e-6 for p in lst) for a, lst in zip(j["p6"], j["pm"])]
    info["close_days_with_an_M_print"] = int(len(j))
    info["close_6_price_equals_an_M_price"] = float(j["agree"].mean()) if len(j) else None
    return out, info


def intraday(sessions: pd.DatetimeIndex, syms: list) -> dict:
    """Signal trade, its bar end, entry mid, half-spread, quote time; one frame per entry."""
    rec = {k: [] for k in ENTRIES}
    # a session's entries come from intraday/ (all three) if it is there, else from the
    # single-pass directories intraday_e*/ (fetch.py `--entries`)
    files = sorted(glob.glob(str(CACHE / "intraday" / "*.json")))
    full = {Path(fn).name for fn in files}
    for dn in sorted(glob.glob(str(CACHE / "intraday_e*"))):
        files += [fn for fn in sorted(glob.glob(str(Path(dn) / "*.json"))) if Path(fn).name not in full]
    have = {k: set() for k in ENTRIES}
    for fn in files:
        r = json.loads(Path(fn).read_text())
        d = pd.Timestamp(r["date"])
        for k in ENTRIES:
            if str(k) not in r["entries"]:
                continue
            if d in have[k]:
                raise ValueError(f"{d.date()} entry {k} is in two cache files")
            have[k].add(d)
            e = r["entries"][str(k)]
            cut = pd.Timestamp(e["decision_cut"])
            for s, v in e["signal"].items():
                q = e["quote"].get(s)
                end = pd.Timestamp(v["bar_t"]) + pd.Timedelta(seconds=60) if v else pd.NaT
                if v and end > cut:
                    raise ValueError(f"{d.date()} {s}: signal bar ends {end} after the decision cut {cut}")
                rec[k].append((d, s, v["c"] if v else np.nan, end, q["mid"] if q else np.nan,
                               q["hs_bp"] if q else np.nan, pd.Timestamp(q["t"]) if q else pd.NaT,
                               pd.Timestamp(e["tau"]), cut))
    out = {"missing_sessions": {k: [x for x in sessions if x not in have[k]] for k in ENTRIES}}
    for k in ENTRIES:
        D = pd.DataFrame(rec[k], columns=["date", "symbol", "sig", "sig_end", "mid", "hs_bp", "q_t", "tau", "cut"])
        piv = lambda c: D.pivot(index="date", columns="symbol", values=c).reindex(index=sessions, columns=syms)  # noqa: E731
        out[k] = {"sig": piv("sig"), "sig_end": piv("sig_end"), "mid": piv("mid"), "hs_bp": piv("hs_bp"), "q_t": piv("q_t"),
                  "tau": D.groupby("date")["tau"].first().reindex(sessions), "cut": D.groupby("date")["cut"].first().reindex(sessions)}
        # an all-missing column pivots to naive NaT; make every timestamp frame tz-aware UTC first
        for key in ("sig_end", "q_t"):
            out[k][key] = out[k][key].apply(lambda c: pd.to_datetime(c, utc=True))
        for key in ("tau", "cut"):
            out[k][key] = pd.to_datetime(out[k][key], utc=True)
        bad_q = int(out[k]["q_t"].apply(lambda c: c >= out[k]["tau"]).sum().sum())
        if bad_q:
            raise ValueError(f"entry {k}: {bad_q} entry quotes are not strictly before tau")
    return out


def fund_assets_crsp() -> pd.DataFrame:
    """Long frame (date, fund, aum_lag): CRSP shares x |price| on the CRSP session before
    `date`. A row whose shares or price is missing inside the fund's life is kept as NaN,
    so `k` can say it is unknown rather than smaller."""
    e = pd.read_parquet(PRIOR / "etf_daily.parquet", columns=["permno", "date", "prc", "shrout"])
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values(["permno", "date"])
    e["aum"] = e["shrout"] * 1000.0 * e["prc"].abs()
    sess = pd.DatetimeIndex(sorted(e["date"].unique()))
    nxt = pd.Series(sess[1:], index=sess[:-1])
    e["for_date"] = e["date"].map(nxt)          # the asset figure dated d is used on the next session
    e = e.dropna(subset=["for_date"])
    return e.rename(columns={"permno": "fund"})[["for_date", "fund", "aum"]].rename(columns={"for_date": "date", "aum": "aum_lag"})


def fund_assets_proshares(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    # `sessions` must start before the first session scored, so that t-1 exists for it
    """Long frame (date, fund, aum_lag) from ProShares' own files, matched to CRSP permnos by
    (ticker, date) through 2025 and by the ticker's permno on CRSP's last day after it."""
    e = pd.read_parquet(PRIOR / "etf_daily.parquet", columns=["permno", "ticker", "date"])
    e["date"] = pd.to_datetime(e["date"])
    last = e[e["date"] == CRSP_LAST].set_index("ticker")["permno"]
    rows = []
    for fn in sorted(glob.glob(str(CACHE / "proshares" / "*.csv"))):
        t = os.path.basename(fn)[:-4]
        x = pd.read_csv(fn)
        x["date"] = pd.to_datetime(x["Date"], format="%m/%d/%Y")
        x["ticker"] = t
        rows.append(x[["date", "ticker", "Assets Under Management"]].rename(columns={"Assets Under Management": "aum"}))
    P = pd.concat(rows)
    if P.duplicated(["date", "ticker"]).any():
        raise ValueError("a ProShares file holds two rows for one date")
    a = P[P["date"] <= CRSP_LAST].merge(e, on=["ticker", "date"], how="inner")
    b = P[P["date"] > CRSP_LAST].copy()
    b["permno"] = b["ticker"].map(last)
    count("proshares_2026_rows_without_a_crsp_fund", b["permno"].isna().sum())
    b = b.dropna(subset=["permno"])
    P = pd.concat([a, b])[["date", "permno", "aum"]]
    P = P[P["date"] >= sessions[0]]
    idx = pd.DatetimeIndex(sessions)
    nxt = pd.Series(idx[1:], index=idx[:-1])
    P["for_date"] = P["date"].map(nxt)
    count("proshares_rows_not_on_an_alpaca_session", P["for_date"].isna().sum())
    P = P.dropna(subset=["for_date"])
    return P.rename(columns={"permno": "fund", "aum": "aum_lag"})[["for_date", "fund", "aum_lag"]].rename(columns={"for_date": "date"})


def k_frame(aum: pd.DataFrame, pairs: pd.DataFrame, sessions, sym_of: dict, *, year_of_pairs=None) -> pd.DataFrame:
    """sessions x symbols: sum of L(L-1) x assets at t-1 over the underlying's funds that
    year. NaN where any such fund's asset figure is missing; 0 where no fund is alive.
    `year_of_pairs` maps a session year to the pairs year to use (2026 -> 2025: a fund's
    multiple is a prospectus fact, carried from the last year the mapping estimated it)."""
    a = aum.copy()
    a["year"] = a["date"].dt.year
    if year_of_pairs:
        a["year"] = a["year"].map(lambda y: year_of_pairs.get(y, y))
    m = a.merge(pairs[["fund", "year", "under", "lev"]], on=["fund", "year"], how="inner")
    m["k"] = m["lev"] * (m["lev"] - 1.0) * m["aum_lag"]
    m["sym"] = m["under"].map(sym_of)
    m = m.dropna(subset=["sym"])
    m["k_na"] = m["k"].isna()
    unknown = m.groupby(["date", "sym"])["k_na"].any().unstack()
    k = m.groupby(["date", "sym"])["k"].sum(min_count=1).unstack()
    k = k.where(~unknown.astype("boolean").fillna(False).astype(bool))
    cols = sorted(set(sym_of.values()))
    k = k.reindex(index=sessions, columns=cols)
    alive = m.groupby(["date", "sym"]).size().unstack().reindex(index=sessions, columns=cols).notna()
    return k.where(alive, 0.0).where(~(alive & k.isna()), np.nan)


# ===================================================================== the panel

def big_moves(x: pd.DataFrame, E: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Big-move days: |x| at or above the name's own 90th percentile of |x| before t.
    "rows252": over the 252 sessions before t (at least 126 eligible), the rule on full data.
    "eligible126": over the name's last 126 ELIGIBLE observations before t. Used for the
    neighbour entries (review, 2026-11-25), which were fetched on alternate sessions only, so
    a 252-row window held about 126 values and its 126-value minimum was met late or not at all."""
    a = x.abs().where(E)
    if rule == "rows252":
        q = a.rolling(252, min_periods=126).quantile(Q_BIG).shift(1)
    elif rule == "eligible126":
        q = pd.DataFrame(np.nan, index=a.index, columns=a.columns)
        for c in a.columns:
            v = a[c].dropna()
            q.loc[v.index, c] = v.rolling(126, min_periods=126).quantile(Q_BIG).shift(1).to_numpy()
    else:
        raise ValueError(f"unknown big-move rule {rule!r}")
    return E & (x.abs() >= q)


def pit_threshold(V: pd.DataFrame, mode: str) -> pd.Series:
    """The Q_EVENT quantile of V's values on the sessions before each session (V is NaN off the
    pool). "expanding": every earlier session; "trailing252": the 252 before. NaN until the
    window holds MIN_PRIOR_OBS values, so nothing is traded before then."""
    if mode not in ("expanding", "trailing252"):
        raise ValueError(f"unknown threshold mode {mode!r}")
    a = V.to_numpy(dtype=float)
    flat = np.concatenate([row[~np.isnan(row)] for row in a]) if len(a) else np.array([])
    ends = np.cumsum([(~np.isnan(row)).sum() for row in a])          # values on sessions 0..i end at ends[i]
    out = np.full(len(V), np.nan)
    for i in range(1, len(V)):
        lo = 0 if mode == "expanding" else (ends[i - 253] if i > 252 else 0)
        pool = flat[lo:ends[i - 1]]
        if len(pool) >= MIN_PRIOR_OBS:
            out[i] = np.quantile(pool, Q_EVENT)
    return pd.Series(out, index=V.index)


def build(group: str, cal: pd.DataFrame, DLY: dict, PR: dict, IN: dict, K: pd.DataFrame, syms: list,
          first: pd.Timestamp, last: pd.Timestamp, entry: int, big_rule: str = "rows252") -> dict:
    """Every frame the books read, for one group and one entry minute."""
    sess = cal.index[(cal.index >= first) & (cal.index <= last)]
    full = cal.index
    g = lambda F: F.reindex(index=full, columns=syms)  # noqa: E731
    C, O, CS = g(PR["close_px"]), g(PR["open_px"]), g(PR["close_sz"])
    gx, dv = g(DLY["g"]), g(DLY["dv"])
    I = IN[entry]
    P, END, M, HS = g(I["sig"]), g(I["sig_end"]), g(I["mid"]), g(I["hs_bp"])
    adv = dv.rolling(20, min_periods=15).median().shift(1)            # sessions before t
    xadv = (CS * C).rolling(20, min_periods=15).median().shift(1)
    Cprev = C.shift(1)
    r = P * gx / Cprev - 1.0                                           # gx[t]: t's distribution and split factor
    r_open = P / O - 1.0
    R1 = C / M - 1.0                                                   # tau mid -> t's closing cross
    R2 = O.shift(-1) * gx.shift(-1) / C - 1.0                          # t's close -> t+1's opening cross
    Kg = K.reindex(index=full, columns=syms)
    S = Kg * r.abs() / adv
    SA = Kg * r.abs() / xadv
    base = (adv >= MIN_ADV) & (Cprev >= MIN_PRICE)
    have = P.notna() & M.notna() & C.notna() & Cprev.notna() & gx.notna()
    sane = ((M / P - 1).abs() <= SANITY_MID_GAP) & (r.abs() <= SANITY_MOVE)
    E = base & have & sane & Kg.notna() & (r != 0)
    win = lambda F: F.loc[sess]  # noqa: E731
    if entry == HEAD:
        count(f"{group}: name-days passing ADV and price, missing an input", int(win(base & ~have).sum().sum()))
        count(f"{group}: name-days failing the sanity bounds", int(win(base & have & ~sane).sum().sum()))
        count(f"{group}: name-days with a fund alive whose assets are unknown", int(win(base & have & sane & Kg.isna()).sum().sum()))
    big = big_moves(r, E, big_rule)
    # LEG 3, ADDED AFTER THE INTERIM READ OF 2016 TO MID-2019 (2026-11-24, 21:35 ET), NOT IN THE
    # CARD'S RULE: the next-session fade the prior review found (levered_etf_reversal's reopen
    # condition), as a rule of its own. Decided after t's close on the whole day's move and t-1
    # assets; entered at t+1's opening cross, exited at t+1's closing cross (an MOO and an MOC).
    r_full = C * gx / Cprev - 1.0
    S3 = Kg * r_full.abs() / adv
    R3 = C.shift(-1) / O.shift(-1) - 1.0
    E3 = base & C.notna() & Cprev.notna() & gx.notna() & Kg.notna() & (r_full != 0) & (r_full.abs() <= SANITY_MOVE) \
        & O.shift(-1).notna() & C.shift(-1).notna()
    if entry == HEAD:
        pre3 = base & C.notna() & Cprev.notna() & gx.notna() & Kg.notna() & (r_full != 0)
        count(f"{group}: leg 3 name-days without t+1's opening or closing cross (not traded)", int(win(pre3 & ~(O.shift(-1).notna() & C.shift(-1).notna())).sum().sum()))
    big3 = big_moves(r_full, E3, "rows252")
    gap = O * gx / Cprev - 1.0                                         # t-1's close to t's open (abroad: the home session)
    us = P / O - 1.0                                                   # t's open to the signal trade (the US session so far)
    out = {k: win(v) for k, v in dict(C=C, O=O, M=M, P=P, END=END, HS=HS, r=r, r_open=r_open, R1=R1, R2=R2, K=Kg, S=S, SA=SA,
                                       E=E, big=big, adv=adv, xadv=xadv, r_full=r_full, S3=S3, R3=R3, E3=E3, big3=big3,
                                       gap=gap, us=us).items()}
    out["sessions"] = sess
    out["tau"] = I["tau"].reindex(sess)
    out["close_ts"] = cal["close_ts"].reindex(sess)
    out["next_open_ts"] = cal["open_ts"].shift(-1).reindex(sess)
    out["next_close_ts"] = cal["close_ts"].shift(-1).reindex(sess)
    return out


def run_leg3(B: dict, th3: pd.Series, regimes: dict) -> tuple[dict, dict]:
    """Leg 3 (not pre-specified; see build): fade the whole day's move over t+1's session.
    Scored through evaluate_window: decided 15 minutes after t's close, entered at t+1's open,
    exited at t+1's close; the weight's newest datum is t's closing cross."""
    E3, big3, S3, r_full = B["E3"], B["big3"], B["S3"], B["r_full"]
    side3 = -np.sign(r_full)
    alive = E3 & (B["K"] > 0)
    S3rel = S3.div(th3, axis=0)
    T3 = alive & (S3rel >= 1.0)
    books = {"T3": T3, "T3-BIG": T3 & big3, "C3-all": E3, "C3-big": big3, "P3-low": big3 & (S3rel < 0.25),
             "C3-Tnames": E3 & pd.DataFrame(np.repeat(T3.any().to_numpy()[None, :], len(E3), axis=0), index=E3.index, columns=E3.columns)}
    out, pnl = {}, {}
    for name, mask in books.items():
        p = leg3_pnl(B, mask)
        pnl[name] = p
        out[name] = {"book": table(p, regimes), "per_event": events(side3.where(mask) * B["R3"], mask), "events": int(mask.sum().sum())}
    import statsmodels.api as sm
    for a_, b_ in (("T3", "C3-big"), ("T3", "C3-Tnames"), ("T3-BIG", "P3-low")):
        Xb = sm.add_constant(pnl[b_].rename(b_))
        fit = sm.OLS(pnl[a_], Xb).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        L = int(4 * (len(Xb) / 100) ** (2 / 9))
        fit2 = sm.OLS(pnl[a_], Xb).fit(cov_type="HAC", cov_kwds={"maxlags": L})
        out[f"{a_} on {b_}"] = {"alpha_bp_per_day": float(fit.params["const"] * 1e4), "alpha_t": float(fit.tvalues["const"]),
                                "alpha_t_at_nw_rule_lags": float(fit2.tvalues["const"]), "nw_rule_lags": L,
                                "beta": float(fit.params[b_]), "r2": float(fit.rsquared)}
    out["S3_threshold_first_session"] = str(th3.first_valid_index().date()) if th3.notna().any() else None
    return out, pnl


def weights(mask: pd.DataFrame, side: pd.DataFrame, scale: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calendar-time weights: side x (1 or scale) on masked cells, gross 1 on each day with any."""
    x = (scale if scale is not None else pd.DataFrame(1.0, index=mask.index, columns=mask.columns)).where(mask, 0.0)
    g = x.abs().sum(axis=1)
    W = x.div(g.where(g > 0, 1.0), axis=0) * side.where(mask, 0.0)
    return W.fillna(0.0)


def window_pnl(W: pd.DataFrame, R: pd.DataFrame, B: dict, *, info: pd.DataFrame, decided: pd.Series,
               entered: pd.Series, exited: pd.Series) -> pd.Series:
    """harness.evaluate_window over EVERY session of B (flat rows included, as the harness
    requires), its P&L dated by the session each window ends on. A window that ends after the
    group's last session (decided on that session, earned on the next) is outside the sample
    and is dropped, and counted: the sample is the sessions P&L is earned on."""
    held_ = W != 0
    info = info.where(held_).apply(lambda c: pd.to_datetime(c, utc=True))
    res = harness.evaluate_window(W, R.where(held_, 0.0), info=info, decided=decided, entered=entered, exited=exited)
    p = res["pnl"]
    after = p.index > B["sessions"][-1]
    if after.any():
        count("windows ending after a group's last session (outside its sample)", int((p[after] != 0).sum()))
    return p.reindex(B["sessions"])


def score(W: pd.DataFrame, B: dict, leg: int) -> tuple[pd.Series, dict]:
    """Daily P&L through harness.evaluate_window, the convention asserted, dated by the session
    each window ends on (leg 1: the decision session; leg 2: the next session, whose opening
    cross ends it). Before the review (2026-11-25) leg 2 was dated by its decision row."""
    if leg == 1:
        R, entered, exited = B["R1"], B["tau"], B["close_ts"]
    else:
        R, entered, exited = B["R2"], B["close_ts"], B["next_open_ts"]
        keep = R.notna()
        dropped = int(((W != 0) & ~keep).sum().sum())
        if dropped:
            W = W.where(keep, 0.0)
            g = W.abs().sum(axis=1)
            W = W.div(g.where(g > 0, 1.0), axis=0)
    rows = W.abs().sum(axis=1) > 0
    pnl = window_pnl(W, R, B, info=B["END"], decided=B["tau"] - pd.Timedelta(seconds=30), entered=entered, exited=exited)
    meta = {"active_days": int(rows.sum()), "names_per_active_day": float((W != 0).sum(axis=1)[rows].mean()) if rows.any() else 0.0,
            "turnover_per_year_full_sample": float(2.0 * rows.sum() * 252 / len(B["sessions"]))}
    if leg == 2:
        meta["leg2_cells_dropped_no_next_open"] = dropped
    return pnl, meta


def row(pnl: pd.Series) -> dict:
    s = explore.stats(pnl)
    s["nw_t"] = float(inference.newey_west_t(pnl.to_numpy())) if pnl.std() > 0 else None
    return s


def table(pnl: pd.Series, regimes: dict) -> dict:
    t = {"full": row(pnl), "early (to 2022)": row(pnl.loc[:SPLIT - pd.Timedelta(days=1)]) if (pnl.index < SPLIT).sum() > 2 else None,
         "late (2023 on)": row(pnl.loc[SPLIT:]) if (pnl.index >= SPLIT).sum() > 2 else None}
    for lab, a, b in ERAS:
        seg = pnl.loc[a:b]
        if len(seg) > 60:
            t[lab] = row(seg)
    for name, lab in regimes.items():
        l = lab.reindex(pnl.index)
        for v in (True, False):
            seg = pnl[l == v]
            if len(seg) > 60:
                t[f"{name}={v}"] = row(seg)
    return t


def events(y: pd.DataFrame, mask: pd.DataFrame, hs: pd.DataFrame | None = None) -> dict:
    """Per-event stats (bp): mean, median, hit; NW t on the daily mean across events."""
    v = y.where(mask)
    flat = v.stack()
    if len(flat) < 3:
        return {"events": int(len(flat))}
    daily = v.mean(axis=1).dropna()
    out = {"events": int(len(flat)), "days": int(len(daily)), "mean_bp": float(flat.mean() * 1e4),
           "median_bp": float(flat.median() * 1e4), "hit": float((flat > 0).mean()),
           "nw_t_daily_mean": float(inference.newey_west_t(daily.to_numpy())) if len(daily) > 3 and daily.std() > 0 else None}
    if hs is not None:
        h = hs.where(mask).stack()
        out["median_quoted_half_spread_bp"] = float(h.median()) if len(h) else None
        out["mean_quoted_half_spread_bp"] = float(h.mean()) if len(h) else None
    return out


# ===================================================================== the run

def run_group(group: str, B: dict, Bn: dict, th: dict, cols_placebo: list, regimes: dict, spy_win: dict,
              nb_sessions: dict | None = None) -> dict:
    """All books for one group at one entry. `Bn` holds the panels for the neighbour entries.
    `th` holds the point-in-time thresholds (Series on B's sessions): every flow comparison is
    S / S*[t] against 1, so a day before the threshold exists has no T event."""
    E, big, S, r = B["E"], B["big"], B["S"], B["r"]
    Srel, SArel = S.div(th["S"], axis=0), B["SA"].div(th["SA"], axis=0)
    side, side_open = np.sign(r), np.sign(B["r_open"])
    alive = E & (B["K"] > 0)
    books = {
        "T": (alive & (Srel >= 1.0), side, None),
        "T-cont": (alive, side, S),
        "C-mom": (E, side, None),
        "C-big": (big, side, None),
        "C-open": (E & side_open.ne(0) & side_open.notna(), side_open, None),
        "P-low": (big & (Srel < 0.25), side, None),
    }
    # V8 (the brief's suggested angle, fixed before the full run): the predicted flow against
    # the closing auction's usual size, SA >= its 90th percentile over the same name-days as S*.
    books["T-SA"] = (alive & (SArel >= 1.0), side, None)
    # T-BIG: T's events that are also big-move days, the like-for-like row against C-big and
    # P-low (T alone is mostly high k/ADV names on ordinary moves). A diagnostic, as in the
    # prior card; added before the full run.
    books["T-BIG"] = (alive & (Srel >= 1.0) & big, side, None)
    # ADDED AFTER THE INTERIM READ OF 2016 TO MID-2019 (2026-11-24, 21:30 ET), not in the card's
    # rule: is T's edge the flow or the names it picks? C-mom-Tnames is plain momentum on the
    # underlyings that ever carry a T event (chosen on S alone), on every eligible day.
    tn = books["T"][0].any()
    books["C-mom-Tnames"] = (E & pd.DataFrame(np.repeat(tn.to_numpy()[None, :], len(E), axis=0), index=E.index, columns=E.columns), side, None)
    if cols_placebo:
        pn = big.copy(); pn.loc[:, [c for c in pn.columns if c not in cols_placebo]] = False
        books["P-none (placebo stocks)"] = (pn, side, None)
    else:
        books["P-none (no fund alive)"] = (big & (B["K"] == 0), side, None)
    out, pnl1, pnl2 = {}, {}, {}
    y1 = side * B["R1"]
    y2 = -side * B["R2"]
    for name, (mask, sd, scale) in books.items():
        W = weights(mask, sd, scale)
        p1, m1 = score(W, B, 1)
        p2, m2 = score(-W, B, 2)
        pnl1[name], pnl2[name] = p1, p2
        sdm = sd.where(mask)
        out[name] = {"leg1": table(p1, regimes), "leg2": table(p2, regimes), "leg1_meta": m1, "leg2_meta": m2,
                     "per_event_leg1": events(sdm * B["R1"], mask, B["HS"]),
                     "per_event_leg2": events(-sdm * B["R2"], mask),
                     # DIAGNOSTIC, NOT PRE-SPECIFIED HERE: the prior review's lead (a fade from t+1's
                     # opening cross to its closing cross), on the same name-days, to place the give-back
                     "per_event_t1_open_to_close_fade_diagnostic": events(-sdm * (B["C"].shift(-1) / B["O"].shift(-1) - 1.0), mask),
                     "per_event_leg1_less_spy_same_window": events(sdm * B["R1"].sub(spy_win["R1"], axis=0), mask & (B["R1"].columns != "SPY")),
                     "both_legs_same_day": row(p1 + p2.reindex(p1.index).fillna(0.0))}
    # does the size add? T on C-big, and T-cont on C-mom (daily P&L, HAC errors)
    import statsmodels.api as sm
    reg = {}
    for a, b in (("T", "C-big"), ("T-cont", "C-mom"), ("T", "C-mom"), ("T-BIG", "C-big"), ("T-BIG", "P-low"), ("T", "C-mom-Tnames")):
        X = sm.add_constant(pnl1[b].rename(b))
        f = sm.OLS(pnl1[a], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        # the lag check (review, 2026-11-25): inference.newey_west_t's own rule, 4(T/100)^(2/9)
        L = int(4 * (len(X) / 100) ** (2 / 9))
        f2 = sm.OLS(pnl1[a], X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
        reg[f"{a} on {b}"] = {"alpha_bp_per_day": float(f.params["const"] * 1e4), "alpha_t": float(f.tvalues["const"]),
                              "alpha_t_at_nw_rule_lags": float(f2.tvalues["const"]), "nw_rule_lags": L,
                              "beta": float(f.params[b]), "beta_t": float(f.tvalues[b]), "r2": float(f.rsquared),
                              "alpha_sharpe_annual": float(f.resid.add(f.params["const"]).mean() / f.resid.std() * np.sqrt(252))}
    out["book_regressions_leg1"] = reg
    # LEG 2 TAKEN APART (review, 2026-11-25): the move r is the overnight gap to t's open (for a
    # foreign-market ETF, its home market's session) plus the US session to the signal trade.
    # Which part does the overnight fade reverse?
    Tm = books["T"][0]
    dec = {"T events": int(Tm.sum().sum())}
    for lab, part in (("gap", B["gap"]), ("us", B["us"])):
        dec[f"T events whose {lab} part has r's sign"] = int((Tm & (np.sign(part) == side)).sum().sum())
        for setname, m in (("T events", Tm), ("every eligible day", E)):
            mm = m & part.notna() & (part != 0)
            dec[f"fade the {lab} part from the close to the next open, {setname}"] = events((-np.sign(part) * B["R2"]).where(mm), mm)
    out["leg2_gap_us_decomposition"] = dec
    # ADDED AFTER THE NEIGHBOURS WERE READ (2026-11-25, 01:15 ET): the neighbours sit on even-
    # numbered sessions, where the headline itself scored far better than on all sessions. Is it
    # the sessions or the fetch? Sharpe by session number modulo 2 (the neighbours' split) and
    # modulo 3 (the three processes that fetched the headline entry).
    if SESSION_NO:
        jn = np.array([SESSION_NO[d] for d in B["sessions"]])
        sr_ = lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else None  # noqa: E731
        out["session_number_split_leg1"] = {n: {"mod2": [sr_(pnl1[n][jn % 2 == k]) for k in range(2)],
                                                "mod3": [sr_(pnl1[n][jn % 3 == k]) for k in range(3)]}
                                            for n in ("T", "C-mom", "C-big", "C-mom-Tnames")}
    # pooled name-day regression of the signed tau-to-close return on |r| and S (date clusters)
    panel = pd.DataFrame({"y": (y1 * 1e4).where(E).stack(), "absr": (r.abs() * 100).where(E).stack(),
                          "S": (S * 100).where(E).stack(), "SA": (B["SA"] * 100).where(E).stack(),
                          "y2": (y2 * 1e4).where(E).stack()}).dropna(subset=["y", "absr", "S"])
    panel["date"] = panel.index.get_level_values(0)
    pr = {}
    for lab, xs, yv in (("y1 ~ |r| + S", ["absr", "S"], "y"), ("y1 ~ |r|", ["absr"], "y"), ("y1 ~ S", ["S"], "y"),
                        ("y1 ~ |r| + SA", ["absr", "SA"], "y"), ("y2 ~ |r| + S", ["absr", "S"], "y2")):
        d = panel.dropna(subset=xs + [yv])
        cd = d["date"].factorize()[0]
        f = sm.OLS(d[yv], sm.add_constant(d[xs])).fit(cov_type="cluster", cov_kwds={"groups": cd})
        pr[lab] = {"n": int(len(d)), "dates": int(cd.max() + 1), **{f"{x}_coef": float(f.params[x]) for x in ["const"] + xs},
                   **{f"{x}_t": float(f.tvalues[x]) for x in ["const"] + xs}}
    # ADDED AFTER THE INTERIM READ: the same model within each underlying (its own mean of y,
    # |r| and S removed), so a name with strong momentum and many funds cannot stand in for flow
    d = panel.dropna(subset=["absr", "S", "y"]).copy()
    nm = d.index.get_level_values(1)
    for c_ in ("y", "absr", "S"):
        d[c_] = d[c_] - d.groupby(nm)[c_].transform("mean")
    cd = d["date"].factorize()[0]
    fe = sm.OLS(d["y"], d[["absr", "S"]]).fit(cov_type="cluster", cov_kwds={"groups": cd})
    pr["y1 ~ |r| + S, within underlying (added after the interim read)"] = {"n": int(len(d)), "dates": int(cd.max() + 1),
        **{f"{x}_coef": float(fe.params[x]) for x in ("absr", "S")}, **{f"{x}_t": float(fe.tvalues[x]) for x in ("absr", "S")}}
    out["panel_regressions"] = {"units": "y in bp; |r|, S and SA in percent", **pr}
    # dose: big-move days with a fund alive, by S and by SA
    dz = {}
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]                       # in units of that day's S*
    bb = big & (B["K"] > 0)
    for i in range(len(edges) - 1):
        m = bb & (Srel >= edges[i]) & (Srel < edges[i + 1])
        dz[f"S/S* {edges[i]:g} to {edges[i + 1]:g}"] = {"leg1": events(y1, m), "leg2": events(y2, m)}
    sa = B["SA"].where(bb).stack()
    if len(sa) > 50:
        qs = np.quantile(sa, [0, .2, .4, .6, .8, 1.0])
        for i in range(5):
            m = bb & (B["SA"] >= qs[i]) & ((B["SA"] < qs[i + 1]) if i < 4 else (B["SA"] <= qs[i + 1]))
            dz[f"SA quintile {i + 1} ({qs[i]:.4f} to {qs[i + 1]:.4f})"] = {"leg1": events(y1, m), "leg2": events(y2, m)}
    out["dose_big_move_days_with_a_fund"] = dz
    # concentration: T's events by underlying, and T without its three busiest underlyings
    Tm = books["T"][0]
    n_by = Tm.sum().sort_values(ascending=False)
    top3 = list(n_by.index[:3])
    Tx = Tm.copy(); Tx[top3] = False
    px_, _ = score(weights(Tx, side), B, 1)
    per_name = {c: {"events": int(n_by[c]), "leg1": events(y1[[c]], Tm[[c]]), "leg2": events(y2[[c]], Tm[[c]])} for c in n_by.index if n_by[c] > 0}
    out["T_concentration"] = {"top3": top3, "top3_share_of_events": float(n_by.iloc[:3].sum() / max(n_by.sum(), 1)),
                              "without_top3_leg1": row(px_), "without_top3_per_event_leg1": events(y1, Tx),
                              "share_of_names_positive_leg1": float(np.mean([v["leg1"].get("mean_bp", 0) > 0 for v in per_name.values() if v["events"] >= 30])) if any(v["events"] >= 30 for v in per_name.values()) else None,
                              "by_underlying": per_name}
    # by year, T
    out["T_by_year"] = {int(yr): {"leg1": events(y1.loc[str(yr)], Tm.loc[str(yr)], B["HS"].loc[str(yr)]), "leg2": events(y2.loc[str(yr)], Tm.loc[str(yr)])}
                        for yr in sorted(set(B["sessions"].year))}
    # neighbours: T, C-big, P-low at the other entry minutes
    nb = {}
    if Bn:
        # the headline entry on the neighbours' sessions, like for like
        on = pd.Series(B["sessions"].isin(next(iter(nb_sessions.values()))), index=B["sessions"])
        for name, mask in (("T", alive & (Srel >= 1.0)), ("C-big", big), ("P-low", big & (Srel < 0.25))):
            mk = mask & pd.DataFrame(np.repeat(on.to_numpy()[:, None], mask.shape[1], axis=1), index=mask.index, columns=mask.columns)
            p1, _ = score(weights(mk, side), B, 1)
            nb[f"entry close-30m {name}, on the neighbours' sessions"] = {"leg1": row(p1.loc[on[on].index]), "per_event_leg1": events(side.where(mk) * B["R1"], mk, B["HS"])}
    for k, Bk in Bn.items():
        sk, Ek = np.sign(Bk["r"]), Bk["E"]
        Sk = Bk["S"].div(th["S"], axis=0)                                # the headline's S*, as before
        for name, mask in (("T", Ek & (Bk["K"] > 0) & (Sk >= 1.0)), ("C-big", Bk["big"]), ("P-low", Bk["big"] & (Sk < 0.25))):
            p1, _ = score(weights(mask, sk), Bk, 1)
            onk = Bk["sessions"].isin(nb_sessions[k])
            nb[f"entry close-{k}m {name}"] = {"leg1": row(p1.loc[onk]), "per_event_leg1": events(sk.where(mask) * Bk["R1"], mask, Bk["HS"])}
    out["neighbours"] = nb
    # capacity, the headline book
    W = weights(books["T"][0], side)
    cap = {}
    for capital in CAPITAL:
        pos = (W.abs() * capital).where(W != 0)
        a = (pos / B["adv"]).stack()
        x = (pos / B["xadv"]).stack()
        both = (2 * pos / B["xadv"]).stack()
        cap[f"${capital / 1e6:.0f}M"] = {"positions": int(len(a)), "median_pct_adv": float(a.median() * 100), "max_pct_adv": float(a.max() * 100),
                                         "share_above_1pct_adv": float((a > 0.01).mean()), "median_pct_close_cross": float(x.median() * 100),
                                         "max_pct_close_cross": float(x.max() * 100), "share_above_5pct_cross": float((x > 0.05).mean()),
                                         "both_legs_one_close_order_median_pct_cross": float(both.median() * 100),
                                         "both_legs_share_above_5pct_cross": float((both > 0.05).mean())}
    out["capacity_T"] = cap
    out["thresholds"] = {k: {"mode": THRESHOLD_MODE, "first_session": str(v.first_valid_index().date()) if v.notna().any() else None,
                             "by_year_mean": {int(y): round(float(x), 5) for y, x in v.groupby(v.index.year).mean().dropna().items()}}
                         for k, v in th.items()}
    out["T_events"] = int(books["T"][0].sum().sum())
    out["T_underlyings"] = int(books["T"][0].any().sum())
    return out, pnl1, pnl2


def regimes_from_spy(cal: pd.DataFrame, DLY: dict) -> dict:
    spy = DLY["all_close"]["SPY"].reindex(cal.index)
    if spy.loc["2016-02-01":].isna().any():
        raise ValueError("SPY has no adjusted daily close on a session")
    ret = spy.pct_change()
    rv = ret.rolling(21).std().shift(1) * np.sqrt(252)            # to t-1
    ma = spy.rolling(200).mean().shift(1)
    return {"stressed (SPY 21d vol >= 20%)": (rv >= 0.20).where(rv.notna()),
            "uptrend (SPY above 200d average)": (spy.shift(1) > ma).where(ma.notna())}


def exposures(pnl: pd.Series, DLY: dict, held: dict, spy_win: pd.Series) -> dict:
    import statsmodels.api as sm
    out = {}
    X = pd.DataFrame({s: DLY["all_close"][s].pct_change() for s in ("SPY", "IEF", "HYG")}).reindex(pnl.index)
    d = pd.concat([pnl.rename("p"), X], axis=1).dropna()
    f = sm.OLS(d["p"], sm.add_constant(d[["SPY", "IEF", "HYG"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out["daily_betas"] = {k: {"beta": float(f.params[k]), "t": float(f.tvalues[k])} for k in ("SPY", "IEF", "HYG")}
    out["daily_betas"]["r2"] = float(f.rsquared)
    w = pd.concat([pnl.rename("p"), spy_win.rename("spy")], axis=1).dropna()
    w = w[w["p"] != 0]
    g = sm.OLS(w["p"], sm.add_constant(w["spy"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out["beta_to_spy_same_window_active_days"] = {"beta": float(g.params["spy"]), "t": float(g.tvalues["spy"]), "r2": float(g.rsquared)}
    for name, s in held.items():
        j = pd.concat([pnl.rename("p"), s.rename("h")], axis=1).dropna()
        out[f"corr_with_{name}"] = {"corr": float(j["p"].corr(j["h"])), "days": int(len(j)), "last": str(j.index.max().date())}
    return out


def held_sleeves() -> dict:
    """The core book's daily backtest P&L, and C4's z book (not its level blend), on the desk's
    panels (main repository data/, read only)."""
    from scripts.cef import band_frontier as bf
    # The panels are named explicitly: a moved script must not guess between two data/ folders
    # (it did, before the move to the private repository on 2026-11-24). Read only (landmine 9).
    if not os.environ.get("QUANTT_CEF_DATA"):
        raise SystemExit("set QUANTT_CEF_DATA to the directory holding cef/ and cef_families/ "
                         "(a CreditTrading checkout's data/, read only)")
    data = Path(os.environ["QUANTT_CEF_DATA"])
    out = {}
    src = (data / "cef" / "cef_prices.parquet", data / "cef" / "cef_nav.parquet")
    T, R = bf.build_targets(source=src)
    out["core_book_backtest"] = harness.evaluate(bf.band(T, bf.BAND_WIDTH), R)["pnl"]
    try:
        from scripts.cef import c4_families as c4
        spec = json.loads((REPO / "ops" / "specs" / "c4_families.frozen.json").read_text())["frozen"]
        fam = data / "cef_families"
        oldd, olds = c4.FAM_DIR, c4.FAM_SRC
        c4.FAM_DIR, c4.FAM_SRC = fam, (fam / "cef_prices.parquet", fam / "cef_nav.parquet")
        try:
            have = set(bf.build_panel(universe=spec["universe"], source=c4.FAM_SRC)["px"].columns)
            U = [t for t in spec["universe"] if t in have]
            T4, R4 = bf.build_targets(universe=U, source=c4.FAM_SRC, mask=c4.declared_mask(U, "M1M2"))
        finally:
            c4.FAM_DIR, c4.FAM_SRC = oldd, olds
        out["c4_z_book_backtest"] = harness.evaluate(bf.band(T4, bf.BAND_WIDTH), R4)["pnl"]
    except Exception as e:        # reported in results.json, never substituted
        out["c4_error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def assets_cross_check() -> dict:
    """CRSP's fund assets against ProShares' own, where both exist (2016-2025): how far apart,
    and how often each one's share count changes (CRSP updates most NYSE Arca funds monthly)."""
    e = pd.read_parquet(PRIOR / "etf_daily.parquet", columns=["permno", "ticker", "date", "prc", "shrout"])
    e["date"] = pd.to_datetime(e["date"])
    rows = []
    for fn in sorted(glob.glob(str(CACHE / "proshares" / "*.csv"))):
        x = pd.read_csv(fn)
        x["date"] = pd.to_datetime(x["Date"], format="%m/%d/%Y")
        x["ticker"] = os.path.basename(fn)[:-4]
        rows.append(x[["date", "ticker", "Assets Under Management", "Shares Outstanding (000)"]])
    P = pd.concat(rows).rename(columns={"Assets Under Management": "aum_ps", "Shares Outstanding (000)": "sh_ps"})
    m = P.merge(e, on=["ticker", "date"])
    m = m[(m["date"] >= INDEX_FIRST) & (m["date"] <= CRSP_LAST)].sort_values(["permno", "date"])
    rel = (m["shrout"] * 1000 * m["prc"].abs() / m["aum_ps"] - 1).abs()
    return {"fund_days": int(len(m)), "funds": int(m["permno"].nunique()), "median_abs_gap": float(rel.median()),
            "p90_abs_gap": float(rel.quantile(0.9)), "p99_abs_gap": float(rel.quantile(0.99)),
            "share_of_days_crsp_shares_change": float(m.groupby("permno")["shrout"].diff().ne(0).mean()),
            "share_of_days_proshares_shares_change": float(m.groupby("permno")["sh_ps"].diff().ne(0).mean())}


def thresholds_for(H: dict) -> dict:
    """The point-in-time S*, SA* and S3* over a sample's own pool (eligible name-days with a fund alive)."""
    on, on3 = H["E"] & (H["K"] > 0), H["E3"] & (H["K"] > 0)
    return {"S": pit_threshold(H["S"].where(on), THRESHOLD_MODE), "SA": pit_threshold(H["SA"].where(on), THRESHOLD_MODE),
            "S3": pit_threshold(H["S3"].where(on3), THRESHOLD_MODE)}


def slice_B(Bd: dict, start: pd.Timestamp) -> dict:
    """A panel dict from `start` on (frames and series by row; `sessions` likewise)."""
    out = {}
    for k, v in Bd.items():
        if k == "sessions":
            out[k] = v[v >= start]
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            out[k] = v.loc[start:]
        else:
            out[k] = v
    return out


def leg3_pnl(H: dict, mask: pd.DataFrame) -> pd.Series:
    side3 = -np.sign(H["r_full"])
    return window_pnl(weights(mask, side3), H["R3"], H, info=pd.DataFrame({c: H["close_ts"] for c in mask.columns}),
                      decided=H["close_ts"] + pd.Timedelta(minutes=15), entered=H["next_open_ts"], exited=H["next_close_ts"])


def threshold_comparison(H: dict, th: dict) -> dict:
    """T (leg 1) and T3 three ways: the full-sample threshold the first run used (a threshold read
    from the future), the expanding one (the rule since the review) and a trailing-252 one, each
    from the first session its point-in-time threshold exists (review, 2026-11-25)."""
    on, on3 = H["E"] & (H["K"] > 0), H["E3"] & (H["K"] > 0)
    sess = H["sessions"]
    pools = {"S": H["S"].where(on), "S3": H["S3"].where(on3)}
    modes = {"full-sample (the first run)": {k: pd.Series(np.quantile(v.loc[:CRSP_LAST].stack(), Q_EVENT), index=sess) for k, v in pools.items()},
             "expanding (the rule)": {k: th[k] for k in pools},
             "trailing-252": {k: pit_threshold(v, "trailing252") for k, v in pools.items()}}
    out = {}
    for key, pool_on, pnl_fn in (("S", on, lambda m: score(weights(m, np.sign(H["r"])), H, 1)[0]), ("S3", on3, lambda m: leg3_pnl(H, m))):
        start = max(th[key].first_valid_index(), modes["trailing-252"][key].first_valid_index())
        after = pd.DataFrame(np.repeat((sess >= start)[:, None], len(H["E"].columns), axis=1), index=sess, columns=H["E"].columns)
        rows = {}
        for lab, t_ in modes.items():
            m = pool_on & H[key].div(t_[key], axis=0).ge(1.0) & after
            p = pnl_fn(m).loc[start:]
            sr_ = lambda x: float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else None  # noqa: E731
            rows[lab] = {"events": int(m.sum().sum()), "sharpe": sr_(p), "nw_t": float(inference.newey_west_t(p.to_numpy())),
                         "to_2022": sr_(p.loc[:"2022-12-31"]), "2023_on": sr_(p.loc["2023-01-01":])}
            # the same book against plain momentum (leg 1) or plain reversal (leg 3) on its own names,
            # every eligible day from `start`, HAC alpha t at 5 lags and at the Newey-West rule's lags
            import statsmodels.api as sm
            elig = (H["E"] if key == "S" else H["E3"]) & after
            ctl = elig & pd.DataFrame(np.repeat(m.any().to_numpy()[None, :], len(sess), axis=0), index=sess, columns=m.columns)
            pc = pnl_fn(ctl).loc[start:]
            Xc = sm.add_constant(pc.rename("ctl"))
            L = int(4 * (len(Xc) / 100) ** (2 / 9))
            rows[lab]["alpha_t_on_same_names_5_lags"] = float(sm.OLS(p, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 5}).tvalues["const"])
            rows[lab]["alpha_t_on_same_names_rule_lags"] = float(sm.OLS(p, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": L}).tvalues["const"])
            rows[lab]["same_names_control_sharpe"] = sr_(pc)
        out["T, leg 1" if key == "S" else "T3, leg 3"] = {"from": str(start.date()), **rows}
    return out


def g_t_check(cal, DLY, PR, IN, K, syms) -> dict:
    """The signal r carries t's distribution factor g_t from CRSP, which CRSP computes from t's
    end-of-day price. The dividend and the split ratio are declared before the ex-date, so the
    point-in-time form uses them without t's price: g = 1 + D_t / C_t-1, with D_t the declared
    amount per share (backed out of CRSP as (1 + ret) x prc_t-1 - prc_t) and C_t-1 the prior
    closing cross; on a split day (|g - 1| > 20%) the split ratio, i.e. CRSP's g, is kept. A
    bound is shown too: no adjustment at all (g = 1). T is rebuilt under each and compared with
    the run (review, 2026-11-25)."""
    e = pd.read_parquet(PRIOR / "etf_daily.parquet", columns=["permno", "date", "ret", "prc"])
    S_ = pd.read_parquet(CACHE / "symbols.parquet")
    ids = S_.loc[S_["group"] == "index", ["permno", "symbol"]]
    e = e[e["permno"].isin(ids["permno"])].copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"])
    e["prc"] = e["prc"].abs()
    e["D"] = (1.0 + e["ret"]) * e.groupby("permno")["prc"].shift(1) - e["prc"]
    D = e.merge(ids, on="permno").pivot(index="date", columns="symbol", values="D").reindex(index=cal.index, columns=syms)
    g = DLY["g"].reindex(index=cal.index, columns=syms)
    Cprev = PR["close_px"].reindex(index=cal.index, columns=syms).shift(1)
    g_decl = (1.0 + D / Cprev).where((g - 1.0).abs() <= 0.20, g)
    variants = {"as run": DLY["g"],
                "declared dividend over t-1's close (point in time)": DLY["g"].copy(),
                "no adjustment (g = 1, a bound)": DLY["g"].where(DLY["g"].isna(), 1.0)}
    variants["declared dividend over t-1's close (point in time)"].loc[g_decl.index, syms] = g_decl.where(g.notna())
    Hs = {lab: build("_gcheck", cal, dict(DLY, g=gv), PR, IN, K, syms, INDEX_FIRST, CRSP_LAST, HEAD) for lab, gv in variants.items()}
    out = {}
    for mode in ("full-sample (the first run)", "expanding (the rule)"):
        masks, srs = {}, {}
        for lab, H in Hs.items():
            on = H["E"] & (H["K"] > 0)
            S = H["S"].where(on)
            t_ = (pd.Series(np.quantile(S.loc[:CRSP_LAST].stack(), Q_EVENT), index=H["sessions"]) if mode.startswith("full")
                  else pit_threshold(S, THRESHOLD_MODE))
            masks[lab] = on & H["S"].div(t_, axis=0).ge(1.0)
            p, _ = score(weights(masks[lab], np.sign(H["r"])), H, 1)
            srs[lab] = float(p.mean() / p.std() * np.sqrt(252))
        base = masks["as run"]
        out[mode] = {"T entries as run": int(base.sum().sum()), "sharpe as run": srs["as run"],
                     **{lab: {"entries that differ from the run": int((base ^ masks[lab]).sum().sum()), "sharpe": srs[lab]}
                        for lab in masks if lab != "as run"}}
    return out


def check_truncation(group, cal, DLY, PR, IN, K, syms, first, last) -> dict:
    """harness.truncation_check on the headline book's weights: every input frame is cut at
    the probe date before `build` runs, the point-in-time threshold is recomputed on the cut
    data, and the probe rows must not move."""
    def cut(F, d):
        return F if d is None else F.loc[:d]

    def build_(d):
        cal_ = cal if d is None else cal.loc[:d]
        DLY_ = {k: cut(v, d) for k, v in DLY.items()}
        PR_ = {k: cut(v, d) for k, v in PR.items()}
        IN_ = {e: {k: cut(v, d) for k, v in IN[e].items()} for e in (HEAD,)}
        B = build("_trunc", cal_, DLY_, PR_, IN_, cut(K, d), syms, first, last if d is None else min(last, d), HEAD)
        on = B["E"] & (B["K"] > 0)
        thS = pit_threshold(B["S"].where(on), THRESHOLD_MODE)
        mask = on & (B["S"].div(thS, axis=0) >= 1.0)
        return weights(mask, np.sign(B["r"]))
    sess = cal.index[(cal.index >= first) & (cal.index <= last)]
    probes = [sess[i] for i in np.linspace(len(sess) // 3, len(sess) - 2, 4).astype(int)]
    return harness.truncation_check(build_, probes)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline-only", action="store_true", help="skip the neighbour entry minutes (while they are still being fetched)")
    ap.add_argument("--through", default=None, help="development only: stop at this session (the index group alone if before 2022-07)")
    a = ap.parse_args(argv)
    use = (HEAD,) if a.headline_only else ENTRIES
    cal = calendar()
    if a.through:
        cal = cal.loc[:a.through]
    S_ = pd.read_parquet(CACHE / "symbols.parquet")
    syms_all = sorted(S_["symbol"])
    idx_syms = sorted(S_.loc[S_["group"] == "index", "symbol"])
    one_syms = sorted(S_.loc[S_["group"] != "index", "symbol"])
    placebo = sorted(S_.loc[S_["group"] == "single_placebo", "symbol"])
    sym_of = dict(zip(S_["permno"].astype(int), S_["symbol"]))
    sessions = cal.index
    SESSION_NO.update({d: j for j, d in enumerate(sessions)})
    DLY = daily(sessions, syms_all + ["IEF", "HYG"])
    DLY["g"], out_g = factor_ratio(sessions, S_, DLY)
    PR, pinfo = prints(sessions, syms_all)
    IN = intraday(sessions, syms_all)
    last_sess = sessions[-1]
    # The neighbours (close-60, close-15) were fetched on ALTERNATE sessions only (fetch.py
    # `--entries 60,15 --shard 0/2`, 2026-11-24 21:55 ET: the account's rate budget was shared
    # with another job and the full set would have taken past 03:00 ET), plus every session
    # fetched whole at the start. Decided before any neighbour return was read. The headline
    # entry is shown on the same sessions beside them (`neighbours`).
    full_files = {pd.Timestamp(p.stem) for p in (CACHE / "intraday").glob("*.json")}
    alternate = {d for j, d in enumerate(sessions) if j % 2 == 0} | full_files
    nb_sessions = {}
    for k in use:
        miss = IN["missing_sessions"][k]
        if k != HEAD:
            miss = [d for d in miss if d in alternate]
            nb_sessions[k] = pd.DatetimeIndex(sorted(alternate))
        if miss:
            raise ValueError(f"entry close-{k}m: {len(miss)} sessions up to {last_sess.date()} have no intraday file, "
                             f"first {miss[0].date()}; finish the fetch (or use --through for a development run)")
    pairs = pd.read_parquet(PRIOR / "pairs.parquet")
    p_idx = pairs[(pairs["kind"] == "index") & pairs["lev"].isin(INDEX_MULTIPLES)]
    p_one = pairs[pairs["kind"] == "single"]
    A_crsp = fund_assets_crsp()
    A_crsp = A_crsp[A_crsp["date"] >= "2015-10-01"]
    full_cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(CACHE / "calendar.parquet")["date"]))
    A_ps = fund_assets_proshares(full_cal)
    A_ps = A_ps[A_ps["date"] >= "2015-10-01"]
    ps_funds = set(A_ps["fund"])
    K = {"index_crsp": k_frame(A_crsp, p_idx, sessions, sym_of), "single_crsp": k_frame(A_crsp, p_one, sessions, sym_of),
         "index_proshares": k_frame(A_ps, p_idx, sessions, sym_of, year_of_pairs={2026: 2025})}
    # V5: ProShares' own assets for ProShares funds, CRSP for the rest (2016-2025)
    mix = pd.concat([A_crsp[~A_crsp["fund"].isin(ps_funds)], A_ps[A_ps["date"] <= CRSP_LAST]])
    K["index_mixed"] = k_frame(mix, p_idx, sessions, sym_of)
    for k_ in K:                       # k_frame already holds every symbol: 0 where no fund is alive
        if list(K[k_].columns) != syms_all:
            raise ValueError(f"{k_}: k frame columns are not the symbol list")
    # ProShares' share of each index underlying's k in 2025 (CRSP), for reading the 2026 rows
    yr_ = "2025" if (sessions.year == 2025).any() else str(sessions[-1].year)   # 2025, except in a --through run
    kc25 = K["index_crsp"].loc[yr_]
    kp25 = k_frame(A_crsp[A_crsp["fund"].isin(ps_funds)], p_idx, sessions, sym_of).reindex(columns=syms_all).fillna(0.0).loc[yr_]
    tot = kc25[idx_syms].sum()
    share_ps = (kp25[idx_syms].sum() / tot.where(tot > 0)).dropna()
    out = {"entries_scored": list(use), "distribution_factor_check": out_g, "written": "2026-11-24", "panel_last_session": str(last_sess.date()), "crsp_last": str(CRSP_LAST.date()),
           "prints": pinfo, "proshares_share_of_2025_k_by_underlying": {k: round(float(v), 3) for k, v in share_ps.sort_values(ascending=False).items()},
           "proshares_share_of_2025_k_all_index": float(kp25[idx_syms].sum().sum() / kc25[idx_syms].sum().sum())}
    out["assets_crsp_vs_proshares"] = assets_cross_check()
    # leveraged assets and k by year, mean over sessions (CRSP, the mapped funds of each group)
    aby = {}
    for lab, pp, Kf in (("index", p_idx, K["index_crsp"]), ("single", p_one, K["single_crsp"])):
        a_ = A_crsp.assign(year=A_crsp["date"].dt.year).merge(pp[["fund", "year"]], on=["fund", "year"])
        a_ = a_[(a_["date"] >= INDEX_FIRST) & (a_["date"] <= CRSP_LAST)]
        tot = a_.groupby("date")["aum_lag"].sum()
        ks = Kf.loc[INDEX_FIRST:CRSP_LAST].sum(axis=1, min_count=1)
        aby[lab] = {int(y): {"mean_assets_usd_bn": round(float(tot[tot.index.year == y].mean()) / 1e9, 2),
                             "mean_k_usd_bn": round(float(ks[ks.index.year == y].mean()) / 1e9, 2)} for y in sorted(set(tot.index.year))}
    out["leveraged_assets_by_year"] = aby
    reg = regimes_from_spy(cal, DLY)
    held = held_sleeves()
    out["held_sleeves"] = {k: (str(v.index.max().date()) if isinstance(v, pd.Series) else v) for k, v in held.items()}
    held = {k: v for k, v in held.items() if isinstance(v, pd.Series)}
    pnl_store = {}

    def one(label, group, syms, Kf, first, last, cols_placebo, th=None, score_from=None, compare=False):
        """One sample. `th` (point-in-time thresholds) is computed here from the sample's own pool
        unless given (the subgroups take the index group's). `score_from` builds the whole history
        first, so the thresholds see every session before the scored ones, then scores only
        from that session on (the 2026 run)."""
        # build's first argument names the exclusion counters: the sample's label, so that the seven
        # index-family samples are not summed into one figure (they were until the review, 2026-11-25)
        B = {e: build(label, cal, DLY, PR, IN, Kf, syms, first, last, e, big_rule="rows252" if e == HEAD else "eligible126")
             for e in use}
        th = thresholds_for(B[HEAD]) if th is None else th
        th = {k_: v.reindex(B[HEAD]["sessions"]) for k_, v in th.items()}
        cmp_ = threshold_comparison(B[HEAD], th) if compare else None
        spyB = build("_spy", cal, DLY, PR, IN, Kf, ["SPY"], first, last, HEAD)
        # EVERY BOOK STARTS WHERE THE RULE CAN FIRST BE APPLIED (review, 2026-11-25): the session
        # the point-in-time S* first exists. Controls scored from an earlier start would be
        # compared with a T that could not trade yet.
        start = th["S"].first_valid_index()
        if start is None:
            raise ValueError(f"{label}: the point-in-time threshold never exists ({MIN_PRIOR_OBS} prior name-days)")
        start = max(start, score_from) if score_from is not None else start
        B = {e: slice_B(b, start) for e, b in B.items()}
        th = {k_: v.loc[start:] for k_, v in th.items()}
        spyB = slice_B(spyB, start)
        H = B[HEAD]
        # SPY over the same windows, each dated by the session it ends on, for the in-window betas
        spy1, spy2, spy3 = spyB["R1"]["SPY"], spyB["R2"]["SPY"].shift(1), spyB["R3"]["SPY"].shift(1)
        res, p1, p2 = run_group(label, H, {e: B[e] for e in use if e != HEAD}, th, cols_placebo, reg, {"R1": spy1},
                                nb_sessions=nb_sessions)
        res["exposures_T_leg1"] = exposures(p1["T"], DLY, held, spy1)
        res["exposures_T_leg2"] = exposures(p2["T"], DLY, held, spy2)
        res["exposures_C-big_leg1"] = exposures(p1["C-big"], DLY, held, spy1)
        res["leg3_next_session_fade_added_after_interim_read"], p3 = run_leg3(H, th["S3"], reg)
        res["exposures_T3"] = exposures(p3["T3"], DLY, held, spy3)
        if cmp_ is not None:
            res["threshold_comparison"] = cmp_
        res["first"], res["last"] = str(H["sessions"][0].date()), str(H["sessions"][-1].date())
        out[label] = res
        pnl_store[label] = (p1, p2, p3)
        return th

    end = min(CRSP_LAST, cal.index[-1])
    if a.through:
        one("index", "index", idx_syms, K["index_crsp"], INDEX_FIRST, end, [], compare=True)
        out["truncation_check_index_T"] = check_truncation("index", cal, DLY, PR, IN, K["index_crsp"], idx_syms, INDEX_FIRST, end)
        if end >= pd.Timestamp("2022-09-01"):
            one("single", "single", one_syms, K["single_crsp"], SINGLE_FIRST, end, placebo)
        out["exclusions"] = EXCL
        (HERE / "results_dev.json").write_text(json.dumps(out, indent=1, default=str))
        summary(out)
        return 0
    s_idx = one("index", "index", idx_syms, K["index_crsp"], INDEX_FIRST, CRSP_LAST, [], compare=True)
    one("index: sector only", "index", [s for s in idx_syms if s not in BROAD], K["index_crsp"], INDEX_FIRST, CRSP_LAST, [], s_idx)
    one("index: broad only (SPY, QQQ, DIA)", "index", list(BROAD), K["index_crsp"], INDEX_FIRST, CRSP_LAST, [], s_idx)
    one("index: foreign-market ETFs (V7)", "index", [s for s in idx_syms if s in FOREIGN], K["index_crsp"], INDEX_FIRST, CRSP_LAST, [], s_idx)
    one("index, ProShares assets for ProShares funds (V5)", "index", idx_syms, K["index_mixed"], INDEX_FIRST, CRSP_LAST, [])
    s_ps = one("index, ProShares-only assets 2016-2025 (V6)", "index", idx_syms, K["index_proshares"], INDEX_FIRST, CRSP_LAST, [])
    one("index, ProShares-only assets, 2026 (out of sample)", "index", idx_syms, K["index_proshares"], INDEX_FIRST, last_sess, [],
        score_from=pd.Timestamp("2026-01-02"))
    one("single", "single", one_syms, K["single_crsp"], SINGLE_FIRST, CRSP_LAST, placebo, compare=True)
    out["g_t_check_index_T"] = g_t_check(cal, DLY, PR, IN, K["index_crsp"], idx_syms)
    out["exclusions"] = {k_: v for k_, v in EXCL.items() if not k_.startswith("_")}
    out["truncation_check_index_T"] = check_truncation("index", cal, DLY, PR, IN, K["index_crsp"], idx_syms, INDEX_FIRST, CRSP_LAST)
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    chart(pnl_store)
    summary(out)
    return 0


def chart(store: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, lab in zip(axes, ("index", "single")):
        p1, p2, p3 = store[lab]
        for name in ("T", "C-mom-Tnames", "C-big", [k for k in p1 if k.startswith("P-none")][0]):
            (p1[name].cumsum() * 100).plot(ax=ax, label=f"{name}, tau to close", lw=1.6 if name == "T" else 1.0)
        (p2["T"].cumsum() * 100).plot(ax=ax, label="T, leg 2: close to next open", ls="--", lw=1.0)
        (p3["T3"].cumsum() * 100).plot(ax=ax, label="T3 (added after interim read): fade over t+1's session", ls=":", lw=1.2)
        ax.set_title(f"{lab}: gross, close - 30 min entry (calendar time, gross 1)", fontsize=9)
        ax.set_ylabel("cumulative return, % (not compounded)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "equity.png", dpi=110); plt.close(fig)


def summary(out: dict) -> None:
    for lab in [k for k in out if isinstance(out[k], dict) and "T" in out[k]]:
        o = out[lab]
        print(f"\n===== {lab}  {o['first']} .. {o['last']}  S* from {o['thresholds']['S']['first_session']}  T events {o['T_events']} in {o['T_underlyings']} names")
        for b in ("T", "T-BIG", "T-cont", "T-SA", "C-mom", "C-mom-Tnames", "C-big", "C-open", "P-low") + tuple(k for k in o if k.startswith("P-none")):
            f1, f2 = o[b]["leg1"]["full"], o[b]["leg2"]["full"]
            e1 = o[b]["per_event_leg1"]
            print(f"  {b:28s} leg1 SR {f1['sharpe'] or 0:5.2f} (nw t {f1['nw_t'] or 0:5.2f}) {f1['ann_ret']:7.2%}/{f1['ann_vol']:6.2%} | "
                  f"leg2 SR {f2['sharpe'] or 0:5.2f} (t {f2['nw_t'] or 0:5.2f}) | per event {e1.get('mean_bp', 0):6.2f}bp n {e1.get('events')} hs {e1.get('median_quoted_half_spread_bp')}")
        for k, v in o["book_regressions_leg1"].items():
            print(f"  {k}: alpha {v['alpha_bp_per_day']:.2f}bp/day t {v['alpha_t']:.2f} beta {v['beta']:.2f}")
        for k, v in o["panel_regressions"].items():
            if isinstance(v, dict):
                print(f"  {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in v.items() if a.endswith(('_coef', '_t'))))


if __name__ == "__main__":
    sys.exit(main())
