"""Closing-auction size and how well it can be forecast at the time an MOC cap is set; the
afternoon continuous-session pool; the price path of working an order early; and what a
per-order cap against the expected cross implies at $100k, $1M, $5M and $10M (measured
2026-11-24).

Report only. It changes nothing on the live path, sends nothing to a broker, and adopts
nothing. Note: `desk/results/cef/CLOSE_AUCTION_FORECAST_2026-11-24.md`; output of record
(aggregates only): `desk/results/cef/close_auction_forecast_2026-11-24/aggregates_2026-11-24.json`.
Both are in the private research repository (simonjarviss/quantt-research); every path below that
does not start with `desk/` is in CreditTrading, named on PYTHONPATH (`_credittrading_root`).

WHY IT EXISTS
-------------
`results/cef/EXEC_CAPACITY_OPEN_CROSS_2026-10-13.md` measured the NYSE closing cross in the 13
core names at a median of 3,703 shares (about $38k), 0.57% of the day's volume, and the book's
orders at a notional-weighted median of 31% of it; `results/cef/EXEC_COST_STUDY_2026-11-05.md`
re-measured 15.2% (p90 35.9%) and ranked "a per-order cap against the expected closing cross,
the excess worked elsewhere" second among its candidates, unbuilt. Paper runs no auction
(the desk's rules, landmine 1), so our own impact on the print is unmeasured (n = 0). This module builds
the measurement such a cap needs: how big the cross is, how well its size can be known before
the order must be in, how much the afternoon's continuous market could absorb instead, and
whether working an order earlier gains or loses against the close the backtest fills at.

THE CROSS (the 2026-10-13 rule, re-used, not re-derived)
-------------------------------------------------------
The closing cross is the primary listing's condition-`6` print ("Market Center Closing Trade",
updates volume); `M` ("Official Close") is a price message whose size is NOT the cross on a day
without a round-lot cross (EXEC_CAPACITY_OPEN_CROSS_2026-10-13.md §1; CTS spec, docs/REFERENCES.md
§12o). /v2/stocks/auctions lists at most one primary `6` per name-day on the core 13 in 2016-22.
So per name-day: one primary `6` = a round-lot cross of that size; a primary `M` and no `6` = no
round-lot cross (an odd lot or none; the primary's tape splits the two on the last
TAPE_SESSIONS sessions with `exec_capacity_open_cross.classify_cross`); two or more primary `6`
= "multi6", counted and never sized. A traded day with no primary print at all is a no-round-lot
day only BEFORE the listing venue's official-close (M) data begin; inside that era it is missing
data, counted and never sized (`cross_panel`). The primary is each name's listing venue today
(asset `exchange` -> SIP code, as quantt.broker.alpaca does), plus the historical code that carried
that venue's prints (`PRIMARY_ALIASES`: Nasdaq's crosses printed on code T); every name-day whose
round-lot cross printed on ANOTHER listing venue is counted and reported (a listing move), never
silently re-assigned. Corrected 2026-11-24 after an independent review (the first run, commit
b353e50, read Nasdaq listings on Q alone and scored missing close data as zero crosses).
Sizes are compared in DOLLARS (shares x the cross price): a dollar cross is unchanged by a split,
and every order is a dollar amount scaled to a book size. Splits are still detected (raw against
split-adjusted daily closes) and reported.

THE CAP TIME, AND THE ALIGNMENT EVERY FORECAST OBEYS
----------------------------------------------------
The book decides on the evening of t and sends at t+1 (shift(2), scripts/research/harness.py).
A live MOC must be at Alpaca before 15:50 ET ("CLS orders submitted after 3:50pm but before 7:00pm
ET will be rejected" [V: docs, 2026-11-05]; NYSE's freeze is ten minutes before the end of Core
Trading Hours, Rule 7.35, so it moves with an early close [V: docs/REFERENCES.md §3]). The cap is
set at CAP_LEAD_MIN = 15 minutes before the close (15:45 on a full day, 12:45 on a 13:00 close):
five minutes before the cut-off, to compute and send. A forecast of the cross on fill day D may
use, and only use:
  * crosses and daily bars dated strictly before D (complete by 16:00/20:00 of D-1);
  * D's 5-minute bars that END at or before the cap time (start + 5 min <= close - 15 min);
  * calendar facts about D known in advance: month and quarter end, the third Friday (monthly
    option expiry; in Mar/Jun/Sep/Dec also the index-rebalance and quad-witching day), the last
    Friday of June [U: the Russell reconstitution day is not verified here], a 13:00 close, the
    session before a 13:00 close, and the name's own ex-date [U: assumed declared before D;
    CEF distributions are declared ahead, not verified per event].
Every rolling feature is built from a series shifted one session, and `truncation_check`
recomputes a sample of rows on data cut at the cap time of D and raises on any difference.
The residual quantiles that turn a point forecast into a low quantile are re-estimated at the
start of each calendar month from residuals dated strictly before that month.
A forecast at 15:45 can size the MOC order, but not the part worked in the afternoon, which must be
fixed before working starts: the cap report therefore sizes the split with PRE-OPEN forecasts (ewm10,
and reg_cal, whose calendar flags are known in advance) and shows the 15:45 one beside them.

SIGNS AND UNITS
---------------
s = +1 buy, -1 sell (a short sale is a sell). P_c = the primary's cross price on D (the round-lot
`6` print; the official close `M` where no `6`). Path gain at reference x:
    G_x = s (P_c - x) / x x 1e4  bp
x = the NBBO mid at a time tau (quotes) or a window's VWAP (5-minute bars). G > 0: the close was
WORSE for our side than x, so working at x would have GAINED against the close, before any spread
or impact. These are signal-conditional: the sample is the book's own orders, whose direction the
discount signal chose.

SAMPLES (split by method, never pooled across a change of method)
---------------------------------------------------------------
  * the core book's paper orders, read from the cef account (GET /v2/orders): `cls` on
    2026-09-29 (13 orders, sent at 10:33 ET) and `late_market` from 2026-09-30 (the desk's rules order-path
    rule 3's paper exception). Order size is the ORDERED quantity (an expired `cls` order still
    says what a live auction order would have been). Rescaled to a book size by the account's
    end-of-day equity on the prior session (GET /v2/account/portfolio/history).
  * the deployed policy's orders in its own backtest (scripts.cef.band_frontier on the frozen spec,
    decide t, fill t+1; whole shares at the spec's capital), 2016 to the panel's last date. A MODEL
    of orders, labelled so; it gives the path statistic a decade instead of 40 sessions.
  * C4's orders are NOT read: this laptop's key file holds no C4 key and no copy of C4's records
    is here (the VM holds them; no ssh). C4's market side is measured in full.

INPUTS AND OUTPUTS
------------------
  fetch    GETs only: data.alpaca.markets /v2/stocks/{auctions,bars,quotes,trades} and
           /v2/stocks/meta/exchanges; paper-api /v2/calendar, /v2/orders (status=all) and
           /v2/account/portfolio/history on the cef account. Keys come from QUANTT_ENV_FILE through
           AlpacaClient; never printed. No order endpoint exists on that client's GET path.
  analyse  offline. --cache-dir (vendor data) and --private-dir (account rows, per-name and
           per-order tables) are refused anywhere git can see and under data/ (landmine 9).
           --aggregates is the JSON of record: counts, medians and shares; no account or order id,
           no per-order row, no equity figure; symbols appear only in the `market_by_name_core`
           block (public market data: cross sizes, pools and spreads of the 13 core names, as the
           2026-10-13 note published them).

Reproduce from the private repository's root (cache and private directories outside any repository):
  PYTHONPATH=<CreditTrading checkout> QUANTT_ENV_FILE=<config/.env> \\
      python3 desk/scripts/cef/close_auction_forecast_2026-11-24/close_auction_forecast.py \\
      fetch --cache-dir <cache> --private-dir <private>
  PYTHONPATH=<CreditTrading checkout> \\
      python3 desk/scripts/cef/close_auction_forecast_2026-11-24/close_auction_forecast.py analyse \\
      --cache-dir <cache> --private-dir <private> --cef-panels <dir holding cef/ and cef_families/> \\
      --aggregates desk/results/cef/close_auction_forecast_2026-11-24/aggregates_2026-11-24.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)        # landmine 11: wide frames below

def _credittrading_root() -> Path:
    """The CreditTrading checkout this script reads from: the C4 universe, `AlpacaClient`,
    `scripts/cef/exec_capacity_open_cross.py` and the shared `scripts/research/inference.py`.

    WHY: the script moved to this separate research repository on 2026-11-24,
    so it no longer sits inside
    that checkout and `parents[3]` no longer finds it. `desk/README.md` says to name the checkout
    on PYTHONPATH. There is no default: a guessed checkout could supply a stale C4 universe or a
    different estimator without saying so.
    """
    for entry in sys.path:
        p = Path(entry) if entry else None
        if p and (p / "scripts/research/harness.py").is_file() and (p / "quantt").is_dir():
            return p.resolve()
    raise SystemExit("set PYTHONPATH to a CreditTrading checkout (one holding "
                     "scripts/research/harness.py and quantt/): see desk/README.md")


REPO = _credittrading_root()

NY, UTC = ZoneInfo("America/New_York"), dt.timezone.utc
FIRST = dt.date(2016, 1, 4)               # Alpaca's SIP history starts in 2016 (fetch_alpaca_auction_history.py)
C4_UNIVERSE = REPO / "results/cef/c4/c4_universe_frozen_2026-10-14.csv"
REGIME_SYMBOL = "SPY"                     # market-vol regime label only (trailing realised vol)
DATA_PATHS = ("/v2/stocks/auctions", "/v2/stocks/bars", "/v2/stocks/quotes", "/v2/stocks/trades",
              "/v2/stocks/meta/exchanges")
REQ_SLEEP_S = 0.31                        # stay under the Basic plan's 200 data requests a minute
CAP_LEAD_MIN = 15                         # the cap is set 15 minutes before the close (module docstring)
CUTOFF_LEAD_MIN = 10                      # MOC cut-off: 10 minutes before the close (Rule 7.35; Alpaca 15:50)
# Afternoon windows, minutes before the close: [lo, hi). On a 16:00 close these are the task's
# 14:00-15:50, 15:00-15:50 and 15:30-15:50; on a 13:00 close they move with it.
WINDOWS = {"120_10": (120, 10), "60_10": (60, 10), "30_10": (30, 10)}
WINDOW_LABEL = {"120_10": "14:00-15:50", "60_10": "15:00-15:50", "30_10": "15:30-15:50"}
SPREAD_EXTRA = {"10_0": (10, 0)}          # the last ten minutes, for comparison only
QUOTE_LEAD_MIN = 125                      # quotes fetched from 125 minutes before the close (5-minute lead-in)
# Reference times for the path, seconds before the close (15:59:50 = 10 s)
TAU_S = {"14:00": 7200, "14:30": 5400, "15:00": 3600, "15:30": 1800, "15:40": 1200,
         "15:45": 900, "15:50": 600, "15:52": 480, "15:55": 300, "15:58": 120, "15:59:50": 10}
SPREAD_RECENT = 100                       # quotes: every one of the last 100 sessions ...
SPREAD_EVERY = 20                         # ... and every 20th session before them, back to FIRST
TAPE_SESSIONS = 250                       # primary tape read on no-6 name-days of the last 250 sessions
GAP_PRINT_SHARE = 0.20                    # a session on which fewer traded names have a primary print is a data gap
TRADECHECK_SESSIONS = 5                   # raw-trade check of the bars on the last 5 sessions, core names
BOOK_SIZES = (100_000, 1_000_000, 5_000_000, 10_000_000)
CAPS = (0.10, 0.20, 0.33)
PARTICIPATION = (0.05, 0.10)


class StudyError(RuntimeError):
    """The data do not have the shape this module's rule assumes, or something is missing."""


# =============================================================================================
# pure functions (tested in test_close_auction_forecast.py)
# =============================================================================================

def close_ts(d: dt.date, close_hhmm: str) -> pd.Timestamp:
    """The session's close on date d (Alpaca calendar wall-clock HH:MM, New York), as UTC."""
    h, m = map(int, close_hhmm.split(":"))
    return pd.Timestamp(dt.datetime.combine(d, dt.time(h, m), NY)).tz_convert("UTC")


def classify_close(rows: list[dict], primary: tuple, listing_codes=("N", "A", "Q")) -> dict:
    """One name-day's closing cross from its /v2/stocks/auctions closing prints (all venues).

    rows: dicts with x (SIP venue), c (condition), p, s, t. primary: the SIP codes that carry the
    name's listing venue's prints (`primary_codes`: one code, or for a Nasdaq listing ("Q", "T")).
    Returns {kind, shares, price, usd, t, other_listing_6, six_venues}:
      round       one primary `6` (or the same `6`, equal in size and price, on two of the name's
                  codes: one cross printed twice, counted once): its size is the whole round-lot cross;
      multi6      two `6` on one code, or two of the name's codes disagreeing: counted, never sized;
      no_round    no primary `6` but a primary `M`: no round-lot cross (an odd lot or none);
      no_primary  no primary print of either kind (the caller decides what that means: `cross_panel`).
    `price` is the `6` price for round, else the EARLIEST primary `M` (on 2026-11-20 SCD's primary
    sent two M prices, 16:00:02 and 16:10:00; quantt.broker.alpaca keeps the later one only when the
    daily bar confirms it, and this module sizes nothing on such a day). `other_listing_6` is True
    when another LISTING venue printed a `6` that day: a sign the listing moved. `six_venues` names
    the codes the `6` came from (so the T-only and the T-and-Q days can be counted)."""
    if isinstance(primary, str):
        raise TypeError("primary must be a tuple of SIP codes, not one string")
    prim = [r for r in rows if r["x"] in primary]
    six = [r for r in prim if r["c"] == "6"]
    em = sorted((r for r in prim if r["c"] == "M"), key=lambda r: pd.Timestamp(r["t"]))
    other6 = any(r["c"] == "6" and r["x"] not in primary and r["x"] in listing_codes for r in rows)
    venues = "".join(sorted({r["x"] for r in six}))
    by_venue: dict = {}
    for r in six:
        by_venue.setdefault(r["x"], []).append(r)
    bad = dict(kind="multi6", shares=np.nan, price=np.nan, usd=np.nan, t=None, other_listing_6=other6,
               six_venues=venues)
    if any(len(v) > 1 for v in by_venue.values()):
        return bad
    if len(by_venue) > 1:
        if len({(float(r["s"]), float(r["p"])) for r in six}) > 1:
            return bad
        six = [six[0]]
    if len(six) == 1:
        sh, px = float(six[0]["s"]), float(six[0]["p"])
        if not (sh > 0 and px > 0):
            raise StudyError(f"a primary 6 print with size {sh} and price {px}")
        return dict(kind="round", shares=sh, price=px, usd=sh * px, t=six[0]["t"], other_listing_6=other6,
                    six_venues=venues)
    if em:
        return dict(kind="no_round", shares=0.0, price=float(em[0]["p"]), usd=0.0, t=em[0]["t"],
                    other_listing_6=other6, six_venues="")
    return dict(kind="no_primary", shares=np.nan, price=np.nan, usd=np.nan, t=None, other_listing_6=other6,
                six_venues="")


def tw_half_spread(t_ns: np.ndarray, bid: np.ndarray, ask: np.ndarray, lo_ns: int, hi_ns: int):
    """Time-weighted quoted half-spread (bp of mid) over [lo, hi), and the seconds it covers.

    Each quote holds from its own time to the next quote's. A quote with bid <= 0, ask <= 0 or
    ask < bid is not a valid NBBO: its interval is EXCLUDED from the weight (counted by the
    caller through the covered seconds), never filled from the quote before it. The interval
    before the first quote in the stream is not covered. Returns (None, 0.0) when nothing is."""
    if len(t_ns) == 0:
        return None, 0.0
    o = np.argsort(t_ns, kind="stable")
    t, b, a = t_ns[o], bid[o].astype(float), ask[o].astype(float)
    ends = np.append(t[1:], np.int64(hi_ns))
    st = np.clip(t, lo_ns, hi_ns)
    en = np.clip(ends, lo_ns, hi_ns)
    w = (en - st).astype(float)
    valid = (b > 0) & (a > 0) & (a >= b)
    w = np.where(valid, w, 0.0)
    if w.sum() <= 0:
        return None, 0.0
    hs = np.where(valid, (a - b) / np.where(valid, a + b, 1.0) * 1e4, 0.0)
    return float((hs * w).sum() / w.sum()), float(w.sum() / 1e9)


def quote_at(t_ns: np.ndarray, bid: np.ndarray, ask: np.ndarray, tau_ns: int):
    """(bid, ask, age seconds) of the last VALID quote at or before tau, or None.

    Only the last quote at or before tau is the NBBO at tau. If it is invalid (crossed or zero) the
    NBBO at tau is not a usable reference and None is returned: an earlier valid quote is a stale
    NBBO, not the one that prevailed."""
    if len(t_ns) == 0:
        return None
    o = np.argsort(t_ns, kind="stable")
    t = t_ns[o]
    i = np.searchsorted(t, tau_ns, side="right") - 1
    if i < 0:
        return None
    b, a = float(bid[o][i]), float(ask[o][i])
    if not (b > 0 and a > 0 and a >= b):
        return None
    return b, a, (tau_ns - int(t[i])) / 1e9


def window_bars(bars: pd.DataFrame, close: pd.Timestamp, lo_min: int, hi_min: int) -> pd.DataFrame:
    """The 5-minute bars lying wholly inside [close - lo_min, close - hi_min): start >= the window
    start and start + 5 min <= the window end. `bars.t` is the bar's START (Alpaca's convention)."""
    a = close - pd.Timedelta(minutes=lo_min)
    b = close - pd.Timedelta(minutes=hi_min)
    return bars[(bars.t >= a) & (bars.t + pd.Timedelta(minutes=5) <= b)]


def causal_rolling_median(x: pd.Series, n: int, min_periods: int) -> pd.Series:
    """Median of the previous n values (the value at D itself is excluded: shift 1 first)."""
    return x.shift(1).rolling(n, min_periods=min_periods).median()


def causal_ewm_mean(x: pd.Series, halflife: float, min_periods: int) -> pd.Series:
    """EWMA of the values strictly before each row (shift 1); NaN values are skipped, not filled."""
    return x.shift(1).ewm(halflife=halflife, min_periods=min_periods, ignore_na=True).mean()


def empirical_quantile(e: np.ndarray, p: float) -> float:
    """The p-quantile of residuals that may hold -inf (a day with no round-lot cross), by the
    inverted CDF (no interpolation, so a quantile inside the -inf mass is -inf, never NaN)."""
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    if len(e) == 0:
        raise StudyError("no residuals to take a quantile of")
    return float(np.quantile(e, p, method="inverted_cdf"))


def cap_split(order_usd: np.ndarray, q_usd: np.ndarray, c: float):
    """MOC part = min(order, c x the low-quantile forecast cross); the rest is worked. A forecast
    of zero (or -inf in logs) sends nothing to the auction. NaN forecasts raise: no default."""
    order_usd = np.asarray(order_usd, float)
    q_usd = np.asarray(q_usd, float)
    if np.isnan(q_usd).any() or np.isnan(order_usd).any():
        raise StudyError("cap_split: a NaN order or forecast")
    moc = np.minimum(order_usd, c * np.maximum(q_usd, 0.0))
    return moc, order_usd - moc


def weighted_median(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    if len(x) == 0:
        return float("nan")
    if (w < 0).any():
        raise ValueError("negative weight")
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def calendar_flags(sessions: list[dt.date], closes: dict) -> pd.DataFrame:
    """Calendar facts about each session, all known in advance (module docstring)."""
    s = pd.Series(sessions)
    nxt = s.shift(-1)
    rows = []
    for i, d in enumerate(sessions):
        n = nxt.iloc[i]
        month_end = (n is None or pd.isna(n) or n.month != d.month)
        tf = third_friday(d.year, d.month)
        # the third-Friday expiry moves to Thursday when the Friday is a holiday
        opex = d == tf or (tf not in closes and d == tf - dt.timedelta(days=1))
        last_fri_june = d.month == 6 and d.weekday() == 4 and (d + dt.timedelta(days=7)).month == 7
        rows.append(dict(date=d, month_end=bool(month_end),
                         quarter_end=bool(month_end and d.month in (3, 6, 9, 12)),
                         opex=bool(opex), quad=bool(opex and d.month in (3, 6, 9, 12)),
                         june_last_fri=bool(last_fri_june),
                         half_day=closes[d] != "16:00",
                         pre_half_day=bool(n is not None and not pd.isna(n) and closes.get(n) not in (None, "16:00"))))
    return pd.DataFrame(rows).set_index("date")


# =============================================================================================
# fetch (GET only)
# =============================================================================================

def _refuse(path: Path, names) -> None:
    from scripts.cef.live_attribution import _refuse_data_dir, _refuse_git_visible
    _refuse_data_dir(path)
    _refuse_git_visible(path, names)


def _client():
    from quantt.broker.alpaca import AlpacaClient
    return AlpacaClient.from_environment("cef")


def dget(c, path: str, params: dict):
    from quantt.broker.alpaca import DATA_BASE
    if path not in DATA_PATHS:
        raise StudyError(f"refusing data path {path}")
    time.sleep(REQ_SLEEP_S)
    return c._get(DATA_BASE, path, params)


def paged(c, path: str, params: dict, key: str, max_pages: int = 2000) -> dict:
    out, tok, pages = {}, None, 0
    while True:
        p = dict(params, **({"page_token": tok} if tok else {}))
        body = dget(c, path, p)
        if not isinstance(body, dict) or key not in body:
            raise StudyError(f"GET {path}: response has no {key!r} key")
        for sym, v in (body[key] or {}).items():
            out.setdefault(sym, []).extend(v)
        pages += 1
        nt = body.get("next_page_token")
        if not nt:
            return out
        if nt == tok or pages >= max_pages:
            raise StudyError(f"GET {path}: paging did not terminate")
        tok = nt


def _z(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def universe() -> pd.DataFrame:
    """Every name of both books with its book and listing exchange (Alpaca asset `exchange`)."""
    from scripts.cef.spec import UNIVERSE
    c4 = pd.read_csv(C4_UNIVERSE)
    if c4.ticker.duplicated().any():
        raise StudyError("C4 universe file repeats a ticker")
    rows = [dict(ticker=t, book="core", exchange="NYSE") for t in sorted(UNIVERSE)]
    both = sorted(set(UNIVERSE) & set(c4.ticker))
    if both:
        raise StudyError(f"in both books' universes: {both}")
    rows += [dict(ticker=r.ticker, book="c4", exchange=r.exchange) for r in c4.itertuples()]
    return pd.DataFrame(rows)


# Historical SIP codes that carried a listing venue's own auction prints in Alpaca's data. Nasdaq
# listings (code Q, "NASDAQ OMX") printed their closing `6` on code T ("NASDAQ Int" in
# /v2/stocks/meta/exchanges): on T alone on 3,951 name-days (3,632 before Q prints begin on
# 2021-04-01, the last T-only day 2024-01-09), and on both T and Q, equal in size and price every time,
# on 1,810 (2021-04-01 → 2024-11-20) [V: measured 2026-11-24 on C4's four Nasdaq listings after an
# independent review; the counts are in the JSON's checks]. Reading Q alone scored the T-only days as
# having no round-lot cross. T's `6` prints on NYSE and NYSE American names are NOT those names'
# primary and stay ignored.
PRIMARY_ALIASES = {"Q": ("T",)}


def primary_codes(U: pd.DataFrame, exchanges: dict) -> dict:
    """{ticker: (the listing venue's SIP code, its historical aliases...)}, through
    quantt.broker.alpaca's verified name map; the first code is the venue's code today (the one the
    tape and the trade reads use). The core's exchange is NYSE (exec_cost_2026-11-05/exec_cost.py
    `primary_codes` asserts the same)."""
    from quantt.broker.alpaca import ASSET_EXCHANGE_SIP_NAME
    out = {}
    for r in U.itertuples():
        name = ASSET_EXCHANGE_SIP_NAME.get(r.exchange)
        codes = sorted(k for k, v in exchanges.items() if v == name)
        if name is None or len(codes) != 1:
            raise StudyError(f"{r.ticker}: exchange {r.exchange!r} maps to SIP codes {codes}")
        for alias in PRIMARY_ALIASES.get(codes[0], ()):
            if alias not in exchanges:
                raise StudyError(f"alias SIP code {alias} is not in /v2/stocks/meta/exchanges")
        out[r.ticker] = (codes[0], *PRIMARY_ALIASES.get(codes[0], ()))
    return out


def sessions_done(c, now: pd.Timestamp) -> list[dict]:
    """Alpaca's calendar from 2015-09-01, every session whose close is more than 20 minutes old."""
    cal = c.calendar(dt.date(2015, 9, 1), now.tz_convert(NY).date())
    return [dict(date=r["date"].isoformat(), open=r["open"], close=r["close"]) for r in cal
            if close_ts(r["date"], r["close"]) + pd.Timedelta(minutes=20) < now]


def sample_quote_sessions(cal: list[str]) -> list[str]:
    """Every one of the last SPREAD_RECENT sessions, and every SPREAD_EVERY-th before them."""
    cal = [d for d in cal if d >= FIRST.isoformat()]
    recent = cal[-SPREAD_RECENT:]
    older = cal[:-SPREAD_RECENT][::SPREAD_EVERY]
    return older + recent


def reduce_quotes(q: list[dict], close: pd.Timestamp) -> dict:
    """One name-day's SIP NBBO stream reduced to what the analysis reads (module docstring)."""
    if not q:
        return {"n": 0}
    t = pd.to_datetime([x["t"] for x in q], utc=True, format="ISO8601").values.astype("datetime64[ns]").astype(np.int64)
    b = np.array([float(x["bp"]) for x in q])
    a = np.array([float(x["ap"]) for x in q])
    c_ns = close.value
    out = {"n": len(q), "n_invalid": int(((b <= 0) | (a <= 0) | (a < b)).sum()), "hs": {}, "cov_s": {}, "tau": {}}
    for k, (lo, hi) in {**WINDOWS, **SPREAD_EXTRA}.items():
        v, cov = tw_half_spread(t, b, a, c_ns - lo * 60 * 10**9, c_ns - hi * 60 * 10**9)
        out["hs"][k], out["cov_s"][k] = v, cov
    for k, sec in TAU_S.items():
        r = quote_at(t, b, a, c_ns - sec * 10**9)
        out["tau"][k] = None if r is None else [r[0], r[1], round(r[2], 3)]
    return out


def fetch_quotes(c, cache: Path, cal: list[dict], closes: dict, syms: list[str], reverse: bool = False) -> None:
    """Step 4: the NBBO stream from QUOTE_LEAD_MIN before the close to the close, on the sample
    sessions, reduced per name-day as it arrives (a day is written whole or not at all)."""
    days = sample_quote_sessions([r["date"] for r in cal])
    for d in (days[::-1] if reverse else days):
        p = cache / "quotes" / f"{d}.json"
        if p.exists():
            continue
        cl = close_ts(dt.date.fromisoformat(d), closes[d])
        got = paged(c, "/v2/stocks/quotes", {"symbols": ",".join(syms), "start": _z(cl - pd.Timedelta(minutes=QUOTE_LEAD_MIN)),
                                             "end": _z(cl), "feed": "sip", "limit": 10000}, "quotes")
        rec = {"date": d, "close": closes[d], "names": {s: reduce_quotes(got.get(s, []), cl) for s in syms}}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(p)
        print(f"quotes {d}: {sum(len(v) for v in got.values())} rows", flush=True)


def fetch(a) -> int:
    cache, priv = a.cache_dir, a.private_dir
    _refuse(cache, ["calendar.json", "exchanges.json", "auctions_close_2016.parquet"])
    _refuse(priv, ["orders_cef.json", "eod_equity_cef.json"])
    cache.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    for sub in ("bars5", "quotes", "tape"):
        (cache / sub).mkdir(exist_ok=True)
    c = _client()
    now = pd.Timestamp.now(tz="UTC")
    cal = sessions_done(c, now)
    (cache / "calendar.json").write_text(json.dumps({"fetched_at": now.isoformat(), "sessions": cal}))
    last = cal[-1]["date"]
    closes = {r["date"]: r["close"] for r in cal}
    print(f"calendar: {len(cal)} sessions to {last}", flush=True)
    ex = c.stock_exchanges()
    (cache / "exchanges.json").write_text(json.dumps(ex))
    U = universe()
    codes = primary_codes(U, ex)
    syms = sorted(U.ticker)
    end_last = close_ts(dt.date.fromisoformat(last), closes[last]) + pd.Timedelta(minutes=30)
    if a.only == "quotes":
        fetch_quotes(c, cache, cal, closes, syms, reverse=a.reverse)
        return 0
    if a.only == "orders":
        fetch_orders(c, priv, now, last, a.refresh_orders)
        return 0
    if a.only == "tradecheck":
        fetch_tradecheck(c, cache, [r["date"] for r in cal], closes, sorted(U[U.book == "core"].ticker))
        fetch_tapecheck_non_nyse(c, cache, [r["date"] for r in cal], closes, U, codes)
        return 0
    # 1. closing-auction prints, every venue, per calendar year
    for y in range(FIRST.year, dt.date.fromisoformat(last).year + 1):
        tag = str(y) if y < dt.date.fromisoformat(last).year else f"{y}_to_{last}"
        p = cache / f"auctions_close_{tag}.parquet"
        if p.exists():
            continue
        lo = pd.Timestamp(dt.datetime(y, 1, 1, tzinfo=NY)).tz_convert("UTC")
        hi = min(pd.Timestamp(dt.datetime(y, 12, 31, 23, 59, tzinfo=NY)).tz_convert("UTC"), end_last)
        got = paged(c, "/v2/stocks/auctions", {"symbols": ",".join(syms), "start": _z(lo), "end": _z(hi),
                                               "feed": "sip", "limit": 10000}, "auctions")
        rows = []
        for sym, days in got.items():
            if sym not in syms:
                raise StudyError(f"unrequested symbol {sym}")
            for day in days:
                for pr in day.get("c") or []:
                    if "s" not in pr:
                        raise StudyError(f"{sym} {day['d'][:10]}: closing print with no size")
                    rows.append(dict(date=day["d"][:10], ticker=sym, x=pr["x"], c=pr["c"], p=float(pr["p"]),
                                     s=float(pr["s"]), t=pr["t"]))
        pd.DataFrame(rows).to_parquet(p, index=False)
        print(f"auctions {y}: {len(rows)} closing prints", flush=True)
    # 2. daily bars, raw and split-adjusted (the split detector), the universe and the regime symbol
    for adj in ("raw", "split"):
        p = cache / f"bars1d_{adj}_to_{last}.parquet"
        if p.exists():
            continue
        got = paged(c, "/v2/stocks/bars", {"symbols": ",".join(syms + [REGIME_SYMBOL]), "timeframe": "1Day",
                                           "start": "2015-09-01T00:00:00Z", "end": _z(end_last),
                                           "adjustment": adj, "feed": "sip", "limit": 10000}, "bars")
        rows = [dict(ticker=s, date=pd.Timestamp(b["t"]).tz_convert(NY).date().isoformat(), o=b["o"], c=b["c"],
                     v=float(b["v"]), vw=b.get("vw"), n=b.get("n"), fetched_at=pd.Timestamp.now(tz="UTC").isoformat())
                for s, v in got.items() for b in v]
        pd.DataFrame(rows).to_parquet(p, index=False)
        print(f"daily bars {adj}: {len(rows)}", flush=True)
    # 3. 5-minute bars, regular session only kept, per month
    months = pd.period_range(FIRST, dt.date.fromisoformat(last), freq="M")
    for m in months:
        tag = str(m) if str(m) < last[:7] else f"{m}_to_{last}"
        p = cache / "bars5" / f"{tag}.parquet"
        if p.exists():
            continue
        lo = pd.Timestamp(dt.datetime(m.year, m.month, 1, tzinfo=NY)).tz_convert("UTC")
        hi = min(pd.Timestamp(dt.datetime((m + 1).year, (m + 1).month, 1, tzinfo=NY)).tz_convert("UTC")
                 - pd.Timedelta(seconds=1), end_last)
        got = paged(c, "/v2/stocks/bars", {"symbols": ",".join(syms), "timeframe": "5Min", "start": _z(lo),
                                           "end": _z(hi), "adjustment": "raw", "feed": "sip", "limit": 10000}, "bars")
        B = pd.DataFrame([(s, b["t"], b["c"], b["v"], b["vw"], b["n"]) for s, v in got.items() for b in v],
                         columns=["ticker", "t", "c", "v", "vw", "n"])
        if len(B):
            B["t"] = pd.to_datetime(B.t, utc=True, format="ISO8601")
            d = B.t.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
            sess = pd.DataFrame([(k, close_ts(dt.date.fromisoformat(k), "09:30"), close_ts(dt.date.fromisoformat(k), v))
                                 for k, v in closes.items() if k[:7] == str(m)], columns=["d", "open_ts", "close_ts"])
            B = B.assign(d=d).merge(sess, on="d", how="inner")
            # regular session only: a bar starting at or after 09:30 and before the close
            B = B[(B.t >= B.open_ts) & (B.t < B.close_ts)][["ticker", "t", "c", "v", "vw", "n"]]
        B.to_parquet(p, index=False)
        print(f"5-min bars {m}: {len(B)}", flush=True)
    # 4. NBBO quotes on the sample sessions, reduced per name-day
    fetch_quotes(c, cache, cal, closes, syms)
    # 5. the primary's tape on no-6 name-days of the last TAPE_SESSIONS sessions (odd lot or none)
    A = pd.concat([pd.read_parquet(f) for f in sorted(cache.glob("auctions_close_*.parquet"))], ignore_index=True)
    recent = [r["date"] for r in cal][-TAPE_SESSIONS:]
    A = A[A.date.isin(recent)]
    for d, g in A.groupby("date"):
        p = cache / "tape" / f"{d}.json"
        if p.exists():
            continue
        need = []
        for s, h in g.groupby("ticker"):
            prim = h[h.x.isin(codes[s])]
            if (prim.c == "M").any() and not (prim.c == "6").any():
                need.append(s)
        rec = {"date": d, "names": {}, "cov": {}}
        if need:
            cl = close_ts(dt.date.fromisoformat(d), closes[d])
            win = [(cl, cl + pd.Timedelta(seconds=60))]
            late = pd.to_datetime(g[g.ticker.isin(need) & (g.c == "M")].t, format="ISO8601", utc=True)
            late = late[late > cl + pd.Timedelta(seconds=59)]
            if len(late):
                lo2, hi2 = late.min() - pd.Timedelta(seconds=2), late.max() + pd.Timedelta(seconds=1)
                if hi2 - lo2 > pd.Timedelta(minutes=30):
                    raise StudyError(f"{d}: late primary prints span {hi2 - lo2}")
                win.append((lo2, hi2))
            for s in need:
                rec["names"][s], rec["cov"][s] = [], []
            for lo, hi in win:
                got = paged(c, "/v2/stocks/trades", {"symbols": ",".join(need), "start": _z(lo), "end": _z(hi),
                                                     "feed": "sip", "limit": 10000}, "trades")
                for s in need:
                    rec["names"][s] += [{"t": r["t"], "p": r["p"], "s": r["s"], "c": list(r.get("c") or [])}
                                        for r in got.get(s, []) if r["x"] == codes[s][0]]
                    rec["cov"][s].append([_z(lo), _z(hi)])
        p.write_text(json.dumps(rec))
    print("tape done", flush=True)
    # 6. the cef account's orders and end-of-day equity (account data: private directory only)
    fetch_orders(c, priv, now, last, a.refresh_orders)
    return 0


def fetch_tradecheck(c, cache: Path, cal: list[str], closes: dict, syms: list[str]) -> None:
    """A second method for the afternoon pool: the raw SIP trades of the core names in each window
    on the last TRADECHECK_SESSIONS sessions, summed (all trades, and by condition code), to set
    beside the 5-minute bars' dollar volume."""
    out = {}
    for d in cal[-TRADECHECK_SESSIONS:]:
        cl = close_ts(dt.date.fromisoformat(d), closes[d])
        lo = cl - pd.Timedelta(minutes=max(v[0] for v in WINDOWS.values()))
        hi = cl - pd.Timedelta(minutes=min(v[1] for v in WINDOWS.values()))
        got = paged(c, "/v2/stocks/trades", {"symbols": ",".join(syms), "start": _z(lo), "end": _z(hi),
                                             "feed": "sip", "limit": 10000}, "trades")
        day = {}
        for sym, rows in got.items():
            T = pd.DataFrame(rows)
            if T.empty:
                continue
            T["t"] = pd.to_datetime(T.t, utc=True, format="ISO8601")
            T["usd"] = T.p * T.s
            T["cond"] = T.c.map(lambda x: "".join(sorted(x or [])))
            rec = {}
            for k, (wlo, whi) in WINDOWS.items():
                m = (T.t >= cl - pd.Timedelta(minutes=wlo)) & (T.t < cl - pd.Timedelta(minutes=whi))
                rec[k] = {"usd_all": float(T.usd[m].sum()), "n": int(m.sum()),
                          "usd_by_condition": {str(kk): float(vv) for kk, vv in T[m].groupby("cond").usd.sum().items()}}
            day[sym] = rec
        out[d] = day
        print(f"tradecheck {d}: {sum(len(v) for v in got.values())} trades", flush=True)
    (cache / "tradecheck.json").write_text(json.dumps(out))


def fetch_tapecheck_non_nyse(c, cache: Path, cal: list[str], closes: dict, U: pd.DataFrame, codes: dict) -> None:
    """The 2026-10-13 rule (the primary's 6 print is the whole round-lot cross) was checked on NYSE
    listings. C4 also holds NYSE American and Nasdaq listings: read their primary's tape for the 30 s
    after the close on the last TRADECHECK_SESSIONS sessions, to set beside the 6 print."""
    names = sorted(U[(U.book == "c4") & (U.exchange != "NYSE")].ticker)
    out = {}
    for d in cal[-TRADECHECK_SESSIONS:]:
        cl = close_ts(dt.date.fromisoformat(d), closes[d])
        got = paged(c, "/v2/stocks/trades", {"symbols": ",".join(names), "start": _z(cl - pd.Timedelta(seconds=1)),
                                             "end": _z(cl + pd.Timedelta(seconds=30)), "feed": "sip", "limit": 10000}, "trades")
        out[d] = {s: [{"t": r["t"], "s": r["s"], "c": list(r.get("c") or [])} for r in got.get(s, [])
                      if r["x"] == codes[s][0] and pd.Timestamp(r["t"]) >= cl] for s in names}
    (cache / "tapecheck_non_nyse.json").write_text(json.dumps(out))
    print(f"tapecheck (non-NYSE listings): {len(names)} names x {len(out)} sessions", flush=True)


def fetch_orders(c, priv: Path, now: pd.Timestamp, last: str, refresh: bool) -> None:
    """Step 6: GET /v2/orders (status=all, every page) and the EOD equity history of the cef paper
    account, into the private directory only."""
    po = priv / "orders_cef.json"
    if not po.exists() or refresh:
        O = c.orders(status="all", after="2026-09-26T00:00:00Z")
        po.write_text(json.dumps({"fetched_at": now.isoformat(), "orders": O}))
        print(f"orders (cef): {len(O)}", flush=True)
    pe = priv / "eod_equity_cef.json"
    if not pe.exists() or refresh:
        E = c.eod_equity_history(dt.date(2026, 9, 28), dt.date.fromisoformat(last))
        pe.write_text(json.dumps({"fetched_at": now.isoformat(),
                                  "points": [[p.date.isoformat(), p.equity, p.stamp.isoformat()] for p in E]}))
        print(f"eod equity (cef): {len(E)} points", flush=True)


# =============================================================================================
# analysis (offline)
# =============================================================================================

def r(x, k=2):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, k)


def tstat(x) -> float | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2)


def qinf(x, p) -> float:
    """Quantile without interpolation, so an infinite participation (a day with no round-lot cross)
    counts as the largest value instead of turning the interpolation into NaN."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, p, method="inverted_cdf")) if len(x) else float("nan")


def q(x, p):
    x = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.quantile(p)) if len(x) else float("nan")


def load_calendar(cache: Path):
    c = json.loads((cache / "calendar.json").read_text())
    ses = [x for x in c["sessions"] if x["date"] >= FIRST.isoformat()]
    return [x["date"] for x in ses], {x["date"]: x["close"] for x in c["sessions"]}, pd.Timestamp(c["fetched_at"])


def load_daily(cache: Path, last: str, fetched: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [cache / f"bars1d_raw_to_{last}.parquet", cache / f"bars1d_split_to_{last}.parquet"]
    raw, spl = (pd.read_parquet(p) for p in paths)
    # A daily bar dated the fetch day is complete only after 20:00 ET (extended hours end then). The
    # fetch time is the bar file's own: its `fetched_at` column where present, else the file's write
    # time less a 10-minute margin (the write follows the request; calendar.json is rewritten by
    # every `fetch --only` run, so it does not date the bars).
    for p, D in zip(paths, (raw, spl)):
        if "fetched_at" in D:
            fday = pd.Timestamp(D.fetched_at.iloc[0]).tz_convert(NY)
        else:
            fday = (pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") - pd.Timedelta(minutes=10)).tz_convert(NY)
        if (D.date == fday.date().isoformat()).any() and fday.time() < dt.time(20, 0):
            raise StudyError(f"{p.name} fetched about {fday} includes that day's bar, which may be partial")
    for D in (raw, spl):
        if D.duplicated(["ticker", "date"]).any():
            raise StudyError("two daily bars for one name-day")
    return raw, spl


def split_events(raw: pd.DataFrame, spl: pd.DataFrame) -> pd.DataFrame:
    """Splits from Alpaca's own raw and split-adjusted daily closes: the ratio raw/adjusted is a
    step function of the date that steps on each split's ex-date."""
    m = raw.merge(spl, on=["ticker", "date"], suffixes=("", "_a"))
    m["ratio"] = m.c / m.c_a
    m = m.sort_values(["ticker", "date"])
    g = m.groupby("ticker").ratio
    # the adjusted close is rounded, so the ratio wobbles in the fifth digit; a split is a
    # PERSISTENT step: the median of the next 5 sessions against the median of the previous 5
    m["after"] = g.transform(lambda x: x[::-1].rolling(5, min_periods=1).median()[::-1])
    m["before"] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).median())
    m["step"] = m.before / m.after
    ev = m[m.before.notna() & ((m.step - 1).abs() > 0.005)]
    # one event per run of flagged days (a step shows on up to 5 consecutive days): keep the day
    # with the largest step in each run
    rows = []
    for t, h in ev.groupby("ticker"):
        idx = list(h.index)
        runs, cur = [], [idx[0]]
        for i in idx[1:]:
            if m.index.get_loc(i) - m.index.get_loc(cur[-1]) <= 5:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        for rr in runs:
            j = max(rr, key=lambda i: abs(math.log(m.at[i, "step"])))
            rows.append(dict(ticker=t, date=m.at[j, "date"], factor=float(m.at[j, "step"])))
    return pd.DataFrame(rows, columns=["ticker", "date", "factor"])


def cross_panel(A: pd.DataFrame, codes: dict, raw: pd.DataFrame, U: pd.DataFrame, sessions: list[str]) -> pd.DataFrame:
    """One row per name-day on which the name had a daily bar or a closing print (2016 on)."""
    A = A[A.date >= FIRST.isoformat()]
    rows = {}
    for (d, s), g in A.groupby(["date", "ticker"], sort=False):
        rows[(d, s)] = classify_close(g.to_dict("records"), codes[s])
    bars = raw[raw.date.isin(sessions) & raw.ticker.isin(codes)].set_index(["date", "ticker"])
    keys = sorted(set(rows) | set(bars.index))
    out = []
    for k in keys:
        x = rows.get(k) or dict(kind="no_primary", shares=np.nan, price=np.nan, usd=np.nan, t=None,
                                other_listing_6=False, six_venues="")
        b = bars.loc[k] if k in bars.index else None
        out.append(dict(date=k[0], ticker=k[1], **x, day_v=float(b.v) if b is not None else np.nan,
                        day_c=float(b.c) if b is not None else np.nan,
                        day_usd=float(b.v * b.vw) if b is not None and b.vw is not None else np.nan))
    X = pd.DataFrame(out)
    X = X[X.date.isin(sessions)]
    traded = X.day_v > 0
    X["venue"] = X.ticker.map({t: c[0] for t, c in codes.items()})
    # A TRADED day with no primary print at all (no 6, no M). What it means depends on whether the
    # name's listing venue publishes an official close (M) in Alpaca's data yet. The M era starts per
    # venue, measured: the first session with an M print from the venue's codes on any of its names
    # (NYSE and NYSE American 2020-11-04; Nasdaq 2021-04-01 here).
    #   before the venue's M era: no M exists to say anything, and the 2026-10-13 note's tape check
    #       found such days to be odd lots or no trade after the bell (§1 there): "no_round", flagged
    #       noM, counted. A label, not a measurement of each day;
    #   in the M era, on a session where NONE of the venue's names has an M (124 such sessions in
    #       2020-11 → 2022-07): Alpaca's M data are missing for the session, so a missing 6 is not
    #       evidence of a zero cross: "missing_M_session", counted, never sized (independent review,
    #       2026-11-24: these had been scored as zero crosses);
    #   in the M era, on a session where other names of the venue have an M: "missing_print", the
    #       same, for this name alone.
    # Two exceptions at any date, counted and never sized:
    #   listing_moved  another listing venue printed a 6 that day (JRS's prints were on another
    #                  listing venue until 2016-08-25);
    #   data_gap       fewer than GAP_PRINT_SHARE of the session's traded names have any primary print
    #                  (2019-08-12: 6 primary prints across the traded names).
    prim = X[traded].assign(has=lambda d: ~d.kind.isin(["no_primary"])).groupby("date").has.mean()
    gap_days = set(prim[prim < GAP_PRINT_SHARE].index)
    am = A[A.c == "M"].assign(venue=lambda d: d.ticker.map({t: c[0] for t, c in codes.items()}))
    am = am[[x in codes[t] for x, t in zip(am.x, am.ticker)]]
    m_start = am.groupby("venue").date.min().to_dict()
    m_sessions = am.groupby("venue").date.agg(lambda v: set(v)).to_dict()
    if set(X.venue) - set(m_start):
        raise StudyError(f"no M print at all for venue(s) {sorted(set(X.venue) - set(m_start))}")
    in_m_era = np.array([d >= m_start[v] for d, v in zip(X.date, X.venue)])
    venue_has_m = np.array([d in m_sessions[v] for d, v in zip(X.date, X.venue)])
    nop = (X.kind == "no_primary") & traded
    moved = nop & X.other_listing_6
    gap = nop & ~X.other_listing_6 & X.date.isin(gap_days)
    rest = nop & ~X.other_listing_6 & ~X.date.isin(gap_days)
    X["noM"] = rest & ~in_m_era
    X.loc[moved, "kind"] = "listing_moved"
    X.loc[gap, "kind"] = "data_gap"
    X.loc[rest & in_m_era & ~venue_has_m, "kind"] = "missing_M_session"
    X.loc[rest & in_m_era & venue_has_m, "kind"] = "missing_print"
    X.loc[X.noM, ["kind", "shares", "usd"]] = ["no_round", 0.0, 0.0]
    X.loc[(X.kind == "no_primary") & ~traded, "kind"] = "not_traded"     # no bar (or zero volume), no primary print
    X.loc[~traded & X.kind.isin(["round", "no_round", "multi6"]), "kind"] = "print_without_bar"
    X.attrs["gap_days"] = sorted(gap_days)
    X.attrs["m_start"] = m_start
    X.attrs["missing_M_sessions"] = {v: sorted(set(X[(X.venue == v) & in_m_era].date) - m_sessions[v])
                                     for v in m_start}
    X["book"] = X.ticker.map(U.set_index("ticker").book)
    X["share_of_day"] = np.where(X.day_v > 0, X.shares / X.day_v, np.nan)
    return X.sort_values(["ticker", "date"]).reset_index(drop=True)


def name_medians(g: pd.DataFrame) -> dict:
    """Per-name statistics over a window, the 2026-10-13 note's columns (round crosses sized;
    days without one counted)."""
    rd = g[g.kind == "round"]
    return dict(sessions=int(g.date.nunique()), round_days=int(len(rd)),
                no_round_days=int((g.kind == "no_round").sum()),
                other_kinds=int((~g.kind.isin(["round", "no_round"])).sum()),
                med_shares=float(rd.shares.median()) if len(rd) else np.nan,
                med_usd=float(rd.usd.median()) if len(rd) else np.nan,
                p10_shares=q(rd.shares, 0.10), p10_usd=q(rd.usd, 0.10),
                med_share_of_day=float(rd.share_of_day.median()) if len(rd) else np.nan)


def pct5(v, k=2) -> list | None:
    """(min, p25, median, p75, max) of the finite values, or None when there are none."""
    v = pd.Series(v, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return [r(x, k) for x in np.percentile(v, [0, 25, 50, 75, 100])] if len(v) else None


def across(T: pd.DataFrame, col: str) -> list:
    v = T[col].replace([np.inf, -np.inf], np.nan).dropna()
    return [r(x, 4 if "share" in col else 0) for x in np.percentile(v, [0, 25, 50, 75, 100])] if len(v) else None


def bars5_daily(cache: Path, closes: dict, sessions: list[str] | None = None) -> pd.DataFrame:
    """Per name-day dollar volume and VWAP in each window, and to the cap time, from the 5-minute
    bars (month files read one at a time). `mbc` = minutes from the bar's START to the close."""
    out = []
    files = sorted((cache / "bars5").glob("*.parquet"))
    if sessions is not None:
        # exactly one file per month from FIRST to the last session, the last month's tagged with it
        last = sessions[-1]
        months = [str(m) for m in pd.period_range(sessions[0], last, freq="M")]
        want = [f"{m}.parquet" if m < last[:7] else f"{m}_to_{last}.parquet" for m in months]
        have = [f.name for f in files]
        if sorted(have) != sorted(want):
            raise StudyError(f"5-minute bar files differ from the months wanted: missing "
                             f"{sorted(set(want) - set(have))[:3]}, extra {sorted(set(have) - set(want))[:3]}")
    for f in files:
        B = pd.read_parquet(f)
        if B.empty:
            continue
        B["t"] = pd.to_datetime(B.t, utc=True)
        B["date"] = B.t.dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
        cl = {d: close_ts(dt.date.fromisoformat(d), closes[d]) for d in B.date.unique()}
        B["close_ts"] = B.date.map(cl)
        B["mbc"] = (B.close_ts - B.t).dt.total_seconds() / 60.0
        if (B.mbc <= 0).any():
            raise StudyError(f"{f.name}: a regular-session bar starts at or after the close")
        B["usd"] = B.v * B.vw
        cols = {"am": B.mbc >= CAP_LEAD_MIN + 5}          # ends at or before the cap time
        for k, (lo, hi) in WINDOWS.items():
            cols[k] = (B.mbc <= lo) & (B.mbc >= hi + 5)
        cols["full"] = pd.Series(True, index=B.index)
        agg = {}
        # Every name-day with at least one regular-session bar. Alpaca emits a bar only when there is a
        # trade, so on such a day a window with no bar had no trade: its volume is ZERO, not missing
        # (independent review, 2026-11-24: counting it as missing raised the thin names' medians). A
        # name-day with no regular-session bar at all has no row here and stays missing.
        full_idx = B.groupby(["ticker", "date"]).size().index
        for k, m in cols.items():
            g = B[m].groupby(["ticker", "date"])
            agg[f"usd_{k}"] = g.usd.sum().reindex(full_idx, fill_value=0.0)
            agg[f"v_{k}"] = g.v.sum().reindex(full_idx, fill_value=0.0)
            if k == "am":
                agg["am_last_end_mbc"] = (g.mbc.min() - 5).reindex(full_idx)   # alignment: >= CAP_LEAD_MIN
        out.append(pd.DataFrame(agg))
    D = pd.concat(out).reset_index()
    for k in WINDOWS:
        D[f"vwap_{k}"] = np.where(D[f"v_{k}"] > 0, D[f"usd_{k}"] / D[f"v_{k}"], np.nan)
    if (D.am_last_end_mbc.dropna() < CAP_LEAD_MIN - 1e-9).any():
        raise StudyError("alignment: an 'am' volume used a bar ending after the cap time")
    return D


def tapecheck_report(cache: Path, X: pd.DataFrame) -> dict:
    """The non-NYSE listings' 6 print against the sum of their primary's 6-coded trades on the tape."""
    p = cache / "tapecheck_non_nyse.json"
    if not p.exists():
        raise StudyError(f"{p} is missing: run `fetch --only tradecheck`")
    T = json.loads(p.read_text())
    agree = disagree = 0
    no6_odd = no6_none = 0
    for d, day in T.items():
        for s, trades in day.items():
            x = X[(X.date == d) & (X.ticker == s)]
            if x.empty:
                raise StudyError(f"tapecheck {d} {s}: no cross row")
            six = sum(t["s"] for t in trades if "6" in t["c"])
            if x.kind.iloc[0] == "round":
                agree += int(abs(six - x.shares.iloc[0]) < 1e-9)
                disagree += int(abs(six - x.shares.iloc[0]) >= 1e-9)
            elif x.kind.iloc[0] == "no_round":
                odd = [t for t in trades if "I" in t["c"] and not (set(t["c"]) & {"M", "6", "9"})]
                no6_odd += int(bool(odd))
                no6_none += int(not odd)
    return {"sessions": sorted(T), "round crosses whose 6 size equals the tape's 6-coded trades": agree,
            "round crosses that disagree": disagree, "no-6 name-days with an odd-lot (I) primary trade": no6_odd,
            "no-6 name-days with none": no6_none}


def tradecheck_report(cache: Path, D5: pd.DataFrame) -> dict:
    """Bars against raw trades, per core name-day and window (the second method for item 3)."""
    p = cache / "tradecheck.json"
    if not p.exists():
        raise StudyError(f"{p} is missing: run `fetch --only tradecheck`")
    T = json.loads(p.read_text())
    rows, cond = [], {}
    for d, day in T.items():
        for sym, rec in day.items():
            b = D5[(D5.date == d) & (D5.ticker == sym)]
            if b.empty:
                raise StudyError(f"tradecheck {d} {sym}: no 5-minute bars")
            for k in WINDOWS:
                rows.append(dict(date=d, ticker=sym, window=k, trades=rec[k]["usd_all"], bars=float(b[f"usd_{k}"].iloc[0])))
                for c_, v in rec[k]["usd_by_condition"].items():
                    cond[c_] = cond.get(c_, 0.0) + v
    R = pd.DataFrame(rows)
    R["ratio"] = R.bars / R.trades
    tot = sum(cond.values())
    top = sorted(cond.items(), key=lambda kv: -kv[1])[:8]
    return {"sessions": sorted(T), "name_window_days": int(len(R)),
            "bars $ / trades $, per name-day-window: (min, p25, median, p75, max)": pct5(R.ratio, 4),
            "bars $ / trades $, summed over everything": r(R.bars.sum() / R.trades.sum(), 4),
            "share of trade $ by condition-code set (top 8)": {(k if k.strip() else "regular (blank)"): r(v / tot, 4) for k, v in top}}


def load_quotes(cache: Path, sessions: list[str]) -> tuple[pd.DataFrame, dict]:
    """Exactly the declared quote sample (`sample_quote_sessions`); a missing session raises."""
    want = sample_quote_sessions(sessions)
    missing = [d for d in want if not (cache / "quotes" / f"{d}.json").exists()]
    if missing:
        raise StudyError(f"{len(missing)} quote-sample sessions not fetched (first {missing[:3]}): run fetch")
    rows, meta = [], {"sessions": 0, "name_days": 0, "no_quotes": 0, "invalid_quotes": 0, "quotes": 0,
                      "sample": f"every one of the last {SPREAD_RECENT} sessions and every {SPREAD_EVERY}th before them"}
    for d in want:
        rec = json.loads((cache / "quotes" / f"{d}.json").read_text())
        meta["sessions"] += 1
        for s, v in rec["names"].items():
            meta["name_days"] += 1
            if not v.get("n"):
                meta["no_quotes"] += 1
                continue
            meta["quotes"] += v["n"]
            meta["invalid_quotes"] += v["n_invalid"]
            row = dict(date=rec["date"], ticker=s)
            for k, h in v["hs"].items():
                row[f"hs_{k}"] = h
                row[f"cov_{k}"] = v["cov_s"][k]
            for k, tq in v["tau"].items():
                row[f"mid_{k}"] = (tq[0] + tq[1]) / 2 if tq else np.nan
                row[f"hsq_{k}"] = (tq[1] - tq[0]) / (tq[1] + tq[0]) * 1e4 if tq else np.nan
                row[f"age_{k}"] = tq[2] if tq else np.nan
            rows.append(row)
    return pd.DataFrame(rows), meta


def distributions(panels: Path, U: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ex-dates: the core's from cef_distributions, C4's from cef_families/cef_actions (dividends > 0).
    Returns the events and each source's last date (no flag is read after it)."""
    core = pd.read_parquet(panels / "cef" / "cef_distributions.parquet")
    core = core[(core.amount > 0) & core.ticker.isin(U[U.book == "core"].ticker)][["ticker", "ex_date"]]
    fam = pd.read_parquet(panels / "cef_families" / "cef_actions.parquet")
    fam = fam[(fam.dividends > 0) & fam.ticker.isin(U[U.book == "c4"].ticker)].rename(columns={"date": "ex_date"})
    E = pd.concat([core.assign(src="core"), fam[["ticker", "ex_date"]].assign(src="c4")], ignore_index=True)
    E["ex_date"] = pd.to_datetime(E.ex_date).dt.strftime("%Y-%m-%d")
    last = {"core": str(core.ex_date.max())[:10], "c4": str(fam.ex_date.max())[:10]}
    return E, last


def build_features(X: pd.DataFrame, D5: pd.DataFrame, cal: pd.DataFrame, E: pd.DataFrame,
                   ex_last: dict) -> pd.DataFrame:
    """The forecasting frame: one row per name-day with a sized outcome (round or no_round).

    ALIGNMENT (asserted by truncation_check): every rolling quantity is shifted one session within
    the name's own series, so row D sees only crosses and daily bars dated before D; `lam` (dollar
    volume 09:30 to the cap time on D) is from bars that end at or before close - CAP_LEAD_MIN;
    calendar and ex-date flags are facts about D known in advance."""
    F = X[X.kind.isin(["round", "no_round", "multi6", "listing_moved", "data_gap", "missing_M_session",
                       "missing_print"])].copy()
    F = F.merge(D5[["ticker", "date", "usd_am"]], on=["ticker", "date"], how="left")
    F = F.sort_values(["ticker", "date"]).reset_index(drop=True)
    F["sized"] = F.kind.isin(["round", "no_round"])
    with np.errstate(divide="ignore"):
        F["ly"] = np.where(F.sized, np.log(F.usd.where(F.sized)), np.nan)        # -inf on a no_round day
    F["ly_pos"] = F.ly.where(F.kind == "round")
    F["lday"] = np.log(F.day_usd.where(F.day_usd > 0))
    F["lam"] = np.log(F.usd_am.where(F.usd_am > 0))
    g = F.groupby("ticker", sort=False)
    F["med20"] = g.ly.transform(lambda x: causal_rolling_median(x, 20, 15))
    F["med60"] = g.ly.transform(lambda x: causal_rolling_median(x, 60, 40))
    F["ewm10"] = g.ly_pos.transform(lambda x: causal_ewm_mean(x, 10, 15))
    F["naive"] = g.ly.shift(1)
    F["zero_rate60"] = g.ly.transform(lambda x: np.isneginf(x).astype(float).where(x.notna()).shift(1)
                                      .rolling(60, min_periods=40).mean())
    F["u_am"] = F.lam - g.lam.transform(lambda x: causal_ewm_mean(x, 20, 15))
    F["u_day1"] = (F.lday - g.lday.transform(lambda x: causal_ewm_mean(x, 20, 15)))
    F["u_day1"] = F.groupby("ticker", sort=False).u_day1.shift(1)
    F["u_x1"] = (F.ly_pos - F.ewm10)
    F["u_x1"] = F.groupby("ticker", sort=False).u_x1.shift(1)
    C = cal.reset_index().assign(date=lambda x: x.date.astype(str))
    F = F.merge(C, on="date", how="left")
    exd = set(zip(E.ticker, E.ex_date))
    F["ex_date"] = [(t, d) in exd for t, d in zip(F.ticker, F.date)]
    # the NEXT SESSION by the exchange calendar (not the name's own next row, which would reveal
    # whether the name trades tomorrow)
    days = list(cal.index.astype(str))
    nxt = dict(zip(days[:-1], days[1:]))
    F["pre_ex"] = [(t, nxt.get(d)) in exd for t, d in zip(F.ticker, F.date)]
    F["ex_known"] = [d <= ex_last[b] for d, b in zip(F.date, F.book)]
    return F


def truncation_check(F: pd.DataFrame, X: pd.DataFrame, D5: pd.DataFrame, cal: pd.DataFrame, E: pd.DataFrame,
                     ex_last: dict, n: int = 60, seed: int = 7) -> int:
    """Recompute the features of sampled rows from data CUT at D (rows dated after D dropped; D's
    own outcome blanked) and raise on any difference. Returns the number of rows checked."""
    rng = np.random.default_rng(seed)
    cand = F[F.sized & F.med60.notna() & F.u_am.notna()]
    pick = cand.iloc[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
    cols = ["med20", "med60", "ewm10", "naive", "zero_rate60", "u_am", "u_day1", "u_x1"]
    for _, row in pick.iterrows():
        Xc = X[(X.ticker == row.ticker) & (X.date <= row.date)].copy()
        last = Xc.date == row.date
        # blank everything about D that is not known at the cap time
        for c in ("shares", "usd", "day_v", "day_usd", "share_of_day"):
            Xc.loc[last, c] = np.nan
        Xc.loc[last, "kind"] = "round"           # D's kind is unknown at the cap time; any sized kind
        Dc = D5[(D5.ticker == row.ticker) & (D5.date <= row.date)]
        Fc = build_features(Xc, Dc, cal, E, ex_last)
        got = Fc[Fc.date == row.date].iloc[0]
        for c in cols:
            a_, b_ = row[c], got[c]
            if not ((pd.isna(a_) and pd.isna(b_)) or (np.isfinite(a_) and np.isfinite(b_) and abs(a_ - b_) < 1e-9)
                    or (np.isinf(a_) and a_ == b_)):
                raise StudyError(f"lookahead: {row.ticker} {row.date} {c} full={a_} cut={b_}")
    return len(pick)


REG_COLS = ["u_am", "u_day1", "month_end", "quarter_end", "opex", "quad", "june_last_fri", "half_day",
            "pre_half_day", "ex_date", "pre_ex"]
VOL_COLS = ["u_am", "u_day1"]                       # the same-day and yesterday's volume surprise only
CAL_COLS = ["month_end", "quarter_end", "opex", "quad", "june_last_fri", "half_day", "pre_half_day"]
# declared variants (all fitted the same way; reported side by side, nothing chosen from them)
REG_SPECS = {"reg": REG_COLS, "reg_noex": [c for c in REG_COLS if c not in ("ex_date", "pre_ex")],
             "reg_vol": VOL_COLS, "reg_cal": CAL_COLS}


def fit_and_forecast(F: pd.DataFrame, first_eval: str = "2017-01-01") -> pd.DataFrame:
    """Point forecasts of log $ cross per model, and their low quantiles (q10, q25), out of sample.

    The regression model (reg) is ewm10 + b'x, b by OLS on the book's positive-cross days strictly
    before each calendar month (refit monthly, expanding window). Residual quantiles for every model:
    pooled over the book's name-days in the 12 months before the month (a no_round day is a -inf
    residual), and per name over its last 250 sized days before the month (pername). Rows are
    evaluated from `first_eval`."""
    F = F.copy()
    F["month"] = F.date.str[:7]
    for c in REG_COLS:
        F[c] = F[c].astype(float)
    F.loc[~F.ex_known, ["ex_date", "pre_ex"]] = np.nan       # unknown after the source's last date
    models = ["naive", "med20", "med60", "ewm10"] + list(REG_SPECS)
    for lab in REG_SPECS:
        F[lab] = np.nan
    months = sorted(F.month.unique())
    coefs = {}
    for book, G in F.groupby("book"):
        for m in months:
            if m < first_eval[:7]:
                continue
            tr = G[(G.month < m) & (G.kind == "round") & G.ewm10.notna()]
            te = G.month == m
            for lab, cols in REG_SPECS.items():
                t2 = tr.dropna(subset=cols)
                if len(t2) < 500:
                    continue
                Xm = np.column_stack([np.ones(len(t2))] + [t2[c].to_numpy() for c in cols])
                y = (t2.ly_pos - t2.ewm10).to_numpy()
                b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
                idx = G.index[te]
                Z = F.loc[idx]
                Xp = np.column_stack([np.ones(len(Z))] + [Z[c].to_numpy() for c in cols])
                F.loc[idx, lab] = Z.ewm10.to_numpy() + Xp @ b
                if lab == "reg":
                    coefs[(book, m)] = dict(zip(["const"] + cols, b))
    # residuals and their trailing quantiles
    for mdl in models:
        F[f"e_{mdl}"] = np.where(F.sized & F[mdl].notna(), F.ly - F[mdl], np.nan)
    for mdl in models:
        for p in (0.10, 0.25):
            F[f"q{int(p * 100)}_{mdl}"] = np.nan
            F[f"q{int(p * 100)}pn_{mdl}"] = np.nan
    mi = {m: i for i, m in enumerate(months)}
    for book, G in F.groupby("book"):
        for m in months:
            if m < first_eval[:7]:
                continue
            lo = months[max(0, mi[m] - 12)]
            past = G[(G.month >= lo) & (G.month < m)]
            idx = G.index[G.month == m]
            for mdl in models:
                e = past[f"e_{mdl}"].to_numpy()
                e = e[~np.isnan(e)]
                if len(e) < 200:
                    continue
                for p in (0.10, 0.25):
                    F.loc[idx, f"q{int(p * 100)}_{mdl}"] = F.loc[idx, mdl] + empirical_quantile(e, p)
    # per-name quantiles: the name's last 250 sized residuals before the month
    for t, G in F.groupby("ticker"):
        for m in sorted(G.month.unique()):
            if m < first_eval[:7]:
                continue
            past = G[G.month < m]
            idx = G.index[G.month == m]
            for mdl in ("ewm10", "reg_noex", "reg_cal"):
                e = past[f"e_{mdl}"].dropna().to_numpy()[-250:]
                if len(e) < 120:
                    continue
                for p in (0.10, 0.25):
                    F.loc[idx, f"q{int(p * 100)}pn_{mdl}"] = F.loc[idx, mdl] + empirical_quantile(e, p)
    F.attrs["coefs"] = coefs
    return F


def regime_labels(raw: pd.DataFrame, sessions: list[str]) -> pd.Series:
    """'high' / 'low' market volatility on each session: SPY's 20-session realised vol of daily
    close-to-close returns ending the session BEFORE D, against its expanding median to that
    session (built from past data only)."""
    s = raw[raw.ticker == REGIME_SYMBOL].set_index("date").c.sort_index()
    rv = np.log(s).diff().rolling(20, min_periods=20).std()
    lagged = rv.shift(1)
    med = lagged.expanding(min_periods=250).median()
    lab = pd.Series(np.where(lagged > med, "high", np.where(lagged <= med, "low", None)), index=s.index)
    return lab.reindex(sessions)


def forecast_metrics(F: pd.DataFrame, mdl: str, mask) -> dict:
    G = F[mask & F[mdl].notna()]
    pos = G[(G.kind == "round") & np.isfinite(G[f"e_{mdl}"])]
    e = pos[f"e_{mdl}"]
    out = dict(name_days=int(len(G)), positive=int(len(pos)), no_round=int((G.kind == "no_round").sum()),
               bias_median=r(e.median(), 3), mae=r(e.abs().mean(), 3), sd=r(e.std(), 3),
               within_factor2=r((e.abs() <= math.log(2)).mean(), 3),
               r2_vs_name_mean=None)
    if len(pos) > 100:
        y = pos.ly
        ybar = pos.groupby("ticker").ly.transform("mean")
        out["r2_vs_name_mean"] = r(1 - ((y - pos[mdl]) ** 2).sum() / ((y - ybar) ** 2).sum(), 3)
    for p in ("q10", "q25"):
        for kind in ("", "pn"):
            col = f"{p}{kind}_{mdl}"
            if col in G and G[col].notna().any():
                H = G[G[col].notna()]
                lab = f"{p}{'_pername' if kind else ''}"
                # all rows: a quantile of -inf (a zero cap: no-round-lot days fill the lowest decile)
                # can never be breached, so this flatters the calibration where caps are zero
                out[f"cover_{lab}"] = r((H.ly < H[col]).mean(), 3)
                out[f"n_{lab}"] = int(len(H))
                fin = np.isfinite(H[col])
                out[f"zero_cap_share_{lab}"] = r((~fin).mean(), 3)
                out[f"cover_{lab}_positive_cap_days"] = r((H.ly[fin] < H[col][fin]).mean(), 3) if fin.any() else None
                # how far below the point forecast the low quantile sits (median ratio)
                out[f"{lab}_over_point_median"] = r(np.exp((H[col] - H[mdl]).median()), 3)
    return out


def cover_by_year(F: pd.DataFrame, col: str, mask) -> dict:
    """Per year: the share of name-days whose q10 cap is zero (q = -inf), and the breach rate (realised
    cross below q) on the days whose cap is positive (independent review, 2026-11-24)."""
    G = F[mask & F[col].notna()]
    out = {}
    for y, H in G.groupby(G.date.str[:4]):
        fin = np.isfinite(H[col])
        out[y] = {"name_days": int(len(H)), "zero_cap_share": r((~fin).mean(), 3),
                  "breach_on_positive_cap_days": r((H.ly[fin] < H[col][fin]).mean(), 3) if fin.any() else None}
    return out


def per_name_cover(F: pd.DataFrame, col: str, mask) -> list:
    G = F[mask & F[col].notna()]
    c = G.groupby("ticker").apply(lambda h: (h.ly < h[col]).mean(), include_groups=False)
    return [r(x, 3) for x in np.percentile(c, [0, 10, 50, 90, 100])] if len(c) else None


# -------------------------------------------------------------------------------- orders

def paper_orders(priv: Path, closes: dict, sessions: list[str]) -> pd.DataFrame:
    """The core book's orders from the cef account, one row per (session, symbol), ORDERED qty."""
    j = json.loads((priv / "orders_cef.json").read_text())
    O = pd.DataFrame(j["orders"])
    O["sub"] = pd.to_datetime(O.submitted_at, format="ISO8601", utc=True)
    O["date"] = O["sub"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d")
    if not O.date.isin(sessions).all():
        raise StudyError("an order submitted on a day that is not a session")
    meth = {("cls", "market"): "cls", ("day", "market"): "late_market"}
    O["method"] = [meth.get((a, b)) for a, b in zip(O.time_in_force, O.type)]
    if O.method.isna().any():
        raise StudyError(f"unrecognised order types {O[O.method.isna()][['time_in_force', 'type']].drop_duplicates().values.tolist()}")
    if O.groupby(["date", "symbol"]).size().max() > 1:
        raise StudyError("two orders for one symbol on one day: the book nets each symbol into one")
    if O.groupby("date").method.nunique().max() > 1:
        raise StudyError("two methods on one day")
    O["s"] = O.side.map({"buy": 1, "sell": -1})
    O["qty"] = O.qty.astype(float)
    O["filled_qty"] = O.filled_qty.astype(float)
    tod = O["sub"].dt.tz_convert(NY).dt.time
    lm = O.method == "late_market"
    # late_market orders went 8 to 2 minutes before the close (the desk's rules order-path rule 3)
    if not all(dt.time(15, 52) <= t <= dt.time(15, 58) for t in tod[lm]):
        raise StudyError("a late_market order outside 15:52-15:58 ET")
    E = json.loads((priv / "eod_equity_cef.json").read_text())["points"]
    eq = {d: e for d, e, _ in E}
    prv = {d: sessions[i - 1] for i, d in enumerate(sessions) if i}
    O["equity_prev"] = [eq.get(prv.get(d)) for d in O.date]
    if O.equity_prev.isna().any() or (O.equity_prev <= 0).any():
        raise StudyError(f"no prior-close equity for {sorted(O[O.equity_prev.isna()].date.unique())}")
    O["phase"] = np.where(O.date == "2026-09-29", "build (cls, from flat)",
                          np.where(O.date == "2026-09-30", "catch-up", "steady state"))
    return O[["date", "symbol", "side", "s", "qty", "filled_qty", "method", "phase", "equity_prev", "status"]].rename(
        columns={"symbol": "ticker"})


def policy_orders(panels: Path) -> tuple[pd.DataFrame, dict]:
    """The deployed policy's orders in its own backtest, decide t, fill t+1 (a MODEL of orders)."""
    from scripts.cef import band_frontier as bf
    from scripts.cef.spec import BAND_WIDTH, CAPITAL_USD, MAX_GROSS_STRESS, summary
    bf.DATA_DIR = panels / "cef"         # read-only: the laptop's copy of the panels (module docstring)
    T, R = bf.build_targets()
    cap = MAX_GROSS_STRESS if MAX_GROSS_STRESS is not None else np.inf
    H = bf.band_gross_capped(T, BAND_WIDTH, cap)
    px = bf.build_panel()["px"]
    B, stale = policy_order_rows(H, px, CAPITAL_USD)
    B = B[B.date >= FIRST.isoformat()].copy()
    meta = {"spec": str(summary()), "capital_usd": CAPITAL_USD,
            "price_panel_last": str(px.index.max().date()), "first_fill": B.date.min(), "last_fill": B.date.max(),
            "orders": int(len(B)), "fill_days": int(B.date.nunique()),
            "orders rounded on the name's last close because the decide-day close is missing": int(stale)}
    return B, meta


def policy_order_rows(H: pd.DataFrame, px: pd.DataFrame, capital: float) -> tuple[pd.DataFrame, int]:
    """Orders of a weight path, dated on the session they FILL (decided at t, filled at t+1: the
    harness's shift(2) convention, band_frontier.evaluate). The order's size is |H_t - H_{t-1}| x
    capital, in dollars; whole-share rounding (Alpaca routes no fraction to an auction) uses t's close,
    or, where t's close is missing for that name, its last close before t (the desk's rules, landmine 6: the
    sleeve sizes on the name's own last close); those orders are counted. An order that rounds to
    zero shares is not an order. The last row of H has no fill session in its index and is dropped."""
    if not H.index.is_monotonic_increasing:
        raise StudyError("H index must be sorted")
    dH = H.diff()
    dH.iloc[0] = H.iloc[0]
    pxf = px.reindex(index=H.index, columns=H.columns)
    last_close = pxf.ffill()
    fill = dict(zip(H.index[:-1], H.index[1:]))
    rows, stale = [], 0
    for t, f in fill.items():
        for k in H.columns:
            w = float(dH.at[t, k])
            if not np.isfinite(w):
                raise StudyError(f"{k} {t.date()}: weight change is not finite")
            if w == 0.0:
                continue
            p = float(pxf.at[t, k])
            if not (np.isfinite(p) and p > 0):
                p = float(last_close.at[t, k])
                if not (np.isfinite(p) and p > 0):
                    raise StudyError(f"{k} traded on {t.date()} with no close on or before it")
                stale += 1
            sh = int(round(abs(w) * capital / p))
            if sh == 0:
                continue
            rows.append(dict(date=f.strftime("%Y-%m-%d"), ticker=k, s=1 if w > 0 else -1, shares=float(sh),
                             usd_decide=sh * p))
    return pd.DataFrame(rows), stale


# -------------------------------------------------------------------------------- report blocks

def auction_size(X: pd.DataFrame, sessions: list[str], U: pd.DataFrame) -> dict:
    out = {}
    last60 = set(sessions[-60:])
    w1013 = {d for d in sessions if "2026-07-21" <= d <= "2026-10-13"}
    last250 = set(sessions[-250:])
    core = sorted(U[U.book == "core"].ticker)
    for lab, win in (("last 60 sessions", last60), ("2026-10-13 note's window, 2026-07-21 to 2026-10-13", w1013),
                     ("last 250 sessions", last250)):
        blk = {"first": min(win), "last": max(win), "sessions": len(win)}
        for book in ("core", "c4"):
            names = U[U.book == book].ticker
            T = pd.DataFrame({t: name_medians(X[(X.ticker == t) & X.date.isin(win)]) for t in names}).T
            T = T[T.sessions > 0]
            blk[book] = {"names": int(len(T)),
                         "median across names of the name's median cross, shares": r(T.med_shares.median(), 0),
                         "median across names of the name's median cross, USD": r(T.med_usd.median(), 0),
                         "median across names of the name's median share of the day's volume": r(T.med_share_of_day.median(), 5),
                         "across names (min, p25, median, p75, max): median cross USD": across(T, "med_usd"),
                         "across names: 10th-percentile cross USD": across(T, "p10_usd"),
                         "across names: share of the day": across(T, "med_share_of_day"),
                         "name-days with no round-lot cross / sized name-days": r(
                             T.no_round_days.sum() / (T.round_days.sum() + T.no_round_days.sum()), 4),
                         "names with at least one no-round-lot day": int((T.no_round_days > 0).sum()),
                         "name-days of another kind (multi6, no primary print, no bar)": int(T.other_kinds.sum())}
            if book == "core" and lab == "last 60 sessions":
                out["market_by_name_core"] = {t: {k: r(v, 5 if "share" in k else 0) for k, v in T.loc[t].items()}
                                              for t in core if t in T.index}
        out[lab] = blk
    # by year and book
    yr = {}
    X2 = X.assign(year=X.date.str[:4])
    for (book, y), g in X2.groupby(["book", "year"]):
        T = pd.DataFrame({t: name_medians(h) for t, h in g.groupby("ticker")}).T
        sized = g.kind.isin(["round", "no_round"])
        yr.setdefault(book, {})[y] = {
            "names": int(len(T)), "sessions": int(g.date.nunique()),
            "median across names of the median cross USD": r(T.med_usd.median(), 0),
            "median across names of the median cross shares": r(T.med_shares.median(), 0),
            "median across names of the median share of the day": r(T.med_share_of_day.median(), 5),
            "median across names of the median day USD volume (daily bar v x vw)": r(
                g[g.day_usd > 0].groupby("ticker").day_usd.median().median(), 0),
            "no-round-lot share of sized name-days": r((g.kind == "no_round").sum() / max(sized.sum(), 1), 4),
            "multi6 / no-primary name-days": [int((g.kind == "multi6").sum()), int((g.kind == "no_primary").sum())]}
    out["by_year"] = yr
    # the last 15 months, month by month: when did the cross deepen, and did the day deepen with it?
    mo = {}
    recent = X2[X2.date >= sessions[-320]]
    for (book, m), g in recent.assign(month=recent.date.str[:7]).groupby(["book", "month"]):
        rd = g[g.kind == "round"]
        mo.setdefault(book, {})[m] = {
            "median across names of the median cross USD": r(rd.groupby("ticker").usd.median().median(), 0),
            "median across names of the median day USD volume": r(g[g.day_usd > 0].groupby("ticker").day_usd.median().median(), 0),
            "median across names of the median share of the day": r(rd.groupby("ticker").share_of_day.median().median(), 5)}
    out["by_month_recent"] = mo
    return out


def afternoon_pool(X: pd.DataFrame, D5: pd.DataFrame, Q: pd.DataFrame, qmeta: dict, sessions: list[str],
                   U: pd.DataFrame) -> dict:
    out = {"windows_ET_on_a_16:00_close": WINDOW_LABEL,
           "note": "dollar volume = sum of 5-minute bar volume x bar VWAP over bars lying wholly in the window; "
                   "half-spread = time-weighted NBBO half-spread in bp of mid (quote sample)"}
    M = X.merge(D5, on=["ticker", "date"], how="left")
    last60 = set(sessions[-60:])
    core = sorted(U[U.book == "core"].ticker)
    # check: regular-session bars + the cross against the daily bar's volume
    chk = M[M.date.isin(last60) & (M.kind == "round") & (M.day_v > 0)]
    out["check: (regular-session 5-min bar volume + cross) / daily bar volume, median and p10/p90"] = [
        r(q((chk.v_full + chk.shares) / chk.day_v, p), 4) for p in (0.5, 0.1, 0.9)]
    for book in ("core", "c4"):
        g = M[M.book == book]
        rows = {}
        for t, h in g[g.date.isin(last60)].groupby("ticker"):
            rd = h[h.kind == "round"]
            row = {"cross_usd_median": float(rd.usd.median()) if len(rd) else np.nan}
            for k in WINDOWS:
                row[f"usd_{k}_median"] = float(h[f"usd_{k}"].median())
                row[f"usd_{k}_p10"] = q(h[f"usd_{k}"], 0.10)
                row[f"ratio_{k}_to_cross_median"] = q(rd[f"usd_{k}"] / rd.usd, 0.5)
            rows[t] = row
        T = pd.DataFrame(rows).T
        blk = {"names": int(len(T))}
        for k in WINDOWS:
            lab = WINDOW_LABEL[k]
            blk[lab] = {"across names: median dollar volume (min, p25, median, p75, max)": across(T, f"usd_{k}_median"),
                        "across names: 10th-percentile day": across(T, f"usd_{k}_p10"),
                        "across names: window $ / same day's cross $ (median day)": pct5(T[f"ratio_{k}_to_cross_median"]),
                        "at 5% participation, median name, USD": r(0.05 * T[f"usd_{k}_median"].median(), 0),
                        "at 10% participation, median name, USD": r(0.10 * T[f"usd_{k}_median"].median(), 0)}
        out[book] = blk
        if book == "core":
            out["market_by_name_core"] = {t: {kk: r(v, 2) for kk, v in T.loc[t].items()} for t in core if t in T.index}
        # by year: the median name's window dollar volume, and the window / cross ratio
        yr = {}
        for y, h in g.assign(year=g.date.str[:4]).groupby("year"):
            per = h.groupby("ticker").agg(w120=("usd_120_10", "median"), w30=("usd_30_10", "median"))
            rd = h[h.kind == "round"]
            yr[y] = {"median name, 14:00-15:50 USD": r(per.w120.median(), 0),
                     "median name, 15:30-15:50 USD": r(per.w30.median(), 0),
                     "median day, 14:00-15:50 $ / cross $": r(q(rd.usd_120_10 / rd.usd, 0.5), 2)}
        out[book]["by_year"] = yr
    # spreads
    qm = dict(qmeta)
    qm["invalid_quote_share"] = r(qm["invalid_quotes"] / max(qm["quotes"], 1), 5)
    out["quote_sample"] = qm
    dense = set(sessions[-SPREAD_RECENT:])
    for book in ("core", "c4"):
        h = Q[Q.ticker.isin(U[U.book == book].ticker)]
        hd = h[h.date.isin(dense)]
        per = hd.groupby("ticker")[[f"hs_{k}" for k in list(WINDOWS) + list(SPREAD_EXTRA)]].median()
        blk = {"sessions": int(hd.date.nunique()), "first": hd.date.min(), "last": hd.date.max()}
        for k in list(WINDOWS) + list(SPREAD_EXTRA):
            lab = WINDOW_LABEL.get(k, "15:50-16:00")
            blk[f"half-spread {lab}, across names (min, p25, median, p75, max), bp"] = pct5(per[f"hs_{k}"])
        hs_era = {}
        for era, (lo, hi) in ERAS.items():
            e = h[(h.date >= lo) & (h.date <= hi)]
            if len(e):
                hs_era[era] = {"sessions": int(e.date.nunique()),
                               "median name, 14:00-15:50 half-spread bp": r(e.groupby("ticker").hs_120_10.median().median(), 2)}
        blk["by_era (sparse sample before the last 100 sessions)"] = hs_era
        out[f"spreads_{book}"] = blk
        if book == "core":
            for t in out.get("market_by_name_core", {}):
                if t in per.index:
                    for k in WINDOWS:
                        out["market_by_name_core"][t][f"half_spread_{k}_bp_median"] = r(per.loc[t, f"hs_{k}"], 2)
    return out


ERAS = {"2016-19": ("2016-01-01", "2019-12-31"), "2020-22": ("2020-01-01", "2022-12-31"),
        "2023-26": ("2023-01-01", "2026-12-31")}
HALVES = {"early 2017-21": ("2017-01-01", "2021-12-31"), "late 2022-26": ("2022-01-01", "2026-12-31")}


def forecast_report(F: pd.DataFrame, ex_last: dict, n_checked: int) -> dict:
    ev = F.date >= "2017-01-01"
    models = ["naive", "med20", "med60", "ewm10", "reg_vol", "reg_cal", "reg_noex", "reg"]
    out = {"target": "log of the day's round-lot cross in USD (a day with no round-lot cross is -inf: below every quantile)",
           "cap_time": f"{CAP_LEAD_MIN} minutes before the close (15:45 on a full day)",
           "models": {"naive": "yesterday's cross", "med20": "median of the last 20 sessions",
                      "med60": "median of the last 60 sessions",
                      "ewm10": "EWMA (half-life 10 sessions) of log cross on round-lot days",
                      "reg_noex": "ewm10 + OLS on the same-day dollar volume to the cap time (vs its own EWMA), "
                                  "yesterday's day-volume surprise and calendar flags; refit monthly on past data",
                      "reg_vol": "ewm10 + the two volume terms of reg_noex only (short-term volume prediction)",
                      "reg_cal": "ewm10 + the calendar flags of reg_noex only",
                      "reg": "reg_noex + the name's ex-date and pre-ex-date flags (only where the distribution "
                             "panel covers the date)"},
           "quantiles": "pooled: the book's residuals over the 12 months before the month; pername: the name's last "
                        "250 sized residuals before the month",
           "evaluation": "every name-day from 2017-01-01 to the last session with a forecast",
           "truncation_check_rows": n_checked, "ex_date_panel_last": ex_last}
    for book in ("core", "c4"):
        mb = ev & (F.book == book)
        out[book] = {m: forecast_metrics(F, m, mb) for m in models}
        for split, spec in (("era", ERAS), ("half", HALVES)):
            out[book][f"reg_noex by {split}"] = {k: forecast_metrics(F, "reg_noex", mb & (F.date >= lo) & (F.date <= hi))
                                                 for k, (lo, hi) in spec.items()}
            for mdl in ("ewm10", "reg_vol", "reg_cal"):
                out[book][f"{mdl} by {split}"] = {k: forecast_metrics(F, mdl, mb & (F.date >= lo) & (F.date <= hi))
                                                  for k, (lo, hi) in spec.items()}
        out[book]["reg_noex by market-vol regime"] = {k: forecast_metrics(F, "reg_noex", mb & (F.regime == k))
                                                      for k in ("low", "high")}
        last60 = set(sorted(F.date.unique())[-60:])
        out[book]["last 60 sessions (the paper orders' window)"] = {
            m: forecast_metrics(F, m, mb & F.date.isin(last60)) for m in ("med20", "ewm10", "reg_noex")}
        out[book]["per-name coverage of q10, reg_noex pooled (min, p10, median, p90, max)"] = per_name_cover(F, "q10_reg_noex", mb)
        out[book]["per-name coverage of q10, reg_noex per-name (min, p10, median, p90, max)"] = per_name_cover(F, "q10pn_reg_noex", mb)
        out[book]["q10 by year: zero-cap share and breach on positive-cap days"] = {
            col: cover_by_year(F, col, mb) for col in ("q10_reg_noex", "q10pn_reg_noex", "q10_reg_cal", "q10pn_reg_cal",
                                                     "q10_ewm10")}
        # calendar flags, descriptive: residual of the EWMA on flagged against unflagged round-lot days
        G = F[mb & (F.kind == "round") & F.e_ewm10.notna() & np.isfinite(F.e_ewm10)]
        fl = {}
        for c in ["month_end", "quarter_end", "opex", "quad", "june_last_fri", "half_day", "pre_half_day",
                  "ex_date", "pre_ex"]:
            h = G[G[c].notna()]
            on, off = h[h[c] == 1], h[h[c] == 0]
            if len(on) >= 20:
                # a plain ratio beside the residual: per name, median round-lot cross USD on flagged
                # days over the median on unflagged days; the median of that across names
                per = h.groupby("ticker").apply(
                    lambda z: (z[z[c] == 1].usd.median() / z[z[c] == 0].usd.median())
                    if (z[c] == 1).sum() >= 3 else np.nan, include_groups=False).dropna()
                fl.setdefault(c, {})["median across names of (median flagged cross / median unflagged cross)"] = \
                    [r(per.median(), 3), int(len(per))]
                # the desk's estimator (scripts/research/inference.py): the difference of the two
                # group means of the residual, its SE clustered by date (name-days on one date share
                # that date's move; a date holding both groups enters once, with the covariance).
                # Replaces a private day-level t (independent review, 2026-11-24).
                from scripts.research.inference import clustered_mean_diff
                cm = clustered_mean_diff(h.e_ewm10.to_numpy(float), (h[c] == 1).to_numpy(), h.date.to_numpy())
                fl[c].update({"days": int(on.date.nunique()), "name_days": int(len(on)),
                              "mean residual flagged minus unflagged (log)": r(cm["diff"], 3),
                              "as a ratio": r(math.exp(cm["diff"]), 3),
                              "t of the difference, clustered by date (inference.clustered_mean_diff)":
                                  r(cm["diff"] / cm["se_diff"], 2),
                              "date clusters": int(cm["clusters"])})
        out[book]["calendar flags (residual of ewm10)"] = fl
        co = F.attrs["coefs"]
        lastm = max((m for (b, m) in co if b == book), default=None)
        out[book]["reg coefficients, last refit"] = {"month": lastm, **{k: r(v, 3) for k, v in co[(book, lastm)].items()}} if lastm else None
        # intraday volume: how much of the error it explains
        G2 = F[mb & (F.kind == "round") & F.u_am.notna() & F.e_ewm10.notna() & np.isfinite(F.e_ewm10)]
        out[book]["corr(ewm10 residual, same-day volume surprise to the cap time)"] = r(np.corrcoef(G2.e_ewm10, G2.u_am)[0, 1], 3)
    return out


def _day_mean(df, col, w) -> pd.Series:
    """Each day's weighted mean of `col` (orders on one day share the market's move, so the day,
    not the order, is the unit of independence)."""
    g = df[np.isfinite(df[col]) & np.isfinite(df[w])]
    if g.empty:
        return pd.Series(dtype=float)
    return g.groupby("date").apply(lambda h: float((h[col] * h[w]).sum() / h[w].sum()), include_groups=False).astype(float)


def _path_stats(df, col, w="w") -> dict:
    g = df[np.isfinite(df[col]) & np.isfinite(df[w])]
    if g.empty:
        return {"orders": 0}
    dm = _day_mean(g, col, w)
    sd = float(dm.std(ddof=1)) if len(dm) > 2 else float("nan")
    return {"orders": int(len(g)), "days": int(len(dm)), "pooled_bp": r((g[col] * g[w]).sum() / g[w].sum()),
            "day_mean_bp": r(dm.mean()), "day_t": tstat(dm),
            "ci95_bp": [r(dm.mean() - 1.96 * sd / math.sqrt(len(dm))), r(dm.mean() + 1.96 * sd / math.sqrt(len(dm)))]
            if len(dm) > 2 else None}


def path_report(O: pd.DataFrame, P: pd.DataFrame, X: pd.DataFrame, D5: pd.DataFrame, Q: pd.DataFrame,
                raw: pd.DataFrame) -> dict:
    """G = s (P_c - ref) / ref x 1e4: > 0 means working at ref would have gained against the close
    (before spread and impact). Weighted by notional at the cross price; day-clustered t."""
    out = {"definition": "G = s (closing print - reference) / reference, bp; > 0: working at the reference "
                         "gains against the close, before spread and impact. SIGNAL-CONDITIONAL: the orders' "
                         "direction is the discount signal's",
           "reference_price_of_the_close": "the primary's round-lot cross price (6), else its earliest official close (M)"}
    # the day's open and the previous session's close (the decide-time close t of an order filled at
    # t+1), from the raw SIP daily bars: context for where in the day the move sits
    dr = raw.sort_values(["ticker", "date"]).copy()
    dr["prev_c"] = dr.groupby("ticker").c.shift(1)
    Xc = X[["date", "ticker", "price", "kind"]].merge(dr[["ticker", "date", "o", "prev_c"]], on=["ticker", "date"], how="left")
    # paper orders: mids at fixed times and window VWAPs
    Op = O.merge(Xc, on=["date", "ticker"], how="left").merge(Q, on=["date", "ticker"], how="left").merge(
        D5[["date", "ticker"] + [f"vwap_{k}" for k in WINDOWS]], on=["date", "ticker"], how="left")
    if Op.price.isna().any():
        raise StudyError(f"paper orders with no closing print: {Op[Op.price.isna()][['date']].drop_duplicates().values.ravel().tolist()}")
    Op["w"] = Op.qty * Op.price
    for k in TAU_S:
        Op[f"G_mid_{k}"] = Op.s * (Op.price - Op[f"mid_{k}"]) / Op[f"mid_{k}"] * 1e4
        Op[f"G_touch_{k}"] = Op[f"G_mid_{k}"] - Op[f"hsq_{k}"]          # crossing at tau pays the half-spread
    for k in WINDOWS:
        Op[f"G_vwap_{k}"] = Op.s * (Op.price - Op[f"vwap_{k}"]) / Op[f"vwap_{k}"] * 1e4
    Op["G_open"] = Op.s * (Op.price - Op.o) / Op.o * 1e4
    Op["G_prev_close"] = Op.s * (Op.price - Op.prev_c) / Op.prev_c * 1e4
    pb = {}
    for lab, sel in (("all paper orders", Op.index == Op.index), ("late_market (2026-09-30 on)", Op.method == "late_market"),
                     ("cls (2026-09-29)", Op.method == "cls"), ("steady state (2026-10-01 on)", Op.phase == "steady state"),
                     ("buys", Op.s > 0), ("sells", Op.s < 0)):
        g = Op[sel]
        pb[lab] = {"mid at tau": {k: _path_stats(g, f"G_mid_{k}") for k in TAU_S},
                   "touch at tau (mid minus the half-spread)": {k: _path_stats(g, f"G_touch_{k}") for k in ("14:00", "15:00", "15:30", "15:45", "15:50", "15:52")},
                   "window VWAP": {WINDOW_LABEL[k]: _path_stats(g, f"G_vwap_{k}") for k in WINDOWS},
                   "the day's open (context)": _path_stats(g, "G_open"),
                   "the decide-time close, t (context)": _path_stats(g, "G_prev_close")}
    pb["missing mids at tau (order count)"] = {k: int(Op[f"mid_{k}"].isna().sum()) for k in TAU_S}
    out["paper_core"] = pb
    # the 2026-11-05 note's C2 at the send (15:52) reproduced in sign and size on the same orders
    # policy orders: window VWAPs only (no quotes before the last 100 sessions)
    Pp = P.merge(Xc, on=["date", "ticker"], how="left").merge(
        D5[["date", "ticker"] + [f"vwap_{k}" for k in WINDOWS]], on=["date", "ticker"], how="left")
    Pp["w"] = Pp.shares * Pp.price
    for k in WINDOWS:
        Pp[f"G_vwap_{k}"] = Pp.s * (Pp.price - Pp[f"vwap_{k}"]) / Pp[f"vwap_{k}"] * 1e4
    Pp["G_open"] = Pp.s * (Pp.price - Pp.o) / Pp.o * 1e4
    Pp["G_prev_close"] = Pp.s * (Pp.price - Pp.prev_c) / Pp.prev_c * 1e4
    po = {"orders": int(len(Pp)), "with a closing print": int(Pp.price.notna().sum())}
    for lab, sel in [("all", Pp.index == Pp.index)] + [(e, (Pp.date >= lo) & (Pp.date <= hi)) for e, (lo, hi) in ERAS.items()] + \
            [(h, (Pp.date >= lo) & (Pp.date <= hi)) for h, (lo, hi) in HALVES.items()] + [("buys", Pp.s > 0), ("sells", Pp.s < 0)]:
        po[lab] = {**{WINDOW_LABEL[k]: _path_stats(Pp[sel], f"G_vwap_{k}") for k in WINDOWS},
                   "the day's open (context)": _path_stats(Pp[sel], "G_open"),
                   "the decide-time close, t (context)": _path_stats(Pp[sel], "G_prev_close")}
    out["policy_model"] = po
    return out


def cap_report(O: pd.DataFrame, P: pd.DataFrame, X: pd.DataFrame, D5: pd.DataFrame, F: pd.DataFrame) -> dict:
    """MOC = min(order, c x the low-quantile forecast cross); the rest is worked in the afternoon."""
    fc = F[["date", "ticker", "q10_reg_noex", "q25_reg_noex", "q10pn_reg_noex", "q10_ewm10", "reg_noex",
            "q10_reg_cal", "q10pn_reg_cal", "q25_reg_cal"]]
    base = X[["date", "ticker", "price", "usd", "kind"]].merge(
        D5[["date", "ticker"] + [f"usd_{k}" for k in WINDOWS]], on=["date", "ticker"], how="left").merge(
        fc, on=["date", "ticker"], how="left")
    Op = O.merge(base, on=["date", "ticker"], how="left")
    Op["usd_ref"] = Op.qty * Op.price / Op.equity_prev          # order USD per USD of equity
    Pp = P.merge(base, on=["date", "ticker"], how="left")
    from scripts.cef.spec import CAPITAL_USD
    Pp["usd_ref"] = Pp.shares * Pp.price / CAPITAL_USD
    out = {"rule": "MOC USD = min(order USD, c x exp(low-quantile forecast of log cross USD)); worked = the rest, "
                   "in the afternoon continuous session. Orders scale linearly with book size; the cross and the "
                   "afternoon pool are the day's own, measured WITHOUT us (paper never entered the auction)",
           "timing": "the worked part must be sized before working starts (14:00 for the 14:00-15:50 window), so "
                     "only a PRE-OPEN forecast can size it: ewm10 and reg_cal use nothing from the day itself. "
                     "reg_noex uses the day's volume to 15:45: a split set then leaves 15:45-15:50 to work, so "
                     "its rows show what a 15:45 cap would send to the auction, not a feasible afternoon schedule",
           "forecasts": {"q10_reg_cal": "PRE-OPEN: 10th percentile, reg_cal (ewm10 + calendar flags), pooled residuals",
                         "q10pn_reg_cal": "PRE-OPEN: 10th percentile, reg_cal, the name's own residuals",
                         "q25_reg_cal": "PRE-OPEN: 25th percentile, reg_cal, pooled",
                         "q10_ewm10": "PRE-OPEN: 10th percentile, ewm10, pooled",
                         "q10_reg_noex": "AT 15:45: 10th percentile, reg_noex, pooled residuals",
                         "q25_reg_noex": "AT 15:45: 25th percentile, reg_noex, pooled",
                         "q10pn_reg_noex": "AT 15:45: 10th percentile, reg_noex, the name's own residuals",
                         "reg_noex": "AT 15:45: the point forecast itself (about the median), for comparison: "
                                     "not a low quantile"}}
    last60 = sorted(F.date.unique())[-60:]
    capn = {}
    for book in ("core", "c4"):
        h = F[(F.book == book) & F.date.isin(last60)]
        per = h.groupby("ticker").agg(q10_pre=("q10_reg_cal", lambda v: float(np.exp(v).median())),
                                      q10=("q10_reg_noex", lambda v: float(np.exp(v).median())),
                                      q25=("q25_reg_noex", lambda v: float(np.exp(v).median())),
                                      point=("reg_noex", lambda v: float(np.exp(v).median())),
                                      zero_q10_days=("q10_reg_noex", lambda v: float(np.isneginf(v).mean())))
        capn[book] = {"names": int(len(per)), "sessions": len(last60),
                      "q10_pre": "the pre-open q10 (reg_cal); q10, q25, point: the 15:45 forecast (reg_noex)",
                      **{f"MOC USD a name-day at c={c:.2f} x {b}, across names (min, p25, median, p75, max)":
                         pct5(c * per[b], 0) for b in ("q10_pre", "q10", "q25", "point") for c in CAPS},
                      "share of the name's days on which q10 is zero (a no-round-lot day is within its "
                      "lowest decile), across names": pct5(per.zero_q10_days, 3)}
    out["MOC capacity per name-day, last 60 sessions (forecast only, no orders)"] = capn
    for lab, D, sel in (("paper_core_steady_state", Op, Op.phase == "steady state"),
                        ("paper_core_all", Op, Op.index == Op.index),
                        ("policy_model_2016_on", Pp, Pp.index == Pp.index),
                        ("policy_model_2023_on", Pp, Pp.date >= "2023-01-01")):
        g = D[sel & D.price.notna()].copy()
        blk = {"orders": int(len(g)), "days": int(g.date.nunique()),
               "orders on a day with no round-lot cross": int((g.kind == "no_round").sum()),
               "orders with no forecast (dropped from the cap rows): 15:45 / pre-open": [
                   int(g.q10_reg_noex.isna().sum()), int(g.q10_reg_cal.isna().sum())]}
        for B in BOOK_SIZES:
            g["o"] = g.usd_ref * B
            ok = g[g.kind.isin(["round", "no_round"])]
            part = np.where(ok.usd > 0, ok.o / ok.usd, np.inf)
            sb = {"no cap: order / cross (median, p90, notional-weighted median)":
                  [r(qinf(part, 0.5), 3), r(qinf(part, 0.9), 3), r(weighted_median(np.where(np.isfinite(part), part, 1e9), ok.o), 3)],
                  "no cap: share of notional in orders above 10% / 25% / 100% of the cross":
                  [r(ok.o[part > x].sum() / ok.o.sum(), 3) for x in (0.10, 0.25, 1.0)],
                  "median order USD": r(g.o.median(), 0), "p90 order USD": r(g.o.quantile(0.9), 0)}
            for fcol in ("q10_reg_cal", "q10pn_reg_cal", "q25_reg_cal", "q10_ewm10",
                         "q10_reg_noex", "q25_reg_noex", "q10pn_reg_noex", "reg_noex"):
                h = ok[ok[fcol].notna()].copy()
                if h.empty:
                    continue
                for c in CAPS:
                    moc, rest = cap_split(h.o.to_numpy(), np.exp(h[fcol].to_numpy()), c)
                    h["moc"], h["rest"] = moc, rest
                    mp = np.where(h.usd > 0, h.moc / h.usd, np.where(h.moc > 0, np.inf, 0.0))
                    row = {"share of notional MOC": r(h.moc.sum() / h.o.sum(), 3),
                           "share of notional worked": r(h.rest.sum() / h.o.sum(), 3),
                           "orders with some worked": int((h.rest > 0).sum()),
                           "realised MOC / cross: median, p90, share above c, share above 2c": [
                               r(qinf(mp[h.moc > 0], 0.5), 3) if (h.moc > 0).any() else None,
                               r(qinf(mp[h.moc > 0], 0.9), 3) if (h.moc > 0).any() else None,
                               r((mp[h.moc > 0] > c).mean(), 3) if (h.moc > 0).any() else None,
                               r((mp[h.moc > 0] > 2 * c).mean(), 3) if (h.moc > 0).any() else None]}
                    wk = h[h.rest > 0]
                    for k in WINDOWS:
                        pw = np.where(wk[f"usd_{k}"] > 0, wk.rest / wk[f"usd_{k}"], np.inf)
                        row[f"worked / {WINDOW_LABEL[k]} $ volume: median, p90, worked-weighted median"] = [
                            r(qinf(pw, 0.5), 3), r(qinf(pw, 0.9), 3),
                            r(weighted_median(np.where(np.isfinite(pw), pw, 1e9), wk.rest), 3)] if len(wk) else None
                        row[f"share of worked notional above 5% / 10% of {WINDOW_LABEL[k]}"] = [
                            r(wk.rest[pw > x].sum() / wk.rest.sum(), 3) for x in PARTICIPATION] if len(wk) else None
                    sb[f"{fcol}, c={c:.2f}"] = row
            blk[f"${B:,.0f}"] = sb
        out[lab] = blk
    return out


def power_report(X: pd.DataFrame, Q: pd.DataFrame, O: pd.DataFrame, U: pd.DataFrame, sessions: list[str]) -> dict:
    """How many live MOC orders it takes to see our own impact, using the paper record as the
    impact-free baseline of the same signal's orders (a paper order never entered the auction)."""
    out = {}
    M = X[["date", "ticker", "price", "kind", "book"]].merge(Q, on=["date", "ticker"], how="inner")
    M = M[M.kind == "round"]
    for k in ("15:50", "15:59:50"):
        M[f"d_{k}"] = (M.price - M[f"mid_{k}"]) / M[f"mid_{k}"] * 1e4
    dense = M[M.date.isin(set(sessions[-SPREAD_RECENT:]))]
    for book in ("core", "c4"):
        h = dense[dense.book == book]
        blk = {"name_days": int(len(h)), "sessions": int(h.date.nunique())}
        for k in ("15:50", "15:59:50"):
            d = h[f"d_{k}"].dropna()
            dd = d - h.loc[d.index].groupby("date")[f"d_{k}"].transform("mean")
            blk[f"print vs mid at {k}: mean, sd, mean |d|, day-demeaned sd (bp)"] = [r(d.mean()), r(d.std()), r(d.abs().mean()), r(dd.std())]
        out[f"unsigned, market ({book})"] = blk
    Op = O[O.method == "late_market"].merge(M[["date", "ticker", "price", "d_15:50", "d_15:59:50"]], on=["date", "ticker"], how="inner")
    Op["w"] = Op.qty * Op.price
    sig = {}
    for k in ("15:50", "15:59:50"):
        Op[f"sd_{k}"] = Op.s * Op[f"d_{k}"]
        st = _path_stats(Op, f"sd_{k}")
        dm = _day_mean(Op[np.isfinite(Op[f"sd_{k}"])], f"sd_{k}", "w")
        if len(dm) < 3:
            sig[k] = {**st, "note": f"{len(dm)} day(s) with quotes: too few for a standard error"}
            continue
        sdd = float(dm.std(ddof=1))
        n_paper = len(dm)
        # Sessions of live MOC orders needed to detect an impact I on the day mean of s (print - mid),
        # live against the paper record of the same signal (a two-sample difference of day means, day
        # sd sdd on both sides), at 80% power and 5% two-sided: the desk's convention
        # (scripts/research/inference.py: mde's multiplier z_{1-a/2} + z_power, checked with
        # power_of_z). SE(diff) = sdd sqrt(1/N + 1/n_paper) must be <= I / k. An earlier version asked
        # only for an expected t of 2, which is about 50% power (independent review, 2026-11-24).
        need, need_inf = {}, {}
        for I in (2.0, 5.0, 10.0):
            N = live_sessions_needed(I, sdd, n_paper)
            need[f"{I:g}bp"] = N if N is not None else "not reachable with this paper baseline"
            need_inf[f"{I:g}bp"] = live_sessions_needed(I, sdd, None)
        sig[k] = {**st, "day sd bp": r(sdd), "paper baseline sessions": n_paper,
                  "live sessions for 80% power at 5% two-sided, against this paper baseline": need,
                  "the same against an unlimited paper baseline": need_inf}
    out["signed by our side, paper orders (late_market): s (print - mid) / mid"] = sig
    ss = O[O.phase == "steady state"].groupby("date").size()
    out["paper orders per session, steady state: median, mean"] = [r(ss.median(), 1), r(ss.mean(), 1)]
    return out


def live_sessions_needed(impact_bp: float, day_sd_bp: float, n_baseline: int | None,
                         power: float = 0.8, alpha: float = 0.05) -> int | None:
    """Live sessions N so that a difference of day means (live minus a baseline of n_baseline days,
    the same day sd on both sides) detects `impact_bp` with `power` at `alpha` two-sided:
    SE = sd sqrt(1/N + 1/n_baseline) <= impact / k, k = z_{1-a/2} + z_power from
    scripts/research/inference.py (`mde` at sigma = n = 1), checked with `power_of_z`. None when the
    baseline alone is too noisy for any N; n_baseline None means an unlimited baseline."""
    from scripts.research.inference import mde, power_of_z
    if impact_bp <= 0 or day_sd_bp <= 0:
        raise ValueError("impact and sd must be positive")
    k = mde(1.0, 1.0, alpha=alpha, power=power, two_sided=True)
    room = (impact_bp / k) ** 2 - (0.0 if n_baseline is None else day_sd_bp ** 2 / n_baseline)
    if room <= 0:
        return None
    N = int(math.ceil(day_sd_bp ** 2 / room))
    se = day_sd_bp * math.sqrt(1 / N + (0.0 if n_baseline is None else 1 / n_baseline))
    if power_of_z(impact_bp / se, alpha=alpha, two_sided=True) < power - 1e-9:
        raise StudyError("the power check on the computed N failed")
    return N


def check_aggregates(out: dict, symbols: list[str], orders: list[dict]) -> None:
    """The JSON of record is public: symbols only inside the two market_by_name_core blocks (public
    market data), and no order id, client order id or account figure anywhere."""
    pruned = json.loads(json.dumps(out, default=str))
    for blk in ("auction_size", "afternoon_pool"):
        pruned.get(blk, {}).pop("market_by_name_core", None)
    rest = json.dumps(pruned)
    for s in symbols:
        if f'"{s}"' in rest or f" {s} " in rest or f'"{s} ' in rest or f' {s}"' in rest:
            raise StudyError(f"symbol {s} appears outside the market_by_name_core blocks")
    text = json.dumps(out, default=str)
    for o in orders:
        for k in ("id", "client_order_id", "asset_id"):
            if o.get(k) and o[k] in text:
                raise StudyError(f"an order's {k} appears in the aggregates")
    if '"equity_prev"' in text or '"equity"' in text:
        raise StudyError("an equity field appears in the aggregates")


# -------------------------------------------------------------------------------- the report

def analyse(a) -> int:
    cache, priv = a.cache_dir, a.private_dir
    _refuse(priv, ["by_name_c4.csv", "splits.csv", "listing.csv"])
    from scripts.cef.live_attribution import _refuse_data_dir
    _refuse_data_dir(a.aggregates)
    sessions, closes, fetched = load_calendar(cache)
    last = sessions[-1]
    U = universe()
    codes = primary_codes(U, json.loads((cache / "exchanges.json").read_text()))
    A = pd.concat([pd.read_parquet(f) for f in sorted(cache.glob("auctions_close_*.parquet"))], ignore_index=True)
    if A.duplicated(["date", "ticker", "x", "c", "t", "s", "p"]).any():
        raise StudyError("a closing print appears twice in the fetched years")
    raw, spl = load_daily(cache, last, fetched)
    out: dict = {"meta": {"measured": dt.datetime.now(UTC).isoformat(timespec="seconds"),
                          "last_complete_session": last, "first_session": sessions[0],
                          "sessions": len(sessions),
                          "half_days": [d for d in sessions if closes[d] != "16:00"],
                          "names": {"core": int((U.book == "core").sum()), "c4": int((U.book == "c4").sum())},
                          "c4_listing_venues": U[U.book == "c4"].exchange.value_counts().to_dict(),
                          "source": "Alpaca SIP: /v2/stocks/auctions, daily and 5-minute bars, NBBO quotes, "
                                    "primary trades; the cef paper account's orders and EOD equity (GET only)"}}
    # ---------------------------------------------------------------- data checks
    X = cross_panel(A, codes, raw, U, sessions)
    sp = split_events(raw[raw.ticker.isin(codes)], spl[spl.ticker.isin(codes)])
    sp.to_csv(priv / "splits.csv", index=False)
    listing = (X[X.other_listing_6].groupby(["ticker", "kind"]).date.agg(["count", "min", "max"]).reset_index())
    listing.to_csv(priv / "listing.csv", index=False)
    rd = X[X.kind == "round"]
    eq_close = (rd.price - rd.day_c).abs() < 0.005 + 1e-9
    zero_vol = X[(X.day_v == 0)]
    out["checks"] = {
        "name_days_by_kind": X.kind.value_counts().to_dict(),
        "name_days_by_kind_and_book": X.groupby(["book", "kind"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "round_cross_on_another_listing_venue_while_primary_has_no_6": int(
            (X.other_listing_6 & ~X.kind.isin(["round"])).sum()),
        "names_with_such_days": int(X[X.other_listing_6 & ~X.kind.isin(["round"])].ticker.nunique()),
        "splits_detected (raw/split-adjusted daily close steps)": int(len(sp)),
        "names_with_a_split": int(sp.ticker.nunique()),
        "round_cross_price_within_half_a_cent_of_daily_bar_close_share": r(eq_close.mean(), 4),
        "zero_volume_daily_bars": int(len(zero_vol)),
        "round_cross_with_zero_size": int((rd.shares <= 0).sum()),
        "multi6_name_days": int((X.kind == "multi6").sum()),
        "data_gap_sessions (fewer than 20% of traded names with a primary print)": X.attrs["gap_days"],
        "no_round days that carry no M (before the venue's M era)": int(X.noM.sum()),
        "M era start by listing venue (first M print from its codes)": X.attrs["m_start"],
        "M-era sessions with no M for any of the venue's names (their no-6 name-days are missing_M_session)": {
            v: {"sessions": len(d), "first": d[0] if d else None, "last": d[-1] if d else None}
            for v, d in X.attrs["missing_M_sessions"].items()},
        "Nasdaq listings: round crosses printed on T only / on T and Q (equal, counted once) / on Q only": [
            int(((X.venue == "Q") & (X.kind == "round") & (X.six_venues == "T")).sum()),
            int(((X.venue == "Q") & (X.kind == "round") & (X.six_venues == "QT")).sum()),
            int(((X.venue == "Q") & (X.kind == "round") & (X.six_venues == "Q")).sum())],
    }
    print(json.dumps(out["checks"], indent=1, default=str), flush=True)
    # tape split of no_round days (odd lot or none) on the last TAPE_SESSIONS sessions
    from scripts.cef.exec_capacity_open_cross import classify_cross
    tape_kinds = {}
    for f in sorted((cache / "tape").glob("*.json")):
        rec = json.loads(f.read_text())
        d = rec["date"]
        g = A[(A.date == d)]
        for s, trades in rec["names"].items():
            # classify_cross reads the NYSE code; the name's own primary codes are mapped onto it
            rows = [dict(x="N", c=x.c, p=x.p, s=x.s, t=x.t) for x in g[(g.ticker == s) & g.x.isin(codes[s])].itertuples()]
            ms = sorted((rr for rr in rows if rr["c"] == "M"), key=lambda rr: pd.Timestamp(rr["t"]))
            # several primary M prices (SCD 2026-11-20): the earliest is the one at the auction
            rows = [rr for rr in rows if rr["c"] != "M"] + ms[:1]
            res = classify_cross("close", rows, trades, close_ts(dt.date.fromisoformat(d), closes[d]), cov=rec["cov"][s])
            tape_kinds[res["kind"]] = tape_kinds.get(res["kind"], 0) + 1
    out["checks"]["no_round_days_split_by_the_primary_tape (last sessions read)"] = {
        "sessions": TAPE_SESSIONS, "kinds": tape_kinds,
        "note": "odd = an odd-lot cross; none = no primary trade at the official close; unknown = not covered"}
    # ---------------------------------------------------------------- 1. auction size
    out["auction_size"] = auction_size(X, sessions, U)
    # ---------------------------------------------------------------- 3. afternoon pool (bars) and spreads (quotes)
    D5 = bars5_daily(cache, closes, sessions)
    Q, qmeta = load_quotes(cache, sessions)
    out["afternoon_pool"] = afternoon_pool(X, D5, Q, qmeta, sessions, U)
    out["afternoon_pool"]["check: 5-minute bars against raw SIP trades (core, last sessions)"] = tradecheck_report(cache, D5)
    # C4's per-name market table, last 60 sessions (private: the JSON carries C4 only across names)
    last60 = set(sessions[-60:])
    c4n = sorted(U[U.book == "c4"].ticker)
    T4 = pd.DataFrame({t: name_medians(X[(X.ticker == t) & X.date.isin(last60)]) for t in c4n}).T
    pool = D5[D5.date.isin(last60) & D5.ticker.isin(c4n)].groupby("ticker")[[f"usd_{k}" for k in WINDOWS]].median()
    hs = Q[Q.date.isin(set(sessions[-SPREAD_RECENT:])) & Q.ticker.isin(c4n)].groupby("ticker")[
        [f"hs_{k}" for k in WINDOWS]].median()
    T4.join(pool).join(hs).to_csv(priv / "by_name_c4.csv")
    out["checks"]["the 6 print against the tape, C4's NYSE American and Nasdaq listings"] = tapecheck_report(cache, X)
    # ---------------------------------------------------------------- 2. forecast
    cal = calendar_flags([dt.date.fromisoformat(d) for d in sessions],
                         {dt.date.fromisoformat(d): closes[d] for d in sessions})
    cal.index = [d.isoformat() for d in cal.index]
    cal.index.name = "date"
    Ev, ex_last = distributions(a.cef_panels, U)
    F = build_features(X, D5, cal, Ev, ex_last)
    n_checked = truncation_check(F, X, D5, cal, Ev, ex_last)
    print(f"truncation check: {n_checked} rows recomputed on data cut at D, no difference", flush=True)
    F = fit_and_forecast(F)
    reg = regime_labels(raw, sessions)
    F["regime"] = F.date.map(reg)
    out["forecast"] = forecast_report(F, ex_last, n_checked)
    # ---------------------------------------------------------------- orders, path, cap
    O = paper_orders(priv, closes, sessions)
    P, pmeta = policy_orders(a.cef_panels)
    out["orders"] = {"paper_core": {"orders": int(len(O)), "sessions": int(O.date.nunique()),
                                    "first": O.date.min(), "last": O.date.max(),
                                    "by_method": O.groupby("method").agg(orders=("ticker", "size"),
                                                                         sessions=("date", "nunique")).to_dict(orient="index"),
                                    "cls_orders_status": O[O.method == "cls"].status.value_counts().to_dict()},
                     "policy_model": pmeta,
                     "c4": "not read: no C4 key in the laptop's key file and no copy of C4's records here"}
    out["path"] = path_report(O, P, X, D5, Q, raw)
    out["cap"] = cap_report(O, P, X, D5, F)
    out["real_money_measurement"] = power_report(X, Q, O, U, sessions)
    # ---------------------------------------------------------------- write
    check_aggregates(out, sorted(set(U.ticker)), json.loads((priv / "orders_cef.json").read_text())["orders"])
    a.aggregates.parent.mkdir(parents=True, exist_ok=True)
    a.aggregates.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"wrote {a.aggregates}", flush=True)
    return 0

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch")
    f.add_argument("--cache-dir", type=Path, required=True)
    f.add_argument("--private-dir", type=Path, required=True)
    f.add_argument("--refresh-orders", action="store_true")
    f.add_argument("--reverse", action="store_true", help="with --only quotes: newest session first")
    f.add_argument("--only", choices=("all", "quotes", "orders", "tradecheck"), default="all",
                   help="quotes: only the quote sample (step 4); orders: only the account reads (step 6); "
                        "either can run beside a running full fetch")
    g = sp.add_parser("analyse")
    g.add_argument("--cache-dir", type=Path, required=True)
    g.add_argument("--private-dir", type=Path, required=True)
    g.add_argument("--cef-panels", type=Path, required=True,
                   help="directory with cef_prices/cef_nav/cef_distributions/cef_splits parquet (read only)")
    g.add_argument("--aggregates", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        return fetch(a)
    return analyse(a)


if __name__ == "__main__":
    raise SystemExit(main())
