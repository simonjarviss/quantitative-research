"""Net-flow engine: per stock, the net passive demand from every index that trades a given close.

    python3 engine.py --close 2026-12-18 [--status best|predicted|announced|final] [--include-us] [--md live|none]
                      [--label LABEL] [--md-from MD_OR_SHEET.csv] [--files A.csv,B.csv] [--notes-from OLD.md]

Reads inputs/<close>/*.csv (schema in inputs.py / README.md), pi per index (pi.py: pi_overrides.csv >
../12_studies/net_flow/pi_calibration.csv > flagged fallback), market data (marketdata.py, or frozen with --md-from)
and the pricing layer v2 (2026-10-15; README §4, §7b): close-pricing rows from close_dislocation/table_pricing.csv
(2016+), table_pricing_post2021.csv (S&P/TSX Oct2021+) and msci_close/table_pricing_msci.csv (MSCI families), the
index-futures hedge gap (12_studies/expiry_basis/), venue close rules (venue_rules.csv) and a depth stress (SIX/HSG
prior). Writes:
  sheets/<close>_<label>.csv            one row per stock (netting key = ISIN; country:ticker when no ISIN)
  sheets/<close>_<label>_contrib.csv    one row per stock x index (the contributions)
  sheets/<close>_<label>_md.csv         the market data used, full precision (re-price later with --md-from)
  sheets/<close>_<label>.md             summary: inputs, pi, notional per market, largest imbalances, offsets,
                                        pricing summary, hedge gap, close timing, depth stress

Demand per index j and stock i (flow_sizing.md §3):
  dF_ij   = new_index_shares - old_index_shares            (or (w_new - w_old) * index_mcap / price)
  D_vis   = pi_visible_j * dF_ij                            visible trackers only (lower bound)
  D       = pi_true_j * dF_ij,  pi_true = R_j * pi_visible  central estimate (volume-implied: unseen passive +
                                                            benchmarked active + LP round trips; upper-ish bound)
  E       = prob * D                                        probability-weighted (predicted rows)
Net = sum over indices trading this close; gross = sum of |D|. US index rows are used only with --include-us
(interlisted Canadian names: same ISIN, so US-line demand nets against the TSX line at the same 16:00 ET close).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import inputs as I
from common import (DEPTH_BP_PER_PCT, DEPTH_BP_PER_PCT_RANGE, ET_TZ, EXPIRY_BASIS, FUT_INDEX, HEDGE_FUTURE,
                    HEDGE_FUTURE_EU_DEFAULT, HEDGE_FUTURE_UK_MID, HEDGE_REGIME_START, HOLD, INDEX, MSCI_EVENT_GROUP,
                    MSCI_PERIOD, MSCI_REVIEW_CLOSES, MSCI_REVIEWS, MSCI_REVIEWS_SOURCE, MSCI_SUBROW_MIN_N, PHI,
                    PRICING_LAMBDA, PRICING_MSCI, PRICING_TABLE, PRICING_TSX_POST2021, SCALE_TABLE_START, SHEETS,
                    STALE_BP, STUDIES, SXF_MONTH_END_START, TSX_MIN_N, TSX_PERIOD, US_INDICES, VENUE_ET_REF_DATE,
                    VENUE_RULES, fmt_bn, region_of_country, sessions)
from pi import pi_for


def _fmt(v, spec: str, na: str = "n/a") -> str:
    """format(v, spec), or `na` when v is missing or not finite (no 'nan' in notes or the .md)."""
    try:
        ok = v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        ok = False
    return format(float(v), spec) if ok else na


# ------------------------------------------------------------------------------------------- selection
def select_status(rows: pd.DataFrame, status: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["group"] = np.where(rows.change_type.isin(I.MEMBERSHIP), "membership", "weights")
    rows["srank"] = rows.status.map(I.STATUS_RANK)
    if status == "best":
        top = rows.groupby(["index", "group"]).srank.transform("max")
        return rows[rows.srank == top].drop(columns=["srank"])
    return rows[rows.status == status].drop(columns=["srank"])


def key_of(r) -> str:
    if isinstance(r.isin, str) and len(r.isin) >= 12:
        return r.isin
    return f"{r.country}:{r.ticker}"


# ---------------------------------------------------------------------------------------------- pricing
# Pricing layer v2 (2026-10-15). PRICING_COLS_V1 are the pricing columns of the v1 sheets; every other v1 column is
# non-pricing and must not change when the pricing layer changes (validation/pricing_v2_regression.md).
PRICING_COLS_V1 = ["pricing_row", "pricing_n", "E_T_dislocation_bp", "E_reversal_T1open_bp", "sd_T1open_name_bp",
                   "quote_hold_T1open_bp", "quote_hold_T5_bp", "pricing_note"]
VENUE_COLS = ["venue", "close_call_local", "normal_print_et", "latest_print_et", "post_close_session",
              "hidden_imbalance_order"]
PRICING_COLS_V2 = (["pricing_row", "pricing_n", "pricing_family", "pricing_group", "pricing_period", "pricing_source",
                    "E_T_dislocation_bp", "E_T_t", "E_reversal_T1open_bp", "rev_T1open_t", "sd_T1open_name_bp",
                    "quote_hold_T1open_bp", "quote_adj_hold_T1open_bp", "E_reversal_T1close_bp", "rev_T1close_t",
                    "sd_T1close_name_bp", "quote_hold_T1close_bp", "quote_adj_hold_T1close_bp", "quote_hold_T5_bp",
                    "quote_adj_hold_T5_bp", "pricing_note", "uk_index", "hedge_future", "hedge_index",
                    "hedge_settles", "hedge_day_type", "hedge_gap_sd_bp", "hedge_gap_basis"] + VENUE_COLS +
                   ["close_one_sided_share", "depth_stress_bp"])
SETTLE_LABEL = {"before the cash close": "before close", "near the close": "near close", "on the close": "on close"}


def pricing_table() -> pd.DataFrame:
    """Every pricing row the engine can use, flow_tercile = crowding = 'all', with the file it came from (`source`).

    close_dislocation/table_pricing.csv       all v1 families, periods 2016+ and all
    close_dislocation/table_pricing_post2021  CA S&P/TSX, period Oct2021+
    msci_close/table_pricing_msci.csv         CA MSCI, EU MSCI (with event-type sub-rows), EU MSCI UK / ex-UK
    The lambda-only quotes are the files' own (d0 + 0.25 x SD of the hold). quote_adj_<hold> adds the expected loss
    when the price keeps moving with demand after the close: + max(0, -E[reversal to the exit]) (msci_close RESULTS §6),
    for every row and hold; for the MSCI rows this equals the file's quote_adj_name_bp_* columns. msci_close §6
    prescribes it for the MSCI rows (and the close_dislocation addendum for the post-2021 S&P/TSX rows); applying it
    to the STOXX 600, DAX-family and Russell rows is the engine's own extension (I)."""
    frames = []
    for path in (PRICING_TABLE, PRICING_TSX_POST2021, PRICING_MSCI):
        if path.exists():
            frames.append(pd.read_csv(path).assign(source=path.relative_to(STUDIES).as_posix()))
    t = pd.concat(frames, ignore_index=True, sort=False)
    t = t[(t.flow_tercile == "all") & (t.crowding == "all")].copy()
    for hold, rev in (("hold_open", "rev_on1_mean_bp"), ("hold1", "rev_r1_mean_bp"), ("hold5", "rev_r1_5_mean_bp")):
        t[f"quote_adj_{hold}"] = t[f"quote_name_bp_{hold}"] + np.maximum(0.0, -t[rev])
    return t.reset_index(drop=True)


def _pick(tab: pd.DataFrame, fam: str, grp: str, period: str):
    x = tab[(tab.family == fam) & (tab.group == grp) & (tab.period == period)]
    return x.iloc[0] if len(x) else None


def _priced(x, fam: str, grp: str, period: str, notes: list[str]) -> dict:
    return dict(pricing_row=f"{fam} / {grp.replace(' | ', ', ')} ({period})", pricing_n=int(x.n), pricing_family=fam,
                pricing_group=grp, pricing_period=period, pricing_source=x.source,
                E_T_dislocation_bp=x.d0_mean_bp, E_T_t=x.d0_t,
                E_reversal_T1open_bp=x.rev_on1_mean_bp, rev_T1open_t=x.rev_on1_t,
                sd_T1open_name_bp=x.rev_on1_sd_name_bp, quote_hold_T1open_bp=x.quote_name_bp_hold_open,
                quote_adj_hold_T1open_bp=x.quote_adj_hold_open,
                E_reversal_T1close_bp=x.rev_r1_mean_bp, rev_T1close_t=x.rev_r1_t,
                sd_T1close_name_bp=x.rev_r1_sd_name_bp, quote_hold_T1close_bp=x.quote_name_bp_hold1,
                quote_adj_hold_T1close_bp=x.quote_adj_hold1,
                quote_hold_T5_bp=x.quote_name_bp_hold5, quote_adj_hold_T5_bp=x.quote_adj_hold5,
                pricing_note="; ".join(n for n in notes if n))


def pricing_row(tab: pd.DataFrame, region: str, country: str, side: int, dom_index: str, dom_type: str,
                price_usd: float) -> dict:
    """Close-pricing row for a net buy (side=+1) or sell (-1); family and group follow the dominant index and type.

    MSCI_CANADA / MSCI_EUROPE: 'CA MSCI' / 'EU MSCI', MSCI_PERIOD; event-type sub-row (migration_in -> up-migration,
    migration_out -> down-migration, add -> direct add, delete -> direct delete) when it has n >= MSCI_SUBROW_MIN_N,
    else the pooled 'net buy' / 'net sell' row with a FLAG note; v1 proxy rows only if the MSCI rows are missing
    (FALLBACK note). Other Canadian rows: 'CA S&P/TSX' TSX_PERIOD if its row has n >= TSX_MIN_N, else 2016+.
    Everything else: the v1 close_dislocation 2016+ rows."""
    meta = INDEX.get(dom_index, {})
    fam = meta.get("pricing")
    notes: list[str] = []
    if fam in ("CA MSCI", "EU MSCI"):
        base = "net buy" if side > 0 else "net sell"
        ev = MSCI_EVENT_GROUP.get(dom_type)
        same_side = ev in (("direct add", "up-migration") if side > 0 else ("direct delete", "down-migration"))
        x, grp = None, base
        if ev and same_side:
            sub = _pick(tab, fam, ev, MSCI_PERIOD)
            if sub is not None and sub.n >= MSCI_SUBROW_MIN_N and np.isfinite(sub.quote_name_bp_hold_open):
                x, grp = sub, ev
            else:
                msg = (f"FLAG: {fam} {ev} sub-row has n {0 if sub is None else int(sub.n)} < {MSCI_SUBROW_MIN_N} "
                       f"({MSCI_PERIOD}), priced off the pooled {base} row")
                alt = _pick(tab, fam, ev, "all")
                if alt is not None and np.isfinite(alt.quote_adj_hold_open):
                    msg += (f"; all-period sub-row (flag only) n {int(alt.n)}: E_T {_fmt(alt.d0_mean_bp, '+.0f')} bp, "
                            f"quote_adj to T+1 open {_fmt(alt.quote_adj_hold_open, '+.0f')}, to T+1 close "
                            f"{_fmt(alt.quote_adj_hold1, '+.0f')}")
                notes.append(msg)
        elif ev:
            notes.append(f"FLAG: {dom_type} on a net {'buy' if side > 0 else 'sell'}, priced off the pooled {base} row")
        else:
            notes.append(f"FLAG: no MSCI event-type row for {dom_type}, priced off the pooled {base} row")
        if x is None:
            x = _pick(tab, fam, base, MSCI_PERIOD)
        if x is not None:
            if country == "GB":
                uk = _pick(tab, "EU MSCI UK", base, MSCI_PERIOD)
                if uk is not None:
                    notes.append(f"UK split, note only (msci_close RESULTS §6): EU MSCI UK / {base} ({MSCI_PERIOD}) "
                                 f"n {int(uk.n)}: E_T {_fmt(uk.d0_mean_bp, '+.0f')} bp, quote_adj to T+1 open "
                                 f"{_fmt(uk.quote_adj_hold_open, '+.0f')}, to T+1 close "
                                 f"{_fmt(uk.quote_adj_hold1, '+.0f')}")
            return _priced(x, fam, grp, MSCI_PERIOD, notes)
        proxy = "CA S&P/TSX" if region == "CA" else meta.get("pricing_proxy") or "EU STOXX 600"
        notes = [f"FALLBACK: no {fam} {MSCI_PERIOD} row in {PRICING_MSCI.name}; priced off the {proxy} proxy row "
                 "(pricing_row)"]
        msci_fallback = True
        fam = meta.get("pricing_proxy")
    else:
        msci_fallback = False
    if region == "CA":
        fam = "CA S&P/TSX"
    if fam is None:
        fam = "EU STOXX 600" if region == "EU" else "CA S&P/TSX"
    if fam == "EU DAX family":
        grp = "net buy" if side > 0 else "net sell"
    elif fam == "US Russell":
        grp = ("add" if side > 0 else "delete") + (" | price >= $5" if price_usd >= 5 else " | price < $5")
    else:
        grp = "add" if side > 0 else "delete"
    period = "2016+"
    x = None
    if fam == "CA S&P/TSX":
        y = _pick(tab, fam, grp, TSX_PERIOD)
        if y is not None and y.n >= TSX_MIN_N:
            x, period = y, TSX_PERIOD
        else:
            notes.append(f"FALLBACK: CA S&P/TSX {TSX_PERIOD} {grp} row missing or n < {TSX_MIN_N}, priced off the "
                         "2016+ row")
    if x is None:
        x = _pick(tab, fam, grp, "2016+")
    if x is None:
        return {}
    exact = dom_type in ("add", "delete", "migration_in", "migration_out") and (
        (fam == "CA S&P/TSX" and dom_index == "SPTSX_COMP") or (fam == "EU STOXX 600" and dom_index == "STOXX600")
        or (fam == "EU DAX family" and dom_index in ("DAX", "MDAX", "SDAX", "TECDAX")) or fam == "US Russell")
    if not exact and not msci_fallback:
        notes.append(f"proxy row: no study row for {dom_index} {dom_type}")
    if country == "GB":
        notes.append(UK_NOTE_FTSE if dom_index in ("FTSE100", "FTSE250") else UK_NOTE_STOXX)
    return _priced(x, fam, grp, period, notes)


# UK notes on the close_dislocation rows (note only; the quote is the row's). close_dislocation RESULTS §1, §7 (Q):
# STOXX 600 UK events, adds -132 bp (t -3.3, n 194) / deletes -120 bp (t -2.6, n 200) on T; of these, UK names with a
# FTSE 100/250 change at the same review: adds +128 bp (t 1.6, n 6) / deletes +111 bp (t 0.3, n 5); without one:
# adds -140 / deletes -126 bp. GB rows led by FTSE 100/250 get the FTSE-change subgroup, every other GB row the STOXX UK
# means.
UK_NOTE_STOXX = "UK: STOXX UK adds -132 bp / deletes -120 bp on T (close_dislocation RESULTS §1, §7)"
UK_NOTE_FTSE = ("UK, FTSE-led row: UK names with a FTSE 100/250 change at the same review moved with demand on T, "
                "adds +128 bp (t 1.6, n 6) / deletes +111 bp (t 0.3, n 5) (Q, close_dislocation RESULTS §7). Tiny "
                "sample, note only. UK names without one: adds -140 / deletes -126 bp")   # no '; ' (notes separator)


# ------------------------------------------------------------------------------------- hedge gap, venue, depth
def _month_end(d: pd.Timestamp) -> bool:
    """Last weekday of the month (the day-type rule of expiry_basis)."""
    return d == d + pd.offsets.BMonthEnd(0)


def _msci_month_unknown(d: pd.Timestamp) -> bool:
    """A Feb/May/Aug/Nov close whose month has no MSCI review close in common.MSCI_REVIEWS."""
    return d.month in (2, 5, 8, 11) and f"{d:%Y-%m}" not in MSCI_REVIEWS


def day_type(close_date) -> str:
    """expiry_basis day type of a close (s02_auction_move.py order: Q-expiry > MSCI review > quarter-end > ...).

    MSCI review closes come from common.MSCI_REVIEWS (12_studies/msci_pi REVIEWS + 2026-11-30). The last business day
    of a Feb/May/Aug/Nov month that has no entry there is ASSUMED to be the review close; day_type_flag says so."""
    d = pd.Timestamp(close_date).normalize()
    mid = pd.Timestamp(d.year, d.month, 15)
    third_fri = mid + pd.Timedelta(days=(4 - mid.weekday()) % 7)
    month_end = _month_end(d)
    if d == third_fri and d.month in (3, 6, 9, 12):
        return "1 Q-expiry/rebal"
    if str(d.date()) in MSCI_REVIEW_CLOSES or (month_end and _msci_month_unknown(d)):
        return "2 MSCI review"
    if month_end and d.month in (3, 6, 9, 12):
        return "3 quarter-end"
    if month_end:
        return "4 other month-end"
    if d == third_fri:
        return "5 monthly expiry"
    return "6 other"


def day_type_flag(close_date) -> str:
    """'' or a FLAG when the close falls in a Feb/May/Aug/Nov month whose MSCI review close is not on the list."""
    d = pd.Timestamp(close_date).normalize()
    if not _msci_month_unknown(d):
        return ""
    head = f"FLAG: no MSCI review close for {d:%Y-%m} in {MSCI_REVIEWS_SOURCE}"
    if _month_end(d):
        return (f"{head}; this last business day is ASSUMED to be the review close (day type 2 MSCI review); "
                "check MSCI's published date")
    return f"{head}; day type {day_type(d)} assumes this close is not the review close; check MSCI's published date"


def _last_session_of_month(d: pd.Timestamp, exch: str = "XTSE") -> bool:
    """True if d is the last session of its month on `exch` (exchange_calendars, common.sessions); the last weekday
    of the month if the calendar is unavailable."""
    try:
        cal = sessions(exch)
        m = cal[(cal.year == d.year) & (cal.month == d.month)]
        if len(m):
            return bool(d == m.max())
    except Exception:  # noqa: BLE001
        pass
    return _month_end(d)


def _gap_stats(x: pd.DataFrame) -> dict:
    a = x.a_bp
    return dict(n=int(len(a)), first=str(x.date.min().date()) if len(a) else "", last=str(x.date.max().date())
                if len(a) else "", sd=float(a.std()), mean=float(a.mean()) if len(a) else np.nan,
                rms=float(np.sqrt((a ** 2).mean())) if len(a) else np.nan,
                n_not_month_end=int((~x.date.map(_month_end)).sum()) if len(a) else 0)


def _gap_sample(st: dict) -> str:
    if not st["n"]:
        return "no dates"
    return (f"{st['first']} to {st['last']}, n {st['n']}; mean {_fmt(round(st['mean'], 1) or 0.0, '+.1f')}, "
            f"RMS {_fmt(st['rms'], '.1f')} bp")


def hedge_table(dtype: str, close_date=None) -> dict:
    """future -> settles / gap sd (bp) / basis, for one day type and close, from 12_studies/expiry_basis.

    settles: table_settle_timing.csv, column 'futures settle (2024-26)'. sd: table_scale_by_type.csv for the seven
    futures that settle before the close (s03 sample: from 2010, |a| < 150 bp); for the on/near-close futures (SXF, FCE,
    FTI, FIB) the sd of the same measure a from data/auction_move_daily.csv since the settlement regime started
    (HEDGE_REGIME_START). SXF on a month-end close (the last TSX session of the month): same-day-type dates from
    SXF_MONTH_END_START only, because the month-end settlement has been built off the close since 21 Jun 2021 (MX
    circular 101-21; expiry_basis RESULTS §3). Every basis gives the sample's first and last date, n, and the mean and
    RMS of a over it (recomputed from the daily file with the same filter)."""
    st = pd.read_csv(EXPIRY_BASIS / "table_settle_timing.csv").set_index("idx")
    raw = pd.read_csv(EXPIRY_BASIS / "table_scale_by_type.csv", header=None, dtype=str)
    stat, typ = raw.iloc[0, 1:].tolist(), raw.iloc[1, 1:].tolist()
    sd = {}
    for _, r in raw.iloc[3:].iterrows():
        for k, (a, b) in enumerate(zip(stat, typ)):
            if a == "sd":
                sd[(r.iloc[0], b)] = float(r.iloc[k + 1])
    daily = pd.read_csv(EXPIRY_BASIS / "data" / "auction_move_daily.csv", parse_dates=["date"])
    daily = daily[(daily.type == dtype) & (daily.a_bp.abs() < STALE_BP)]
    month_end = close_date is not None and _last_session_of_month(pd.Timestamp(close_date).normalize())
    out = {}
    for fut, idx in FUT_INDEX.items():
        lab = SETTLE_LABEL.get(st.at[idx, "futures settle (2024-26)"], "unknown") if idx in st.index else "unknown"
        if (idx, dtype) in sd:
            g = _gap_stats(daily[(daily.idx == idx) & (daily.date >= SCALE_TABLE_START)])
            chk = ("" if abs(round(g["sd"], 1) - sd[(idx, dtype)]) < 0.051
                   else f" (recomputed sd {_fmt(g['sd'], '.2f')})")
            out[fut] = dict(settles=lab, sd=sd[(idx, dtype)], stats=g,
                            basis=f"expiry_basis table_scale_by_type.csv, sd, {dtype}: {_gap_sample(g)}{chk}")
        elif fut in HEDGE_REGIME_START:
            start, why = HEDGE_REGIME_START[fut], ""
            if fut == "SXF" and month_end:
                start, why = SXF_MONTH_END_START, (" (month-end close: SXF month-end settlement off the close since "
                                                   "then, MX circular 101-21)")
            x = daily[(daily.fut == fut) & (daily.date >= start)]
            g = _gap_stats(x)
            out[fut] = dict(settles=lab, sd=g["sd"] if g["n"] >= 5 else np.nan, stats=g,
                            basis=f"expiry_basis auction_move_daily.csv, sd, {dtype}, since {start}{why}: "
                                  f"{_gap_sample(g)}")
            if why:
                out[fut]["regime_stats"] = _gap_stats(daily[(daily.fut == fut)
                                                            & (daily.date >= HEDGE_REGIME_START[fut])])
        else:
            out[fut] = dict(settles=lab, sd=np.nan, stats=None, basis="no measure")
    return out


def _holdings_isins(etf: str, close_date) -> tuple[set, str]:
    """ISINs held by a cached iShares fund file (no web request): the latest file dated on or before the close,
    else the latest file. Used only to tell FTSE 100 from FTSE 250 names for the hedge future."""
    dated = []
    for p in sorted((HOLD / etf).glob(f"{etf}_*.json")):
        m = re.search(r"_(asof|live)(\d{8})\.json$", p.name)
        if m:
            dated.append((pd.Timestamp(m.group(2)), p))
    if not dated:
        return set(), ""
    before = [x for x in dated if x[0] <= pd.Timestamp(close_date)]
    _, p = max(before) if before else max(dated)
    rows = json.loads(p.read_text(encoding="utf-8-sig")).get("aaData", [])
    out = {str(r[8 + len(r) - 13]) for r in rows if len(r) >= 13 and r[3 + len(r) - 13] == "Equity"}
    return out, p.name


def uk_index_after(g: pd.DataFrame | None, key: str, isf: set, midd: set) -> str:
    """FTSE 100 / FTSE 250 membership after the close, for the hedge future: this close's FTSE rows first, then the
    cached ISF / MIDD holdings; 'unknown' if neither says."""
    types = set() if g is None else set(zip(g["index"], g.change_type))
    if any(j == "FTSE100" and t in ("add", "migration_in") for j, t in types):
        return "FTSE 100"
    if any(j == "FTSE100" and t in ("delete", "migration_out") for j, t in types) or any(j == "FTSE250" for j, _ in types):
        return "FTSE 250"
    if any(j == "FTSE100" for j, _ in types) or key in isf:
        return "FTSE 100"
    return "FTSE 250" if key in midd else "unknown"


def hedge_future_for(country: str, region: str, uk_index: str) -> str:
    if country == "GB":
        return HEDGE_FUTURE_UK_MID if uk_index == "FTSE 250" else HEDGE_FUTURE["GB"]
    if country in HEDGE_FUTURE:
        return HEDGE_FUTURE[country]
    return HEDGE_FUTURE_EU_DEFAULT if region == "EU" else ""


def pricing_meta(close_date) -> dict:
    dtype = day_type(close_date)
    isf, f100 = _holdings_isins("ISF", close_date)
    midd, f250 = _holdings_isins("MIDD", close_date)
    return dict(day_type=dtype, day_flag=day_type_flag(close_date), hedge=hedge_table(dtype, close_date), isf=isf,
                midd=midd, uk_files=f"{f100}, {f250}")


# venue_rules.csv: 'HH:MM[:SS] ET' times (single or 'a-b' ranges) are written for VENUE_ET_REF_DATE; venue_cells()
# shifts them for the close date by the change in (venue time zone - New York) offset (zoneinfo).
_ET_TIME = re.compile(r"(?<![\d:])(\d{1,2}:\d{2}(?::\d{2})?)(?:-(\d{1,2}:\d{2}(?::\d{2})?))?(?= ET\b)")


def tz_gap_h(tz: str, close_date) -> float:
    """Hours by which local time in `tz` is ahead of New York on the close date (at 12:00 local)."""
    d = pd.Timestamp(close_date)
    t = datetime(d.year, d.month, d.day, 12, tzinfo=ZoneInfo(tz))
    return (t.utcoffset() - t.astimezone(ZoneInfo(ET_TZ)).utcoffset()).total_seconds() / 3600


def et_shift_h(tz: str, close_date) -> float:
    """Hours to add to venue_rules.csv's ET times for this close (0 on 30 Nov, 18 Dec and 19 Jun 2026)."""
    return tz_gap_h(tz, VENUE_ET_REF_DATE) - tz_gap_h(tz, close_date) if tz else 0.0


def _shift_et(text: str, h: float) -> str:
    if not h or not text:
        return text

    def one(t: str) -> str:
        p = t.split(":")
        hh, mm = divmod((int(p[0]) * 60 + int(p[1]) + round(h * 60)) % 1440, 60)
        return ":".join([f"{hh:0{len(p[0])}d}", f"{mm:02d}"] + p[2:])
    return _ET_TIME.sub(lambda m: one(m.group(1)) + (f"-{one(m.group(2))}" if m.group(2) else ""), text)


def venue_rules() -> pd.DataFrame:
    if not VENUE_RULES.exists():
        return pd.DataFrame(columns=["country"] + VENUE_COLS + ["tag", "source", "notes", "tz"]).set_index("country")
    return (pd.read_csv(VENUE_RULES, dtype=str, keep_default_na=False).drop_duplicates("country")
            .set_index("country"))


def venue_cells(close_date, countries) -> dict:
    """country -> {venue column: text} for this close: venue_rules.csv with ET times shifted to the close date, and
    on a 'gap' row every blank cell set to 'gap: <notes>'."""
    vr, out = venue_rules(), {}
    for k in set(countries):
        if k not in vr.index:
            out[k] = {col: "not in venue_rules.csv" for col in VENUE_COLS}
            continue
        r = vr.loc[k]
        h = et_shift_h(r.get("tz", ""), close_date)
        gap = r.get("tag", "") == "gap"
        out[k] = {col: (f"gap: {r.get('notes', '')}" if gap and not r[col] else
                        r[col] if col == "venue" else _shift_et(r[col], h)) for col in VENUE_COLS}
    return out


def attach_pricing(s: pd.DataFrame, c: pd.DataFrame, close_date) -> pd.DataFrame:
    """Pricing layer v2: close-pricing row, hedge gap, venue close rules and depth stress for every stock."""
    tab = pricing_table()
    pr = []
    for r in s.itertuples():
        side = 1 if r.net_sh > 0 else -1
        pr.append(pricing_row(tab, r.region, r.country, side, r.dominant_index, r.dominant_type,
                              r.price if r.region == "US" else np.nan))
    s = pd.concat([s, pd.DataFrame(pr, index=s.index)], axis=1)
    meta = pricing_meta(close_date)
    by_key = {k: g for k, g in c.groupby("key")}
    uk = [uk_index_after(by_key.get(r.key), r.key, meta["isf"], meta["midd"]) if r.country == "GB" else ""
          for r in s.itertuples()]
    fut = [hedge_future_for(r.country, r.region, u) for r, u in zip(s.itertuples(), uk)]
    ht = meta["hedge"]
    s["uk_index"] = uk
    s["hedge_future"] = fut
    s["hedge_index"] = [FUT_INDEX.get(f, "") for f in fut]
    s["hedge_settles"] = [ht.get(f, {}).get("settles", "") for f in fut]
    s["hedge_day_type"] = meta["day_type"]
    s["hedge_gap_sd_bp"] = [ht.get(f, {}).get("sd", np.nan) for f in fut]
    flag = f"{meta['day_flag']}; " if meta["day_flag"] else ""
    s["hedge_gap_basis"] = [flag + ht.get(f, {}).get("basis", "no index future mapped")
                            + (f" (uk_index unknown, {HEDGE_FUTURE['GB']} assumed)" if u == "unknown" else "")
                            for f, u in zip(fut, uk)]
    vc = venue_cells(close_date, s.country)
    for col in VENUE_COLS:
        s[col] = [vc[k][col] for k in s.country]
    one = s.phi * s.net_sh.abs()
    s["close_one_sided_share"] = one / (s.close_base_sh + one)
    s["depth_stress_bp"] = DEPTH_BP_PER_PCT * 100.0 * s.close_one_sided_share
    return s


# ------------------------------------------------------------------------------------------------ core
def contributions(rows: pd.DataFrame, close_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    w_form = rows.old_index_shares.isna() | rows.new_index_shares.isna()
    dF = rows.new_index_shares - rows.old_index_shares
    dF_w = (rows.new_weight - rows.old_weight) * rows.index_mcap / rows.price
    rows["dF"] = np.where(w_form, dF_w, dF)
    rows["form"] = np.where(w_form, "weights", "shares")
    pis = pd.DataFrame([pi_for(j, close_date) for j in sorted(rows["index"].unique())])
    rows = rows.merge(pis, on="index", how="left")
    rows["D_vis"] = rows.pi_visible * rows.dF
    rows["D"] = rows.pi_true * rows.dF
    rows["E"] = rows.D * rows.prob
    rows["key"] = [key_of(r) for r in rows.itertuples()]
    return rows, pis


def aggregate(c: pd.DataFrame, md: pd.DataFrame | None, close_date, md_source: str = "supplied") -> pd.DataFrame:
    """One row per stock. With market data `md`, a stock whose key has no md row keeps blank price/FX/ADV/close
    base (so it drops out of every EUR total) and gets adv_basis = close_basis = 'not in <md_source> md'; run() lists
    those names in a bold note."""
    c = c[np.isfinite(c.D)].copy()
    c["region"] = c.country.map(region_of_country)
    out = []
    for key, g in c.groupby("key"):
        non_us = g[g.region != "US"]
        lead = (non_us if len(non_us) else g).iloc[0]
        net, gross = g.D.sum(), g.D.abs().sum()
        by_idx = g.groupby("index").D.sum()
        dom = by_idx.abs().idxmax()
        dom_type = g.loc[g["index"] == dom].sort_values("D", key=abs).change_type.iloc[-1]
        contrib = "; ".join(f"{j} {v:+,.0f}" for j, v in by_idx.sort_values(key=abs, ascending=False).items())
        types = "; ".join(f"{j}:{t}" for j, t in g.groupby("index").change_type.agg(
            lambda s: "/".join(sorted(set(s)))).items())
        rec = dict(key=key, isin=lead.isin if isinstance(lead.isin, str) else "", ticker=lead.ticker,
                   name=lead["name"], country=lead.country, region=lead.region, ccy=lead.currency,
                   price=g.price.dropna().iloc[0] if g.price.notna().any() else np.nan,
                   net_sh=net, gross_sh=gross, net_sh_visible=g.D_vis.sum(), exp_net_sh=g.E.sum(),
                   min_prob=g.prob.min(), n_indices=g["index"].nunique(),
                   offset=bool((g.D > 0).any() and (g.D < 0).any()),
                   offset_ratio=abs(net) / gross if gross > 0 else np.nan,
                   ca_sh=g.loc[g.region == "CA", "D"].sum(), us_sh=g.loc[g.region == "US", "D"].sum(),
                   eu_sh=g.loc[g.region == "EU", "D"].sum(),
                   contributions=contrib, change_types=types, dominant_index=dom, dominant_type=dom_type,
                   statuses="/".join(sorted(set(g.status))),
                   pi_fallback=bool(g.fallback.any()), R_assumed=bool(g.R_assumed.any()))
        out.append(rec)
    s = pd.DataFrame(out)
    if s.empty:
        return s
    if md is not None and len(md):
        m = md.copy()
        agg = m.groupby("key").agg(adv_sh=("adv_sh", lambda v: v.sum(min_count=1)),
                                   close_base_sh=("close_base_sh", lambda v: v.sum(min_count=1)),
                                   adv_basis=("adv_basis", "first"), close_basis=("close_basis", "first"),
                                   close_n_days=("close_n_days", "max"),
                                   eur_per_ccy=("eur_per_ccy", "first"), cad_per_ccy=("cad_per_ccy", "first"),
                                   md_price=("price", "first"))
        if "v_T" in m:
            agg["v_T"] = m.groupby("key").v_T.sum(min_count=1)
        s = s.merge(agg.reset_index(), on="key", how="left")
        s["price"] = s.price.fillna(s.md_price)
        s = s.drop(columns=["md_price"])
        miss = ~s.key.isin(set(m.key))
        s.loc[miss, ["adv_basis", "close_basis"]] = f"not in {md_source} md"
    else:
        for col in ["adv_sh", "close_base_sh", "eur_per_ccy", "cad_per_ccy", "close_n_days"]:
            s[col] = np.nan
        s["adv_basis"] = s["close_basis"] = "no market data"
    s["notional_local"] = s.net_sh * s.price
    s["gross_notional_local"] = s.gross_sh * s.price
    s["notional_eur"] = s.notional_local * s.eur_per_ccy
    s["notional_cad"] = s.notional_local * s.cad_per_ccy
    s["gross_notional_eur"] = s.gross_notional_local * s.eur_per_ccy
    s["exp_notional_eur"] = s.exp_net_sh * s.price * s.eur_per_ccy
    s["pct_adv"] = s.net_sh.abs() / s.adv_sh
    s["pct_adv_visible"] = s.net_sh_visible.abs() / s.adv_sh
    s["close_mult"] = s.net_sh.abs() / s.close_base_sh
    s["phi"] = s.region.map(PHI)
    s["exp_close_print_mult"] = (s.close_base_sh + s.phi * s.net_sh.abs()) / s.close_base_sh
    s = attach_pricing(s, c, close_date)
    s["side"] = np.where(s.net_sh > 0, "BUY", np.where(s.net_sh < 0, "SELL", "FLAT"))
    s = s.sort_values("notional_eur", key=lambda v: v.abs(), ascending=False).reset_index(drop=True)
    return s


SHEET_COLS = ["key", "isin", "ticker", "name", "country", "ccy", "side", "price", "net_sh", "gross_sh",
              "net_sh_visible", "exp_net_sh", "min_prob", "notional_local", "notional_eur", "notional_cad",
              "gross_notional_eur", "exp_notional_eur", "adv_sh", "pct_adv", "pct_adv_visible", "close_base_sh",
              "close_mult", "exp_close_print_mult", "n_indices", "offset", "offset_ratio", "contributions",
              "change_types", "ca_sh", "us_sh", "eu_sh", "dominant_index", "statuses", "pi_fallback", "R_assumed"
              ] + PRICING_COLS_V2 + ["adv_basis", "close_basis", "close_n_days", "eur_per_ccy", "cad_per_ccy"]


MD_COLS = ["key", "country", "ccy", "price", "adv_sh", "adv_basis", "close_base_sh", "close_basis", "close_n_days",
           "eur_per_ccy", "cad_per_ccy", "v_T"]


def load_md(path) -> pd.DataFrame:
    """Frozen market data for a re-run (no web request): an md file (sheets/<close>_<label>_md.csv,
    validation/<close>_marketdata.csv) or a sheet CSV, whose market-data columns are then used as stored."""
    m = pd.read_csv(path)
    return m[[c for c in MD_COLS if c in m.columns]].copy()


def run(close_date: str, status: str = "best", include_us: bool = False, md: pd.DataFrame | None = None,
        md_source: str = "none", label: str | None = None, extra_notes: list[str] | None = None,
        write: bool = True, files: list[str] | None = None) -> dict:
    close_date = str(pd.Timestamp(close_date).date())
    rows = I.load(close_date)
    if files:
        rows = rows[rows["_file"].isin(files)].reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"no inputs in inputs/{close_date}/" + (f" among {files}" if files else ""))
    if not include_us:
        rows = rows[~rows["index"].isin(US_INDICES)]
    rows = select_status(rows, status)
    c, pis = contributions(rows, close_date)
    dropped = c[~np.isfinite(c.D)].groupby("index").size()
    extra_notes = list(extra_notes or [])
    for j, n in dropped.items():
        extra_notes.append(f"**{j}: {n} rows dropped - no pi for this index on or before {close_date}** "
                           "(add a row to pi_overrides.csv)")
    if md is None and md_source == "live":
        import marketdata as M
        lines = c.drop_duplicates(["key", "country"])[["key", "isin", "ticker", "country", "currency", "price"]]
        md = M.md_live(lines.rename(columns={"currency": "ccy"}))
    md_label = md_source if md_source not in ("", "none") else "supplied"   # e.g. validate_*.py pass md only
    s = aggregate(c, md, close_date, md_source=md_label)
    if md is not None and len(md) and len(s):
        miss = s[s.adv_basis.astype(str).str.startswith("not in ")]
        if len(miss):
            src = "the frozen market data (`--md-from`)" if md_source == "frozen" else f"the {md_label} market data"
            msg = (f"**{len(miss)} name(s) have no row in {src}: {', '.join(miss.ticker.astype(str))}. Their FX, ADV "
                   "and close base (and price, unless the input gives one) are blank (`adv_basis` = 'not in "
                   f"{md_label} md'), so they are left out of every € total, the hedge gap and the depth stress.**")
            extra_notes.append(msg)
            print("WARNING: " + msg.strip("*"), file=sys.stderr)
    label = label or status
    res = dict(sheet=s, contrib=c, pis=pis, close_date=close_date, status=label)
    if write:
        base = SHEETS / f"{close_date}_{label}"
        cols = [x for x in SHEET_COLS if x in s] + [x for x in s.columns if x not in SHEET_COLS]
        s[cols].to_csv(f"{base}.csv", index=False, float_format="%.6g")
        cc = c[["key", "isin", "ticker", "name", "country", "index", "change_type", "status", "prob", "dF", "form",
                "pi_visible", "R", "pi_true", "D_vis", "D", "E", "price", "currency", "fallback", "R_assumed",
                "pi_source", "source", "source_ts", "notes"]]
        cc.to_csv(f"{base}_contrib.csv", index=False, float_format="%.6g")
        res["paths"] = [f"{base}.csv", f"{base}_contrib.csv"]
        if md is not None and len(md):
            md.to_csv(f"{base}_md.csv", index=False)
            res["paths"].append(f"{base}_md.csv")
        (SHEETS / f"{close_date}_{label}.md").write_text(summary_md(res, extra_notes))
        res["paths"].append(f"{base}.md")
    return res


# --------------------------------------------------------------------------------------------- summary
def _tbl(df: pd.DataFrame, cols: list[str], fmt: dict) -> str:
    if df.empty:
        return "_none_\n"
    h = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        d = r._asdict()
        h += "| " + " | ".join(fmt.get(c, lambda v: str(v))(d[c]) for c in cols) + " |\n"
    return h


def _q(a, b) -> str:
    """lambda-only quote -> quote with the expected-loss add-back; one number when the add-back is zero."""
    if not np.isfinite(a):
        return "n/a"
    a, b = (round(a) or 0), (round(b) or 0) if np.isfinite(b) else b
    return f"{a:+.0f}" if not np.isfinite(b) or b == a else f"{a:+.0f} → {b:+.0f}"


def _short_print(v: str) -> str:
    v = str(v)
    if v.startswith("gap"):
        return "gap"
    return v.split(" (")[0] if v and v != "nan" else "n/a"


NEXT_Q_MONTH = {3: "June", 6: "September", 9: "December", 12: "March"}
VENUE_SOURCES_DATES = ("2026-11-30", "2026-12-18")   # dates european_closing_rules.md §0 checks as full trading days


def _names(t, k: int = 12) -> str:
    t = [str(x) for x in t]
    return ", ".join(t[:k]) + (f", +{len(t) - k} more" if len(t) > k else "")


def pricing_sections_md(s: pd.DataFrame, close_date: str) -> list[str]:
    """Pricing summary, hedge gap, close timing and depth stress sections of the sheet .md (pricing layer v2)."""
    meta = pricing_meta(close_date)
    dtype = meta["day_type"]
    f0 = lambda v: f"{(round(v) or 0):+.0f}" if np.isfinite(v) else "n/a"     # noqa: E731
    L = ["## Pricing summary (pricing layer v2, 2026-10-15)", "",
         "Per pricing row and side, bp against the pre-close reference (T−1 close), + = the price moves with net "
         "demand. `E[T−1→T]` is the row's mean trade-day dislocation (t). Quotes are per name: λ-only "
         f"(E[T−1→T] + {PRICING_LAMBDA} × per-name SD of the hold, close_dislocation §9) → with the expected-loss "
         "add-back, + max(0, −E[reversal to the exit]) (msci_close RESULTS §6). One number = no add-back. "
         "Gross of commissions, spread, impact, borrow and taxes. Rows: MSCI families from "
         "`12_studies/msci_close/table_pricing_msci.csv` (Oct2021+; event-type sub-row where n ≥ "
         f"{MSCI_SUBROW_MIN_N}), S&P/TSX from `close_dislocation/table_pricing_post2021.csv` (Oct2021+ where n ≥ "
         f"{TSX_MIN_N}), other families from `close_dislocation/table_pricing.csv` (2016+). Hedge and timing columns "
         "are detailed in the next two sections. A proxy row borrows the nearest family (e.g. FTSE 100/250 changes "
         "priced off STOXX 600 add/delete rows, capping or share updates off STOXX 600 add/delete or DAX-family net "
         "buy/sell rows); see `pricing_note`.", ""]
    if "pricing_row" not in s or s.pricing_row.isna().all():
        return L + ["_no pricing rows attached_", ""]
    rows = []
    for (pr, side), g in s.groupby(["pricing_row", "side"], sort=False):
        x = g.iloc[0]
        fut = g.groupby("hedge_future").size()
        hedge = ", ".join(f"{f or 'none'} {g.loc[g.hedge_future == f, 'hedge_settles'].iloc[0]} "
                          f"{_fmt(g.loc[g.hedge_future == f, 'hedge_gap_sd_bp'].iloc[0], '.1f')} bp ({n})"
                          for f, n in fut.sort_values(ascending=False).items())
        tim = ", ".join(sorted({f"{c} {_short_print(v)}" for c, v in zip(g.country, g.normal_print_et)}))
        nt = g.pricing_note.fillna("")
        flags = f"{int(nt.str.contains('FLAG|FALLBACK').sum())} / {int(nt.str.contains('proxy row').sum())}"
        rows.append(dict(row=pr, side=side, names=len(g), net=g.notional_eur.sum(), n=x.pricing_n,
                         ET=(f"{x.E_T_dislocation_bp:+.0f} ({_fmt(x.E_T_t, '+.1f')})"
                             if np.isfinite(x.E_T_dislocation_bp) else "n/a"),
                         open=_q(x.quote_hold_T1open_bp, x.quote_adj_hold_T1open_bp),
                         close1=_q(x.quote_hold_T1close_bp, x.quote_adj_hold_T1close_bp),
                         t5=_q(x.quote_hold_T5_bp, x.quote_adj_hold_T5_bp),
                         hedge=hedge, timing=tim, flags=flags))
    t = pd.DataFrame(rows).sort_values("net", key=lambda v: v.abs(), ascending=False)
    L.append(_tbl(t, ["row", "side", "names", "net", "n", "ET", "open", "close1", "t5", "hedge", "timing", "flags"],
                  {"net": fm_eur, "n": lambda v: f"{v:.0f}" if v == v else "n/a"})
             .replace("| row | side | names | net | n | ET | open | close1 | t5 | hedge | timing | flags |",
                      "| pricing row | side | names | net € | n | E[T−1→T] (t) | quote, hold to T+1 open | "
                      "quote, hold to T+1 close | quote, hold to T+5 | hedge future: settles, 1-sd gap (names) | "
                      "normal print, ET | names flagged / on a proxy row |"))
    notes = s.pricing_note.fillna("")
    fl = s[notes.str.contains("FLAG|FALLBACK")]
    if len(fl):
        L.append("**Flagged rows** (pooled row used because the event-type sub-row is too thin, or a fallback):\n")
        for msg, g in fl.groupby(fl.pricing_note.str.split("; ").str[0]):
            L.append(f"- {', '.join(g.ticker.astype(str))}: {msg}" + (
                f" ({g.pricing_note.iloc[0].split('; ')[1]})" if "all-period sub-row" in g.pricing_note.iloc[0] else ""))
        L.append("")
    uk = sorted({n for v in notes for n in v.split("; ") if n.startswith("UK")})
    if uk:
        gb = s.loc[s.country == "GB", "ticker"].astype(str).tolist()
        per = {n: s.loc[notes.map(lambda v, n=n: n in v.split("; ")), "ticker"].tolist() for n in uk}
        L += [f"**UK names** ({len(gb)}: {_names(gb)}):", ""] + [
            f"- {n}" + (f" — {len(per[n])} names: {_names(per[n])}" if len(uk) > 1 else "") for n in uk] + [""]
    # hedge gap
    L += ["## Hedge gap (index futures vs the cash close)", "",
          f"Day type of this close: **{dtype}** (12_studies/expiry_basis). A risk position hedged with the listing "
          "market's index future at that future's daily settlement carries the index-level move in the closing "
          "auction after the settlement. `1-sd €` = |net notional| × gap sd, **assuming beta = 1** to the future's "
          "index for every name (not estimated). The sum assumes the gaps are perfectly correlated (the European "
          "ones are highly correlated, expiry_basis §3). Single-stock risk is in the quotes above, not here.", ""]
    if meta["day_flag"]:
        L += [f"**{meta['day_flag']}.**", ""]
    ht = meta["hedge"]
    hg = s.assign(hf=s.hedge_future.replace("", "none")).groupby("hf").agg(
        names=("key", "size"), net=("notional_eur", "sum"), exp=("exp_notional_eur", "sum")).reset_index()
    stat = lambda f, k: (ht.get(f, {}).get("stats") or {}).get(k, np.nan)    # noqa: E731
    hg["index"] = hg.hf.map(FUT_INDEX).fillna("")
    hg["settles"] = hg.hf.map(lambda f: ht.get(f, {}).get("settles", ""))
    hg["sd"] = hg.hf.map(lambda f: ht.get(f, {}).get("sd", np.nan))
    hg["mean"] = hg.hf.map(lambda f: stat(f, "mean"))
    hg["rms"] = hg.hf.map(lambda f: stat(f, "rms"))
    hg["mr"] = [f"{_fmt(m, '+.1f')} / {_fmt(r, '.1f')}" for m, r in zip(hg["mean"], hg.rms)]
    hg["risk"] = hg.net.abs() * hg.sd / 1e4
    hg["risk_exp"] = hg.exp.abs() * hg.sd / 1e4
    hg["basis"] = hg.hf.map(lambda f: ht.get(f, {}).get("basis", "no index future mapped"))
    hg = hg.sort_values("risk", ascending=False)
    L.append(_tbl(hg, ["hf", "index", "settles", "names", "net", "sd", "mr", "risk", "exp", "risk_exp", "basis"],
                  {"net": fm_eur, "exp": fm_eur, "risk": fm_eur, "risk_exp": fm_eur,
                   "sd": lambda v: f"{v:.1f}" if np.isfinite(v) else "n/a"})
             .replace("| hf | index | settles | names | net | sd | mr | risk | exp | risk_exp | basis |",
                      "| future | index | settles | names | net notional € | gap sd, bp | gap mean / RMS, bp | "
                      "1-sd € | "
                      "prob-weighted net € | 1-sd € (prob-weighted) | sd source and sample |"))
    L.append(f"Sum of 1-sd risk: {fm_eur(hg.risk.sum())} (all changes happen), {fm_eur(hg.risk_exp.sum())} "
             "(probability-weighted).\n")
    for r in hg[np.isfinite(hg.rms) & np.isfinite(hg.sd) & (hg.rms > 1.1 * hg.sd)].itertuples():
        L.append(f"- **{r.hf}: the gap has a non-zero mean on this day type** ({r.mean:+.1f} bp), so its RMS "
                 f"({r.rms:.1f} bp) is {r.rms / r.sd - 1:.0%} above the sd. On the RMS the 1-sd € would be "
                 f"{fm_eur(abs(r.net) * r.rms / 1e4)} ({fm_eur(abs(r.exp) * r.rms / 1e4)} probability-weighted).")
    L.append("- Futures that settle **before** the close (Eurex FDAX, FXXP, FSMI; ICE Z, Y2; MEFF FIE): the gap "
             "is the auction move after the daily settlement (Eurex: VWAP of the minute before 17:30 CET, FSMI 17:20; "
             "ICE 16:28–16:30 UK; MEFF 17:29–17:30 CET). On-close tools remove it: Eurex Trade at Index Close (TAIC), "
             "ICE Trade at Index Close (TIC), Montréal Basis Trade on Close (BTC); none on MEFF (expiry_basis §3, `X`).")
    L.append("- **On / near** the close (FCE, FTI, FIB on the close; SXF a 15:59–16:00 ET VWAP): the sd shown is the "
             "measured residual since that settlement regime began, not zero.")
    sx = ht.get("SXF", {})
    if "regime_stats" in sx and (s.hedge_future == "SXF").any():
        g, g0 = sx["stats"], sx["regime_stats"]
        L.append(f"- **SXF on a month-end close:** since {SXF_MONTH_END_START} the SXF settlement on the last "
                 "business day of the month is a TWAP of the implied basis blended with BTC quotes, built off the "
                 "close (MX circular 101-21, `X`, 03_microstructure/index_futures_at_close.md). The sd therefore uses "
                 f"same-day-type dates ({dtype}) from {SXF_MONTH_END_START} only: {_fmt(g['sd'], '.1f')} bp, "
                 f"n {g['n']} "
                 f"({g['first']} to {g['last']}"
                 + (f"; {g['n_not_month_end']} of them were review dates that were not month-ends" if
                    g["n_not_month_end"] else "")
                 + f"), against {_fmt(g0['sd'], '.1f')} bp, n {g0['n']} since {HEDGE_REGIME_START['SXF']} `Q`.")
    if dtype == "1 Q-expiry/rebal":
        nxt = NEXT_Q_MONTH.get(pd.Timestamp(close_date).month, "next quarterly")
        L.append("- **Quarterly expiry close:** the front contracts expire before the close (FESX/FXXP index average "
                 "11:50–12:00 CET, FDAX Xetra intraday auction 13:00, FSMI SIX opening, Z/Y2 EDSP from the 10:10 LSE "
                 "intraday auction, SXF official opening level with trading stopped the day before). The close hedge "
                 f"goes in the next ({nxt}) contract; ICE TIC and Eurex FES1 do not trade the expiring contract that "
                 "day (expiry_basis §3, T5).")
    if s.country.eq("GB").any():
        unk = s.loc[s.uk_index.eq("unknown"), "ticker"].astype(str).tolist()
        L.append(f"- UK names use Z if they are FTSE 100 members after the close, else Y2 (this close's FTSE rows, "
                 f"then the cached iShares files {meta['uk_files']})."
                 + (f" uk_index unknown, Z assumed: {_names(unk)}." if unk else ""))
    L.append("")
    # timing
    d = pd.Timestamp(close_date)
    cet, ukh = tz_gap_h("Europe/Paris", d), tz_gap_h("Europe/London", d)
    shift = 6 - cet
    vr = venue_rules()
    ctry = sorted(set(s.country))
    gaps = [c for c in ctry if c not in vr.index or vr.at[c, "tag"] == "gap"]
    eu_cov = [c for c in ctry if c not in gaps and "european_closing_rules" in vr.at[c, "source"]]
    na = [c for c in ctry if c not in gaps and c not in eu_cov]
    txt = ("From `03_microstructure/european_closing_rules.md` §0 and `closing_auctions.md` §2, §6. ET times are for "
           f"this close ({d.day} {d:%b %Y}), converted with each venue's time zone (`tz` in venue_rules.csv, "
           "zoneinfo): "
           f"CET = ET + {cet:g} h, UK = ET + {ukh:g} h, Toronto = ET.")
    if shift:
        txt += (f" The US and Europe change clocks on different dates, so every European time here is {abs(shift):g} "
                f"h {'later' if shift > 0 else 'earlier'} in ET than on 30 Nov or 18 Dec 2026.")
    if str(d.date()) in VENUE_SOURCES_DATES and eu_cov:
        txt += (" `european_closing_rules.md` §0 lists 30 Nov and 18 Dec 2026 as normal full trading days at every "
                f"European venue it covers (here: {', '.join(eu_cov)}); the 24 and 31 Dec half days and closures do "
                "not touch either date.")
        if na:
            txt += (f" {', '.join(na)}: `02_markets/event_calendar.md` §2.4 lists no holiday or early close on "
                    "either date (`D`).")
    else:
        txt += (" Half days and closures on this date are not checked here: the sources check 30 Nov and 18 Dec "
                "2026 only.")
    if gaps:
        txt += f" Not covered by these sources: {', '.join(gaps)} (gap rows, below)."
    L += ["## Close timing by listing venue (`venue_rules.csv`)", "", txt, ""]
    vt = s.groupby("country").agg(names=("key", "size"), net=("notional_eur", "sum"), venue=("venue", "first"),
                                  normal=("normal_print_et", "first"), latest=("latest_print_et", "first"),
                                  post=("post_close_session", "first"), hidden=("hidden_imbalance_order", "first")
                                  ).reset_index().sort_values("net", key=lambda v: v.abs(), ascending=False)
    L.append(_tbl(vt, ["country", "venue", "names", "net", "normal", "latest", "post", "hidden"],
                  {"net": fm_eur}).replace("| country | venue | names | net | normal | latest | post | hidden |",
                                           "| country | venue | names | net € | normal print (ET) | latest print "
                                           "(ET) | post-close session | hidden imbalance order |"))
    # depth stress
    lo, hi = DEPTH_BP_PER_PCT_RANGE
    L += ["## Depth stress (STRESS only, not in the quote)", "",
          "```",
          "close_one_sided_share = phi x |net_sh| / (close_base_sh + phi x |net_sh|)    (expected closing auction = "
          "close_base_sh x exp_close_print_mult)",
          f"depth_stress_bp       = {DEPTH_BP_PER_PCT:g} bp x 100 x close_one_sided_share      ({lo:g}-{hi:g} bp per 1%)",
          "```", "",
          "phi = share of the demand printing in the auction (CA 0.71, EU 0.58); close_base_sh = normal closing-auction "
          "volume. The prior is SIX / University of St. Gallen (2020): removing 10% of one side of a Swiss blue-chip "
          "auction moves the close 8–12 bp (`X`, closing_auction_literature.md), extrapolated here far beyond 10%. "
          "Our own close_dislocation study finds that flow/ADV does **not** raise the close move (RESULTS §5), so "
          "this column is a stress for thin or mispredicted auctions, not a central charge.", ""]
    ds = s[np.isfinite(s.depth_stress_bp)].sort_values("depth_stress_bp", ascending=False).head(10)
    L.append(_tbl(ds, ["ticker", "name", "side", "close_mult", "close_one_sided_share", "depth_stress_bp",
                       "quote_adj_hold_T1open_bp"],
                  {"name": lambda v: str(v)[:28], "close_mult": lambda v: f"{v:,.1f}x" if np.isfinite(v) else "n/a",
                   "close_one_sided_share": lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a",
                   "depth_stress_bp": f0, "quote_adj_hold_T1open_bp": f0}))
    if np.isfinite(s.depth_stress_bp).any():
        L.append(f"Median over the sheet: {s.depth_stress_bp.median():.0f} bp. Where central demand is many times the "
                 "normal auction (`close_mult` >> 1) the share tends to 100% and the stress to its 100 bp ceiling, so "
                 "the column then flags thin auctions but does not rank names.\n")
    return L


def fm_eur(v) -> str:
    """Signed euro amount: -€1.2bn / €340.0m."""
    if v is None or not np.isfinite(v):
        return "n/a"
    return ("-" if v < 0 else "") + "€" + fmt_bn(abs(v))


def summary_md(res: dict, notes: list[str]) -> str:
    s, c, pis = res["sheet"], res["contrib"], res["pis"]
    f0 = lambda v: f"{v:,.0f}" if np.isfinite(v) else "n/a"   # noqa: E731
    fm = lambda v: fmt_bn(v)                                    # noqa: E731
    fx = lambda v: f"{v:,.1f}x" if np.isfinite(v) else "n/a"    # noqa: E731
    fp = lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a"      # noqa: E731
    L = [f"# Net index flow on the {res['close_date']} close ({res['status']})", "",
         f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `16_flow_engine/engine.py`. "
         "Demand = pi_true x change in index shares, pi_true = R x visible pi (flow_sizing.md §4.4). "
         "Visible-tracker demand (lower bound) is in the CSV (`net_sh_visible`). Gross of crowding: pre-positioned "
         "inventory is not netted (crowding_and_timing.md).", ""]
    L += ["## Inputs used", ""]
    inp = c.groupby(["index", "status"]).agg(rows=("key", "size"), types=("change_type", lambda v: ", ".join(
        f"{k} {n}" for k, n in v.value_counts().items())), source=("source", lambda v: v.iloc[0][:90])).reset_index()
    L.append(_tbl(inp, ["index", "status", "rows", "types", "source"], {}))
    L += ["## Passive shares used", ""]
    p = pis.copy()
    p["pi_visible"] = p.pi_visible.map(lambda v: f"{v:.3%}" if np.isfinite(v) else "MISSING")
    p["pi_true"] = p.pi_true.map(lambda v: f"{v:.3%}" if np.isfinite(v) else "MISSING")
    p["R"] = p.R.map(lambda v: f"{v:.1f}" if np.isfinite(v) else "none")
    p["pi_date"] = p.pi_date.astype(str).str[:10]
    p["flags"] = [("FALLBACK" if a else "") + (" R assumed" if b else "") for a, b in zip(p.fallback, p.R_assumed)]
    L.append(_tbl(p, ["index", "pi_visible", "R", "pi_true", "pi_date", "flags", "pi_source"], {}))
    if pis.fallback.any():
        L.append("**Fallback pi in use** for the flagged indices: `../12_studies/net_flow/pi_calibration.csv` was "
                 "not found or has no row for them on or before this date.\n")
    L += ["## Notional by market (central estimate)", ""]
    if "notional_eur" in s and s.notional_eur.notna().any():
        mk = s.assign(buy=np.where(s.net_sh > 0, s.notional_eur, 0), sell=np.where(s.net_sh < 0, s.notional_eur, 0))
        t = mk.groupby("country").agg(names=("key", "size"), buy_eur=("buy", "sum"), sell_eur=("sell", "sum"),
                                      net_eur=("notional_eur", "sum"), gross_eur=("gross_notional_eur", "sum"),
                                      exp_net_eur=("exp_notional_eur", "sum")).reset_index()
        t = t.sort_values("gross_eur", ascending=False)
        L.append(_tbl(t, ["country", "names", "buy_eur", "sell_eur", "net_eur", "gross_eur", "exp_net_eur"],
                      {k: fm for k in ["buy_eur", "sell_eur", "net_eur", "gross_eur", "exp_net_eur"]}))
        L.append(f"Total: buy €{fmt_bn(t.buy_eur.sum())}, sell €{fmt_bn(t.sell_eur.sum())}, "
                 f"gross €{fmt_bn(t.gross_eur.sum())} (sum over stocks of the gross index trades).\n")
        nomd = s.adv_basis.astype(str).str.startswith("not in ") if "adv_basis" in s else pd.Series(False, s.index)
        if nomd.any():
            L.append(f"**Not in these totals: {int(nomd.sum())} name(s) with no market data ({_names(s.ticker[nomd])}; "
                     "see Notes).**\n")
    else:
        L.append("_no prices/FX: notional not computed_\n")
    L += ["## Largest net imbalances", ""]
    top = s.head(20)
    L.append("`notional_eur` assumes every change happens; `exp_notional_eur` weights predicted rows by `prob`. "
             "`pct_adv` = |net| / consolidated ADV (`pct_adv_visible` in the CSV uses visible-tracker demand).\n")
    L.append(_tbl(top, ["ticker", "name", "country", "side", "net_sh", "notional_eur", "exp_notional_eur", "pct_adv",
                        "close_mult", "min_prob", "contributions"],
                  {"net_sh": f0, "notional_eur": fm, "exp_notional_eur": fm, "pct_adv": fx, "close_mult": fx,
                   "min_prob": fp, "name": lambda v: str(v)[:28]}))
    L += ["## Largest close multiples (|net| / normal closing auction)", ""]
    cm = s[np.isfinite(s.close_mult)].sort_values("close_mult", ascending=False).head(12)
    L.append(_tbl(cm, ["ticker", "name", "side", "net_sh", "close_base_sh", "close_mult", "close_basis",
                       "close_n_days"],
                  {"net_sh": f0, "close_base_sh": f0, "close_mult": fx, "name": lambda v: str(v)[:28],
                   "close_n_days": lambda v: f"{v:.0f}" if v == v else ""}))
    L += ["## Names where indices offset", ""]
    off = s[s.offset].sort_values("gross_notional_eur", key=lambda v: v.abs(), ascending=False).head(20)
    L.append(_tbl(off, ["ticker", "name", "net_sh", "gross_sh", "offset_ratio", "contributions"],
                  {"net_sh": f0, "gross_sh": f0, "offset_ratio": fp, "name": lambda v: str(v)[:28]}))
    L += pricing_sections_md(s, res["close_date"])
    if notes:
        L += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", required=True)
    ap.add_argument("--status", default="best", choices=["best", "predicted", "announced", "final"])
    ap.add_argument("--include-us", action="store_true")
    ap.add_argument("--md", default="live", help="live (yfinance/TSX archive) or none")
    ap.add_argument("--md-from", default=None,
                    help="frozen market data (md CSV or a sheet CSV): re-price without new market data or web requests")
    ap.add_argument("--label", default=None, help="sheet label (default: the status)")
    ap.add_argument("--files", default=None, help="comma-separated input files in inputs/<close>/ (default: all)")
    ap.add_argument("--notes-from", default=None, help="keep the '## Notes' bullets of an earlier sheet .md")
    a = ap.parse_args()
    notes = []
    if a.notes_from:
        txt = open(a.notes_from).read()
        if "## Notes" in txt:
            notes = [ln[2:] for ln in txt.split("## Notes", 1)[1].splitlines() if ln.startswith("- ")]
    md = None
    if a.md_from:
        md = load_md(a.md_from)
        notes.append(f"Market data frozen from `{a.md_from}` (`--md-from`; no new market data).")
    r = run(a.close, a.status, a.include_us, md=md, md_source="frozen" if md is not None else a.md, label=a.label,
            extra_notes=notes, files=a.files.split(",") if a.files else None)
    print("\n".join(r.get("paths", [])))
    print(r["sheet"][["ticker", "name", "side", "net_sh", "notional_eur", "pct_adv", "close_mult"]].head(15)
          .to_string(index=False))
