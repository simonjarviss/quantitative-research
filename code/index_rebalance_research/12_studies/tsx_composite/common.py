"""Shared helpers: paths, TSX calendar, review dates, rule parameters by era.

All rule values come from specs/spdji/sp_tsx_composite.md (SPTSX-M June 2025 archived edition,
change log App. I) and params/sp_tsx_composite.yaml. Where the spec gives no history the current
rule is applied backward and the fact is flagged in RESULTS.md.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE

FIRST_REVIEW = (2006, 3)
LAST_REVIEW = (2026, 6)


def tsx_calendar() -> pd.DatetimeIndex:
    """TSX sessions = dates on which >= 500 Canadian Compustat lines carry a price."""
    p = DATA / "tsx_calendar.parquet"
    if p.exists():
        return pd.DatetimeIndex(pd.read_parquet(p).date)
    s = pd.read_parquet(DATA / "secd.parquet", columns=["iid", "datadate"])
    s = s[s.iid.str.endswith("C")]
    n = s.groupby("datadate").size()
    cal = pd.DatetimeIndex(sorted(n[n >= 500].index))
    pd.DataFrame({"date": cal}).to_parquet(p, index=False)
    return cal


def nth_weekday(y, m, weekday, n):
    d = pd.Timestamp(y, m, 1)
    off = (weekday - d.weekday()) % 7
    return d + pd.Timedelta(days=off + 7 * (n - 1))


def last_on_or_before(cal, d):
    i = cal.searchsorted(pd.Timestamp(d), side="right") - 1
    return cal[i]


def first_after(cal, d):
    i = cal.searchsorted(pd.Timestamp(d), side="right")
    return cal[i] if i < len(cal) else pd.NaT


def review_dates(cal) -> pd.DataFrame:
    """One row per quarterly review with the reconstructed timeline.

    T   = 3rd-Friday close (last TSX session on or before the 3rd Friday)
    eff = first session after T (effective open)
    R   = reference date. Reviews from Sep 2021: 10 TSX business days before the first Friday
          (SPTSX-M p.9; change log 20-Sep-21). Before: last trading day of Feb/May/Aug/Nov (change log).
    A   = reconstructed announcement (after the close). From Sep 2021: first Friday (releases Sep 2021,
          Dec 2023, Sep 2024-Jun 2026 in spec §5.2). Before: second Friday = T-5 sessions (releases
          Jun 2019, Jun 2020 only; earlier years assumed).
    H3, H6 = last session on or before T_nominal minus 3 / 6 calendar months.
    """
    rows = []
    y, m = FIRST_REVIEW
    while (y, m) <= LAST_REVIEW:
        t_nom = nth_weekday(y, m, 4, 3)
        T = last_on_or_before(cal, t_nom)
        eff = first_after(cal, T)
        f1 = nth_weekday(y, m, 4, 1)
        new_regime = (y, m) >= (2021, 9)
        if new_regime:
            i = cal.searchsorted(f1)          # index of first session >= F1
            R = cal[i - 10]
            A = last_on_or_before(cal, f1)
        else:
            pm = pd.Timestamp(y, m, 1) - pd.Timedelta(days=1)
            R = last_on_or_before(cal, pm)
            A = last_on_or_before(cal, nth_weekday(y, m, 4, 2))
        H3 = last_on_or_before(cal, t_nom - pd.DateOffset(months=3))
        H6 = last_on_or_before(cal, t_nom - pd.DateOffset(months=6))
        rows.append(dict(review=f"{y}-{m:02d}", year=y, month=m, H6=H6, H3=H3, R=R, A=A, T=T, eff=eff,
                         regime="F1/R10" if new_regime else "F2/month-end"))
        m += 3
        if m > 12:
            y, m = y + 1, m - 12
    return pd.DataFrame(rows)


def rules_for(review: str) -> dict:
    """Rule parameters in force at a review (spec §2.4 change log; earlier = current rule applied backward)."""
    y, m = int(review[:4]), int(review[5:])
    ym = (y, m)
    return dict(
        size_entry=0.0005 if ym < (2019, 3) else 0.0004,      # 15-Mar-19: 0.05% -> 0.04%
        size_stay=0.00025,
        turn_entry=0.50, turn_stay=0.25,
        us_volume=ym >= (2017, 9),                            # 04-Aug-17: Canada+US, dual-listed CA-only test
        ca_only_entry=0.25, ca_only_stay=0.125,
        vwap_days=3 if ym < (2017, 12) else 10,               # 30-Oct-17: 3-day -> 10-day VWAP
        domicile="dom_or_inc" if ym >= (2025, 6) else "dom",
    )
