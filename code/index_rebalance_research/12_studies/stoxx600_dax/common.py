"""Shared helpers: paths, calendars, review timelines, rule parameters by era, countries, tax regimes, stats.

Rule sources: 11_methodology/specs/stoxx/stoxx_europe_600.md, dax_family.md, _shared.md and params/*.yaml
(STOXX-MG Sep 2026, DAX-MG v12.7.3). The specs carry the current rules and the change log since 2023 only.
Older rule versions used below are flagged in RESULTS.md as either (a) current rule applied backward or
(b) inferred from the data / from the structure of the current rank table (not from an archived guide).
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
warnings.filterwarnings("ignore", category=RuntimeWarning)          # all-NaN slices in nan-aggregates are expected
warnings.filterwarnings("ignore", category=FutureWarning)

FIRST_REVIEW = (2008, 3)
LAST_REVIEW = (2026, 6)
START = "2006-06-01"          # 12m volume + 6m momentum before the first H6 (Sep 2007)

# ---------------------------------------------------------------- countries
# listing-country codes (Compustat g_security.excntry) and their main exchange codes
MAIN_EXCH = {194: "GBR", 171: "DEU", 286: "FRA", 104: "NLD", 132: "BEL", 192: "PRT", 151: "CHE", 256: "SWE",
             144: "DNK", 167: "FIN", 228: "NOR", 209: "ITA", 201: "ESP", 273: "AUT", 172: "IRL", 276: "POL",
             107: "GRC", 237: "ISL", 235: "CZE", 198: "LUX", 154: "DEU", 163: "DEU", 149: "DEU", 280: "CHE"}
CORE = ["GBR", "DEU", "FRA", "CHE", "SWE", "NLD", "ITA", "ESP", "DNK", "NOR", "FIN", "BEL", "AUT", "IRL", "PRT", "LUX"]


def stoxx_country_ok(cty: str, review: str) -> bool:
    """STOXX Europe country eligibility by review. Core developed countries always; the fringe calendar is
    read from Compustat 150376 membership (first/last member dates): ISL to the Sep 2008 review, GRC to the
    Jun 2016 review, CZE Sep 2013 - Jun 2018, POL from Sep 2018. Inferred from data (published STOXX
    country decisions, known in advance), not from the spec."""
    if cty in CORE:
        return True
    y, m = int(review[:4]), int(review[5:])
    ym = (y, m)
    return {"ISL": ym <= (2008, 9), "GRC": ym <= (2016, 6), "CZE": (2013, 9) <= ym <= (2018, 6),
            "POL": ym >= (2018, 9)}.get(cty, False)


EURO = {"DEU", "FRA", "NLD", "BEL", "PRT", "ITA", "ESP", "AUT", "IRL", "FIN", "LUX", "GRC"}


def bloc(cty: str) -> str:
    """Currency bloc used when a country has too few controls."""
    if cty in EURO:
        return "EUR"
    if cty in ("SWE", "DNK", "NOR", "ISL", "FIN"):
        return "NORDIC"
    if cty in ("POL", "CZE"):
        return "CEE"
    return {"GBR": "GBP", "CHE": "CHF"}.get(cty, "OTHER")


def buy_tax(cty: str, d) -> float:
    """Tax on a cash buy at date d (fraction), by listing country, per 03_microstructure/transaction_taxes.md.
    UK 0.5% SDRT; IE 1% stamp duty; FR FTT 0.2% from 1 Aug 2012, 0.3% from 1 Jan 2017, 0.4% from 1 Apr 2025
    (FR rate history before 2025 is from memory; the file confirms 0.3% before Apr 2025 and 0.4% after);
    IT 0.1% on venue from 1 Mar 2013, 0.2% from 2026; ES 0.2% from 16 Jan 2021; BE flagged 'unconfirmed'
    (counted as taxed, 0.35% prior, unverified). Size thresholds (FR/ES > EUR 1bn, IT < EUR 500m exempt) ignored."""
    d = pd.Timestamp(d)
    if cty == "GBR":
        return 0.005
    if cty == "IRL":
        return 0.01
    if cty == "FRA":
        return 0.0 if d < pd.Timestamp("2012-08-01") else 0.002 if d < pd.Timestamp("2017-01-01") else \
            0.003 if d < pd.Timestamp("2025-04-01") else 0.004
    if cty == "ITA":
        return 0.0 if d < pd.Timestamp("2013-03-01") else 0.001 if d < pd.Timestamp("2026-01-01") else 0.002
    if cty == "ESP":
        return 0.0 if d < pd.Timestamp("2021-01-16") else 0.002
    if cty == "BEL":
        return 0.0035
    return 0.0


# ---------------------------------------------------------------- calendars
def nth_weekday(y, m, weekday, n):
    d = pd.Timestamp(y, m, 1)
    off = (weekday - d.weekday()) % 7
    return d + pd.Timedelta(days=off + 7 * (n - 1))


def last_on_or_before(cal, d):
    i = cal.searchsorted(pd.Timestamp(d), side="right") - 1
    return cal[i]


def first_on_or_after(cal, d):
    i = cal.searchsorted(pd.Timestamp(d), side="left")
    return cal[i] if i < len(cal) else pd.NaT


def first_after(cal, d):
    i = cal.searchsorted(pd.Timestamp(d), side="right")
    return cal[i] if i < len(cal) else pd.NaT


def eu_calendar() -> pd.DatetimeIndex:
    """Pan-European sessions = dates on which >= 300 panel lines carry a price (drops 1 Jan, Good Friday,
    Easter Monday, 25/26 Dec). Built by 03_panel.py."""
    return pd.DatetimeIndex(pd.read_parquet(DATA / "eu_calendar.parquet").date)


def xetra_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(DATA / "xetra_calendar.parquet").date)


def _quarters():
    y, m = FIRST_REVIEW
    while (y, m) <= LAST_REVIEW:
        yield y, m
        m += 3
        if m > 12:
            y, m = y + 1, m - 12


def review_dates(cal, family: str) -> pd.DataFrame:
    """One row per quarterly review with the reconstructed timeline (dates are sessions of `cal`).

    T   = 3rd-Friday close (last session on or before the 3rd Friday; STOXX-MG §5.4 / DAX-MG §4.3.1)
    eff = first session after T
    C   = ranking cut-off = last session of the month before the review month
          (STOXX-MG §7.2.2 / DAX-MG §4.2.1; current rule applied to all years)
    A   = announcement. STOXX: first trading day of the review month (STOXX-MG §5.4; time of day not stated;
          treated as after the close). DAX: 3rd trading day, after 22:00 CET (DAX-MG §4.3.2). Current rules
          applied backward to 2008.
    H3, H6 = last session on or before the 3rd Friday minus 3 / 6 calendar months.
    regular = DAX family: Regular Exit/Entry rules apply (Mar/Sep for MDAX/SDAX/TecDAX all years; for DAX
          Sep only before the Sep 2021 reform, Mar/Sep after; the pre-2021 DAX schedule is inferred).
    """
    rows = []
    for y, m in _quarters():
        t_nom = nth_weekday(y, m, 4, 3)
        T = last_on_or_before(cal, t_nom)
        eff = first_after(cal, T)
        C = last_on_or_before(cal, pd.Timestamp(y, m, 1) - pd.Timedelta(days=1))
        first = first_on_or_after(cal, pd.Timestamp(y, m, 1))
        i0 = cal.get_loc(first)
        A = first if family == "stoxx" else cal[i0 + 2]
        H3 = last_on_or_before(cal, t_nom - pd.DateOffset(months=3))
        H6 = last_on_or_before(cal, t_nom - pd.DateOffset(months=6))
        rows.append(dict(review=f"{y}-{m:02d}", year=y, month=m, H6=H6, H3=H3, C=C, A=A, T=T, eff=eff))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- DAX family rule eras
def dax_rules(review: str) -> dict:
    """Rank rules in force at a DAX-family review.

    era 3 (Sep 2021 ->): DAX-MG v12.7.3 table (spec §3): DAX 40, MDAX 50, SDAX 70, TecDAX 30; EBITDA screen for
           DAX newcomers; Regular rules in Mar and Sep for all four indices.  [spec]
    era 2 (Sep 2018 - Jun 2021): DAX 30, MDAX 60, SDAX 70, TecDAX 30 (tech allowed in DAX/MDAX/SDAX; TecDAX may
           hold DAX members - confirmed in Compustat: SAP/Infineon/Telekom enter TecDAX 2018-09-24). MDAX and
           SDAX boundaries (rank 90 and 160) equal today's, so today's MDAX/SDAX thresholds are used. DAX 30
           thresholds = the current 30-member TecDAX table (45/25/40(35)/30(35)). DAX regular rules Sep only.
           No EBITDA screen.  [inferred]
    era 1 (<= Jun 2018): DAX 30, MDAX 50, SDAX 50, TecDAX 30; MDAX/SDAX exclude technology names and TecDAX
           excludes DAX members. Thresholds: DAX/TecDAX as era 2; MDAX/SDAX = the current offsets around the
           era's boundary (fast exit B+20, fast entry B-7, regular exit B+13 (alt B+7), regular entry B (alt
           B+7)). The pre-2018 dual ranking (free-float cap AND order-book turnover) is not modelled; ranks are
           free-float cap only, with the liquidity screen.  [inferred / current concept applied backward]
    Thresholds: fx = fast exit (member out if rank > fx), fe = fast entry (rank <= fe), rx/rxa = regular exit
    (and its alternate candidate rank), re/rea = regular entry (and alternate).
    """
    y, m = int(review[:4]), int(review[5:])
    ym = (y, m)
    era = 3 if ym >= (2021, 9) else 2 if ym >= (2018, 9) else 1

    def std(B):
        return dict(fx=B + 20, fe=B - 7, rx=B + 13, rxa=B + 7, re=B, rea=B + 7)
    t30 = dict(fx=45, fe=25, rx=40, rxa=35, re=30, rea=35)
    if era == 3:
        n = dict(DAX=40, MDAX=50, SDAX=70, TecDAX=30)
        thr = dict(DAX=std(40), MDAX=std(90), SDAX=std(160), TecDAX=t30)
    elif era == 2:
        n = dict(DAX=30, MDAX=60, SDAX=70, TecDAX=30)
        thr = dict(DAX=t30, MDAX=std(90), SDAX=std(160), TecDAX=t30)
    else:
        n = dict(DAX=30, MDAX=50, SDAX=50, TecDAX=30)
        thr = dict(DAX=t30, MDAX=std(80), SDAX=std(130), TecDAX=t30)
    regular_dax = m == 9 or (era == 3 and m == 3)
    regular_other = m in (3, 9)
    return dict(era=era, n=n, thr=thr, ebitda=era == 3, tech_split=era == 1,
                regular=dict(DAX=regular_dax, MDAX=regular_other, SDAX=regular_other, TecDAX=regular_other))


# ---------------------------------------------------------------- statistics
def clt(x, g):
    """Mean / cluster-robust SE (clusters g), CR0 as in the TSX study: SE = sqrt(sum_g (sum e)^2) / N."""
    x = np.asarray(x, float); g = np.asarray(g); ok = ~np.isnan(x); x, g = x[ok], g[ok]
    n = len(x)
    if n < 5 or pd.Series(g).nunique() < 3:
        return np.nan
    e = pd.Series(x - x.mean()).groupby(g).sum()
    se = np.sqrt((e ** 2).sum()) / n
    return x.mean() / se if se > 0 else np.nan


def cell(x, g, scale=100):
    x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
    g = pd.Series(np.asarray(g)).reset_index(drop=True)
    ok = x.notna()
    if ok.sum() == 0:
        return "–"
    t = clt(x[ok], g[ok])
    ts = "" if np.isnan(t) else f" ({t:+.1f})"
    return f"{x[ok].mean()*scale:+.2f}{ts}"
