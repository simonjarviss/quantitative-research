"""Shared paths, review calendar, legs and statistics for the msci_close study (12_studies/msci_close/).

Question: how the close moves on MSCI review closes for MSCI Canada and MSCI Europe Standard-index changes, and how
fast it reverts. Design follows ../close_dislocation/ (legs, sign convention, matched-control cells, summ() and the
review-clustered t). Events and visible passive demand come from ../msci_pi/ (demand.csv, REVIEWS).

Run every script from <repo>/code with PYTHONPATH=<repo>/code
(run_all.sh sets it) so `import r2` resolves. R2 is read-only. Sister studies are read, never written:
  ../msci_pi/            common.REVIEWS (MSCI implementation closes T), demand.csv (events, D_vis, ADV60)
  ../close_dislocation/  common.summ / clt (statistics), legs.py (leg arithmetic)
  ../tsx_composite/      Canadian daily panel (data/panel.npz, panel_lines.parquet, ds_names.parquet, calendar,
                         member_matrix.npy = S&P/TSX Composite members)
  ../stoxx600_dax/       European daily panel (data/panel.npz, panel_lines.parquet, eu_calendar, member_stoxx.npy)
Writes only to this folder (data/ = R2 extracts and intermediates; deliverables at the top level).
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE
STUDIES = HERE.parent
MPI = STUDIES / "msci_pi"
CD = STUDIES / "close_dislocation"
TSX = STUDIES / "tsx_composite"
EU = STUDIES / "stoxx600_dax"
DATA.mkdir(exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_cd = _load("cd_common", CD / "common.py")          # close_dislocation statistics, unchanged
summ, clt, trimmed = _cd.summ, _cd.clt, _cd.trimmed
CLIP = _cd.CLIP                                      # daily return clip [-50%, +100%]
LEGS = _cd.LEGS
legs_mod = _load("cd_legs", CD / "legs.py")          # daily_legs / split_session / subday_legs / flatten, unchanged
KS, K0 = legs_mod.KS, legs_mod.K0

_mp = _load("mp_common", MPI / "common.py")
REVIEWS = _mp.REVIEWS                                # MSCI implementation closes (T)
DS_END = pd.Timestamp("2026-06-24")                  # last date of ds2primqtprc / ds2primqtri

TSX_REDESIGN = pd.Timestamp("2021-10-18")            # first close under the new TSX MOC rules
STD_EVENTS = ["ADD_STD", "DEL_STD", "MIG_UP", "MIG_DOWN"]
ALL_INDEX_EVENTS = STD_EVENTS + ["ADD_SC", "DEL_SC"]
MATERIAL = 0.001          # drop events with |net visible demand| < 0.10% of shares outstanding
UK_HOLIDAY_T = {pd.Timestamp("2020-08-31")}          # UK closed on T: UK lines implemented a day later (dropped)
DM_EUROPE = ["GBR", "FRA", "DEU", "CHE", "NLD", "SWE", "ESP", "ITA", "DNK", "BEL", "FIN", "NOR", "IRL", "AUT", "PRT"]

LEG_SHOW = ["d0", "d0_on", "d0_id", "on1", "id1", "r1", "r1_5", "r1_20"]


def reviews() -> pd.DataFrame:
    r = pd.DataFrame([(k, pd.Timestamp(v)) for k, v in REVIEWS.items()], columns=["review", "T"])
    return r[r["T"] <= DS_END].reset_index(drop=True)


def ca_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(TSX / "data" / "tsx_calendar.parquet").date)


def eu_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(EU / "data" / "eu_calendar.parquet").date)


def window_dates(cal: pd.DatetimeIndex, Ts, lo=-7, hi=21) -> list:
    out = set()
    for T in Ts:
        i = cal.get_loc(pd.Timestamp(T))
        for k in range(lo, hi + 1):
            if 0 <= i + k < len(cal):
                out.add(cal[i + k].date())
    return sorted(out)


def period_ca(T) -> str:
    return "Oct2021+" if pd.Timestamp(T) >= TSX_REDESIGN else "pre-Oct2021"
