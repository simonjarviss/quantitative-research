"""Shared config for the Russell June reconstitution study.

Run every script from <repo>/code with
PYTHONPATH=<repo>/code so `import r2` resolves.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

STUDY = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("RUSSELL_CACHE", STUDY / "_cache"))   # raw pulls from R2 (deletable; rebuilt by s01_fetch.py)
OUT = STUDY / "output"                                             # deliverable CSVs
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

R1000, R2000, R3000 = 6967, 6968, 6969          # Datastream index-list codes (month-end, survivorship-free)
IWM_PERMNO = 88222                              # iShares Russell 2000 ETF in CRSP (securitytype FUND)

# 14 multi-manager / quant / event pods (17 13F filer numbers), checked in tr_13f.s34 2026-10-08
PODS = {57986: "Millennium", 16210: "Citadel Advisors", 7808: "Balyasny", 14335: "Schonfeld",
        26244: "ExodusPoint", 75106: "Point72", 10736: "Jane Street", 11912: "Hudson Bay",
        12274: "Verition", 10643: "Marshall Wace", 8853: "Squarepoint", 22891: "Walleye",
        23308: "Walleye", 17779: "Walleye", 10751: "Two Sigma", 12759: "Two Sigma", 28088: "Qube"}

# June reconstitution calendar.  T = recon trade day (changes effective after its close).
# src codes: OFF = official FTSE Russell schedule (press release / notice) fetched 2026-10-08;
#            NYSE = NYSE info memo; RULE = derived from the ground rules; PAT = assumed from the
#            pattern of neighbouring years (rank day 6-7 Fridays before T; prelim 2 / 3 weeks before T),
#            then checked empirically in s03 (rank-day fit test).
CAL = pd.DataFrame([
    # year rank_day      rank_src prelim        prelim_src T
    (2008, "2008-05-30", "RULE-endMay", "2008-06-13", "PAT", "2008-06-27"),
    (2009, "2009-05-29", "RULE-endMay", "2009-06-12", "PAT", "2009-06-26"),
    (2010, "2010-05-28", "RULE-endMay", "2010-06-11", "PAT", "2010-06-25"),
    (2011, "2011-05-31", "RULE-endMay", "2011-06-10", "PAT", "2011-06-24"),
    (2012, "2012-05-31", "RULE-endMay", "2012-06-08", "PAT", "2012-06-22"),
    (2013, "2013-05-31", "RULE-endMay", "2013-06-14", "PAT", "2013-06-28"),
    (2014, "2014-05-30", "RULE-endMay", "2014-06-13", "PAT", "2014-06-27"),
    (2015, "2015-05-29", "RULE-endMay", "2015-06-12", "PAT", "2015-06-26"),
    (2016, "2016-05-27", "OFF", "2016-06-10", "OFF", "2016-06-24"),
    (2017, "2017-05-12", "OFF", "2017-06-09", "NYSE", "2017-06-23"),
    (2018, "2018-05-11", "PAT", "2018-06-08", "PAT", "2018-06-22"),
    (2019, "2019-05-10", "OFF", "2019-06-07", "OFF", "2019-06-28"),
    (2020, "2020-05-08", "OFF", "2020-06-05", "OFF", "2020-06-26"),
    (2021, "2021-05-07", "PAT", "2021-06-04", "PAT", "2021-06-25"),
    (2022, "2022-05-06", "OFF", "2022-06-03", "OFF", "2022-06-24"),
    (2023, "2023-04-28", "OFF", "2023-05-19", "OFF", "2023-06-23"),
    (2024, "2024-04-30", "OFF", "2024-05-24", "OFF", "2024-06-28"),
    (2025, "2025-04-30", "RULE", "2025-05-23", "RULE", "2025-06-27"),
], columns=["year", "rank_day", "rank_src", "prelim", "prelim_src", "T"])
for c in ["rank_day", "prelim", "T"]:
    CAL[c] = pd.to_datetime(CAL[c])
CAL = CAL.set_index("year")

MAIN_YEARS = list(range(2016, 2026))
EXT_YEARS = list(range(2008, 2016))
HORIZONS = ["H6", "H3", "HR"]                 # end-Dec, end-Mar, rank day
GROUPS = ["BUY_add", "BUY_mig", "SELL_del", "SELL_mig"]
SIDE = {"BUY_add": 1, "BUY_mig": 1, "SELL_del": -1, "SELL_mig": -1}


def session():
    import r2
    return r2.session()


def month_end(d) -> pd.Timestamp:
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).normalize()


def snapshots(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(pre, post) month-end membership snapshots around the June recon.
    pre = end-May. post = end-June, or end-July when T is the last trading day of June
    (2013, 2019, 2024): the end-June snapshot then still shows pre-recon membership."""
    T = CAL.loc[year, "T"]
    last_june_bd = pd.Timestamp(year, 6, 30)
    while last_june_bd.weekday() >= 5:
        last_june_bd -= pd.Timedelta(days=1)
    post = pd.Timestamp(year, 7, 31) if T == last_june_bd else pd.Timestamp(year, 6, 30)
    return pd.Timestamp(year, 5, 31), post
