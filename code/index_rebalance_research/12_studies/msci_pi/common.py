"""Shared paths, the MSCI review calendar, fund registry and statistics for the msci_pi study (12_studies/msci_pi/).

Run every script from <repo>/code with PYTHONPATH=<repo>/code
(run_all.sh sets it) so `import r2` resolves. R2 is read-only. Sister studies are read, never written:
  ../net_flow/            fe_ols / winsor (common.py), pi_calibration.csv (we only APPEND rows, s07)
  ../tsx_composite/       Canadian daily panel (data/panel.npz: P, VC consolidated volume, SH, AJ; panel_lines.parquet)
  ../stoxx600_dax/        European daily panel (data/panel.npz: P, VP primary volume, SH, AJ, FX; panel_lines.parquet)
Writes only to this folder (data/ = downloads and intermediates; deliverables at the top level) and appends rows to
../net_flow/pi_calibration.csv (s07, after a timestamped backup in data/backup/).
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ISHCH = DATA / "ishch"
OUT = HERE
STUDIES = HERE.parent
NF = STUDIES / "net_flow"
TSX = STUDIES / "tsx_composite"
EU = STUDIES / "stoxx600_dax"
PI_CAL = NF / "pi_calibration.csv"
for _d in (DATA, ISHCH):
    _d.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_nf = _load("nf_common", NF / "common.py")
fe_ols, winsor = _nf.fe_ols, _nf.winsor

# ------------------------------------------------------------------------------------------- calendar
# MSCI review implementation closes (trade date T = the close at which index funds trade; MSCI "effective" = T+1).
# Source: MSCI review press releases / the published effective dates (11_methodology/specs/msci/gimi_size_segments.md
# §5: Nov 2025 = 24 Nov, May 2027 = 27 May), each CONFIRMED here by the volume spike of the European panel
# (median log V_T/ADV across all lines, s00 output in data/review_dates_check.csv) and by the iShares UCITS
# holdings diff T-1 -> T. Feb 2018 and Aug 2019 show no volume spike (small quarterly reviews); their T is taken
# from the holdings diff.
REVIEWS = {
    "2018-02": "2018-02-28", "2018-05": "2018-05-31", "2018-08": "2018-08-31", "2018-11": "2018-11-30",
    "2019-02": "2019-02-28", "2019-05": "2019-05-28", "2019-08": "2019-08-27", "2019-11": "2019-11-26",
    "2020-02": "2020-02-28", "2020-05": "2020-05-29", "2020-08": "2020-08-31", "2020-11": "2020-11-30",
    "2021-02": "2021-02-26", "2021-05": "2021-05-27", "2021-08": "2021-08-31", "2021-11": "2021-11-30",
    "2022-02": "2022-02-28", "2022-05": "2022-05-31", "2022-08": "2022-08-31", "2022-11": "2022-11-30",
    "2023-02": "2023-02-28", "2023-05": "2023-05-31", "2023-08": "2023-08-31", "2023-11": "2023-11-30",
    "2024-02": "2024-02-29", "2024-05": "2024-05-31", "2024-08": "2024-08-30", "2024-11": "2024-11-25",
    "2025-02": "2025-02-28", "2025-05": "2025-05-30", "2025-08": "2025-08-26", "2025-11": "2025-11-24",
    "2026-02": "2026-02-27", "2026-05": "2026-05-29", "2026-08": "2026-08-31",
}
R2_END = pd.Timestamp("2026-06-24")      # last date with Canadian consolidated volume; EU panel to 2026-06-30


def reviews() -> pd.DataFrame:
    r = pd.DataFrame([(k, pd.Timestamp(v)) for k, v in REVIEWS.items()], columns=["review", "T"])
    r["in_r2"] = r["T"] <= R2_END
    return r


def weekdays_before(d, n):
    d = pd.Timestamp(d)
    out = []
    while len(out) < n:
        d -= pd.Timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def weekday_after(d):
    d = pd.Timestamp(d) + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


# ---------------------------------------------------------------------------------------- funds (CH JSON)
# iShares UCITS funds whose daily holdings history is served by ishares.com/ch (asOfDate=YYYYMMDD; checked
# 2026-10-12 for 2019 and 2026 dates). Segment = MSCI size segment of the benchmark.
CH_FUNDS = {
    "CSCA": dict(pid=253721, index="MSCI Canada", segment="STD", market="CA"),
    "IMEA": dict(pid=251861, index="MSCI Europe", segment="STD", market="EU"),
    "SWDA": dict(pid=251882, index="MSCI World", segment="STD", market="WORLD"),
    "WSML": dict(pid=296576, index="MSCI World Small Cap", segment="SC", market="WORLD", start="2018-03-27"),
}
CH_URL = "https://www.ishares.com/ch/individual/en/products/{pid}/x/1495092304805.ajax?tab=all&fileType=json&asOfDate={d:%Y%m%d}"

# DM Europe (MSCI Europe) countries, ISO2 of the listing / ISIN country used for classification of fund lines.
EU_CTRY = ["GB", "FR", "DE", "CH", "NL", "SE", "ES", "IT", "DK", "BE", "FI", "NO", "IE", "AT", "PT"]
EUROZONE = {"FR", "DE", "NL", "ES", "IT", "BE", "FI", "IE", "AT", "PT"}
