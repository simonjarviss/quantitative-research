"""Step 1 (Canada): download iShares Canada XIC (S&P/TSX Capped Composite) and XIU (S&P/TSX 60) holdings around
every quarterly S&P/TSX review, 2006-03 .. 2026-09.

Source `ishca` (06_data/free_public_sources.md): BlackRock Canada holdings CSV with `asOfDate=YYYYMMDD`.
Probe 2026-10-11: daily dates work from about 2011-2015; before that only month-end dates return data
(e.g. XIC 2010-09-17 empty, 2010-09-30 235 rows; XIU 2009-06-15 empty, 2009-06-30 60 rows).

Per review and ETF we try T-1 and T (holdings "as of T" already show the review trades booked at the T close;
checked on Jun 2026: ALS/EXE/SDE/SXGC/HPS.A present on 19 Jun, GSY/PET/TCL.A/WPK gone). When both exist we also take
T-2 (a no-event placebo pair T-2 -> T-1) and T+1 (late bookings). When either is empty we fall back to the last
session of the month before the review month and of the review month (month-end pair, noisier), and when that is
empty too, to the nearest available pair spanning T (T-2..T-6 and T+1..T+5).

Politeness: >= 1.6 s between requests, neutral User-Agent, no identifiers. Every response is cached under
data/ishca/ (empty responses as *.empty) so reruns make no requests.
Run: PYTHONPATH=<repo>/code python3 s01_fetch_ishca.py
"""
from __future__ import annotations
import time
import urllib.request
from pathlib import Path
import pandas as pd

from common import DATA, tsx_sessions, tsx_reviews

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
ETF = {"XIC": ("239837", "ishares-core-sptsx-capped-composite-index-etf"),
       "XIU": ("239832", "ishares-sptsx-60-index-etf")}
OUTDIR = DATA / "ishca"
OUTDIR.mkdir(parents=True, exist_ok=True)
_last = [0.0]


def url(tk, d):
    pid, slug = ETF[tk]
    return (f"https://www.blackrock.com/ca/investors/en/products/{pid}/{slug}/1464253357814.ajax?"
            f"fileType=csv&fileName={tk}_holdings&dataType=fund&asOfDate={d:%Y%m%d}")


def fetch(tk, d) -> bool:
    """Download holdings of ETF tk as of date d; return True if the file has equity rows."""
    p = OUTDIR / f"{tk}_{d:%Y%m%d}.csv"
    pe = OUTDIR / f"{tk}_{d:%Y%m%d}.empty"
    if p.exists():
        return True
    if pe.exists():
        return False
    wait = 1.6 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    body = b""
    for k in range(3):
        try:
            req = urllib.request.Request(url(tk, d), headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            break
        except Exception as e:  # noqa: BLE001
            print("  retry", tk, d.date(), e, flush=True)
            time.sleep(5 * (k + 1))
    _last[0] = time.time()
    ok = b'"Equity"' in body
    if ok:
        p.write_bytes(body)
    else:
        pe.write_bytes(body[:400])
    return ok


def main():
    cal = tsx_sessions()
    rv = tsx_reviews()
    n_ok = 0
    for r in rv.itertuples():
        i = cal.get_loc(r.T)
        days = {"Tm1": cal[i - 1], "T": cal[i]}
        for tk in ETF:
            got = {k: fetch(tk, d) for k, d in days.items()}
            if all(got.values()):
                fetch(tk, cal[i - 2]); fetch(tk, cal[i + 1])
                mode = "daily"
            else:
                m0 = pd.Timestamp(r.T.year, r.T.month, 1)
                pm = cal[cal.searchsorted(m0) - 1]                       # last session of the previous month
                me = cal[cal.searchsorted(m0 + pd.offsets.MonthEnd(0), side="right") - 1]
                a, b = fetch(tk, pm), fetch(tk, me)
                mode = "month-end" if a and b else "none"
                if mode == "none":
                    # nearest pair spanning T: last available of T-2..T-6, first available of T+1..T+5
                    bb = next((cal[i - k] for k in range(2, 7) if fetch(tk, cal[i - k])), None)
                    aa = next((cal[i + k] for k in range(1, 6) if fetch(tk, cal[i + k])), None)
                    mode = "nearest" if bb is not None and aa is not None else "none"
            n_ok += mode != "none"
            print(f"{r.review} {tk} {mode}", flush=True)
    print("done", n_ok)


if __name__ == "__main__":
    main()
