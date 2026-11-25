"""Step 1: download iShares UCITS daily holdings (ishares.com/ch JSON, asOfDate) around every MSCI review 2018-02 ..
2026-08 and at every quarter-end 2017-12 .. 2026-09.

Funds (common.CH_FUNDS): CSCA (MSCI Canada), IMEA (MSCI Europe), SWDA (MSCI World), WSML (MSCI World Small Cap,
from Mar 2018). Around each review T we take T-2, T-1, T and T+1 (weekdays; a holiday returns an empty file and the
pair builder steps back). Holdings "as of T" already show the trades booked at the T close (checked on the known
Aug 2026 Canada migrations, s03). Quarter-end files feed the reference-fund ownership fractions (s02).
Politeness: polite.get (>= 1.6 s between requests across processes, neutral User-Agent, no identifiers). Every
response is cached in data/ishch/ (empty or failed responses as .empty), so reruns make no requests.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import polite  # noqa: E402
from common import CH_FUNDS, CH_URL, ISHCH, reviews, weekday_after, weekdays_before  # noqa: E402


def fetch(fund: str, d) -> bool:
    d = pd.Timestamp(d)
    meta = CH_FUNDS[fund]
    if "start" in meta and d < pd.Timestamp(meta["start"]):
        return False
    p = ISHCH / f"{fund}_{d:%Y%m%d}.json"
    pe = ISHCH / f"{fund}_{d:%Y%m%d}.empty"
    if p.exists():
        return True
    if pe.exists():
        return False
    s, b = polite.get(CH_URL.format(pid=meta["pid"], d=d))
    ok = False
    if s == 200 and b.lstrip(b"\xef\xbb\xbf")[:1] == b"{":
        try:
            ok = len(json.loads(b.decode("utf-8-sig")).get("aaData", [])) > 5
        except ValueError:
            ok = False
    (p if ok else pe).write_bytes(b if ok else f"HTTP {s} len {len(b)}".encode())
    return ok


def main():
    jobs = []
    for r in reviews().itertuples():
        days = weekdays_before(r.T, 3)[::-1] + [r.T, weekday_after(r.T)]     # T-3 .. T+1
        for f in CH_FUNDS:
            for d in days[1:]:
                jobs.append((f, d))
            jobs.append((f, days[0], (days[1], days[2])))
    for q in pd.date_range("2017-12-31", "2026-09-30", freq="QE"):
        d = q
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        for f in CH_FUNDS:
            jobs.append((f, d))
    n = 0
    for j in jobs:
        f, d = j[0], j[1]
        if len(j) == 3:      # T-3 only when both T-2 and T-1 are empty
            if any((ISHCH / f"{f}_{pd.Timestamp(x):%Y%m%d}.json").exists() for x in j[2]):
                continue
        ok = fetch(f, d)
        n += 1
        if n % 25 == 0:
            print(n, f, d.date(), ok, flush=True)
    print("done", n)


if __name__ == "__main__":
    main()
