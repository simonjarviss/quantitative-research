"""Fetch ProShares' own daily history of each fund's shares outstanding and assets.

    python3 desk/results/ideas/levered_etf_close/fetch_proshares.py

WHY. CRSP (R2) ends 2025-12-31, and the test needs each leveraged fund's assets as of the
session before every decision, including 2026. ProShares publishes, per fund, a CSV of every
day's NAV, shares outstanding (thousands) and assets under management:
    https://accounts.profunds.com/etfdata/ByFund/<TICKER>-historical_nav.csv
(read 2026-11-24; source note in desk/docs/REFERENCES.md section 12le). It is the issuer's own
record, so it is a primary source. It covers ProShares funds only: Direxion, GraniteShares,
Tradr, Defiance, T-Rex and the rest publish today's holdings, not a history, so for 2026 the
flow estimate is ProShares-only and says so (explore.py measures the share of each
underlying's 2025 leveraged assets that ProShares carries).

WHICH TICKERS. Every fund ticker in the levered_etf_reversal mapping (pairs.parquet, CRSP's
last ticker of each fund). Only an HTTP 404 is read as "not a ProShares fund" and recorded so.
Any other HTTP error, a network failure, or a 200 whose first line is not the expected header
raises, naming the ticker: a server fault is never recorded as a fund ProShares does not run
(review, 2026-11-25; the first run's 394 non-ProShares tickers were all 404s). Funds launched after 2025-12-31 are not in the mapping and
are therefore missing from 2026 (stated in the card).

WHEN A ROW IS KNOWN. The row dated d is the fund's NAV and shares after d's close; the issuer
posts it that evening or the next morning [U: the posting time is not stated on the file]. The
test uses the row dated t-1 for a decision at about 15:30 on t, so even a next-morning post
is in time.

Writes ~/quantt_research/levered_etf_close/proshares/<TICKER>.csv (raw, as served) and
proshares_index.json (which tickers were served). Plain HTTPS GETs; no key, no account.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

RES = Path.home() / "quantt_research"
OUT = RES / "levered_etf_close" / "proshares"
URL = "https://accounts.profunds.com/etfdata/ByFund/{t}-historical_nav.csv"
HEADER = "Date,ProShares Name,Ticker,NAV,Prior NAV,NAV Change (%),NAV Change ($),Shares Outstanding (000),Assets Under Management"
UA = "Mozilla/5.0 (research data client)"


def get(t: str) -> bytes | None:
    """The file's bytes, or None when the endpoint answers 404 (not a ProShares fund).
    Any other HTTP status raises; urllib's own network errors propagate."""
    req = urllib.request.Request(URL.format(t=t), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                raise RuntimeError(f"{t}: HTTP {r.status} from the ProShares endpoint")
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"{t}: HTTP {e.code} from the ProShares endpoint; not recorded as not-ProShares") from e


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(RES / "levered_etf_reversal" / "pairs.parquet")
    tickers = sorted(set(pairs["fund_ticker"].dropna()))
    index = {}
    for t in tickers:
        f = OUT / f"{t}.csv"
        if f.exists():
            index[t] = "proshares"
            continue
        body = get(t)
        time.sleep(0.5)
        if body is None:
            index[t] = "not served as the ProShares CSV (HTTP 404)"
            continue
        first = body.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
        if first != HEADER:
            raise RuntimeError(f"{t}: HTTP 200 but the first line is {first[:80]!r}, not the ProShares header")
        f.write_bytes(body)
        index[t] = "proshares"
    (OUT.parent / "proshares_index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    got = [t for t, v in index.items() if v == "proshares"]
    print(f"{len(got)} of {len(tickers)} mapped fund tickers served by ProShares: {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
