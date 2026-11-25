"""Downloaders and parsers for the MSCI Nov 2026 shadow (paper only; public files; no accounts).

Every web request goes through ../../12_studies/msci_pi/polite.py: neutral browser User-Agent, no identifiers,
>= 1.6 s between requests across processes. Raw files are written to raw/<fetch-date>/ and never modified.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.STUDY_PI))
import polite  # noqa: E402


def rawdir(fetch_date: str) -> Path:
    d = C.RAW / fetch_date
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------- iShares CH JSON
def ch_path(outdir: Path, fund: str, asof: str | None) -> Path:
    tag = f"asof{pd.Timestamp(asof):%Y%m%d}" if asof else "latest"
    return outdir / f"{fund}_{tag}.json"


def fetch_ch(fund: str, asof: str | None, outdir: Path, use_cache: bool = True) -> Path:
    """Holdings JSON of an iShares UCITS fund (CH site) as of a date (None = latest). Dated files are first looked
    up in the msci_pi study cache (same endpoint, fetched 2026-10-12) and copied, so no request is repeated."""
    p = ch_path(outdir, fund, asof)
    if p.exists():
        return p
    if asof and use_cache:
        q = C.ISHCH_CACHE / f"{fund}_{pd.Timestamp(asof):%Y%m%d}.json"
        if q.exists():
            shutil.copy2(q, p)
            return p
    url = C.CH_JSON.format(pid=C.CH_FUNDS[fund]["pid"]) + (f"&asOfDate={pd.Timestamp(asof):%Y%m%d}" if asof else "")
    st, b = polite.get(url)
    if st != 200 or not b.lstrip(b"\xef\xbb\xbf").startswith(b"{"):
        raise RuntimeError(f"{fund} {asof}: HTTP {st}, {len(b)} bytes")
    if not json.loads(b.decode("utf-8-sig")).get("aaData"):   # date not posted yet: don't cache the empty answer
        raise RuntimeError(f"{fund} {asof}: empty holdings (not posted yet?), {len(b)} bytes; re-run later")
    p.write_bytes(b)
    return p


def _raw(x):
    return x.get("raw") if isinstance(x, dict) else x


def parse_ch(p: Path, fund: str | None = None) -> pd.DataFrame:
    """Tidy equity lines: ticker, name, sector, mv (fund ccy), weight, shares, isin, price (local), location,
    exchange, mkt_ccy, fx (fund ccy per unit of the quoted price, implied = mv / (shares x price))."""
    j = json.loads(p.read_text(encoding="utf-8-sig"))
    rows = []
    for r in j.get("aaData", []):
        if len(r) < 13:
            continue
        o = len(r) - 13
        rows.append(dict(ticker=str(r[0]).strip(), name=r[1], sector=r[2], asset_class=r[3 + o],
                         mv=_raw(r[4 + o]), weight=_raw(r[5 + o]), shares=_raw(r[7 + o]), isin=r[8 + o],
                         price=_raw(r[9 + o]), location=r[10 + o], exchange=r[11 + o], mkt_ccy=r[12 + o]))
    df = pd.DataFrame(rows)
    for c in ("mv", "weight", "shares", "price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df.asset_class == "Equity") & (df.shares > 0) & (df.price > 0)].copy()
    df["weight"] = df.weight / 100.0
    df["fx"] = df.mv / (df.shares * df.price)
    df["fund"] = fund or p.stem.split("_")[0]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------------------- iShares US CSV
def fetch_us(fund: str, outdir: Path) -> tuple[Path | None, str]:
    pid, slug = C.US_FUNDS[fund]
    st, b = polite.get(C.US_CSV.format(pid=pid, slug=slug))
    txt = b[:3000].decode("utf-8-sig", errors="replace")
    if st != 200 or "Ticker," not in txt or "<html" in txt[:500].lower():
        return None, f"HTTP {st}, {'HTML' if b'<html' in b[:500].lower() else len(b)}"
    asof = None
    for line in txt.splitlines()[:12]:
        if line.startswith("Fund Holdings as of"):
            asof = pd.to_datetime(line.split(",", 1)[1].strip().strip('"'))
    p = outdir / f"{fund}_us_asof{asof:%Y%m%d}.csv"
    p.write_bytes(b)
    return p, f"ok, holdings as of {asof.date()}"


def parse_us(p: Path) -> pd.DataFrame:
    raw = p.read_bytes().decode("utf-8-sig", errors="replace").splitlines()
    i = next(k for k, l in enumerate(raw) if l.startswith("Ticker,"))
    body = [l for l in raw[i:] if l.count(",") >= 10]
    df = pd.read_csv(io.StringIO("\n".join(body)), thousands=",", dtype={"Ticker": str})
    df = df[df["Asset Class"] == "Equity"].copy()
    for c in ("Market Value", "Quantity", "Price", "Weight (%)"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------------------------------------------------------------------------------- MSCI public lists
def fetch_msci_list(kind: str, review: str, outdir: Path) -> tuple[Path | None, int]:
    """kind 'STD' or 'SC'; review 'YYYY-MM'. Returns (path or None, HTTP status)."""
    ts = pd.Timestamp(review + "-01")
    url = C.MSCI_LIST[kind].format(mon=ts.strftime("%b"), yy=ts.strftime("%y"))
    st, b = polite.get(url)
    if st != 200 or not b.startswith(b"%PDF"):
        return None, st
    p = outdir / Path(url).name
    p.write_bytes(b)
    return p, st
