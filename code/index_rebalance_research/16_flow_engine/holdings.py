"""iShares holdings: download (cached), parse to one tidy frame, and turn two dates into index-share changes.

Funds (common.ETF): XIC, XIU (blackrock.ca, CSV for shares + JSON for ISIN) and EXSA, ISF, MIDD, EXS1, EXS3, EXS2
(ishares.com/ch JSON, which carries ISIN, shares and price). All accept asOfDate=YYYYMMDD; the date requested is the
holdings date (checked: XIC "as of" header = request date; EU JSON prices change with the date).

Tidy columns: etf, asof, ticker, name, isin, shares, price, mv (fund currency), weight (fraction), location, exchange,
mkt_ccy, country (ISO2 of the listing exchange).

Holdings -> index shares. A fully replicating tracker holds H_i = k * F_i, where F_i are index (float-adjusted,
capped) shares and k = fund AUM / index float cap (flow_sizing.md §3.1). Between two dates
    F_i,1 / F_i,0 = (H_i,1 / H_i,0) / (k_1 / k_0),   k_1/k_0 = median over continuing names of H_i,1/H_i,0
(creations/redemptions scale every line; most names have no index change). With k_0 from the index MV,
F_i,0 = H_i,0 / k_0 and dF_i = F_i,1 - F_i,0. New names: F_i,1 = H_i,1/k_1; dropped names: F_i,0 = H_i,0/k_0.
"""
from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from common import ETF, ETF_URL, HOLD, EXCH_COUNTRY, http_get


def _raw(x):
    if isinstance(x, dict):
        return x.get("raw")
    return x


def _num(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace(",", "").replace("’", "").replace("'", "").replace("’", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def path_for(etf: str, asof: str | None, kind: str) -> Path:
    d = HOLD / etf
    d.mkdir(parents=True, exist_ok=True)
    tag = f"asof{pd.Timestamp(asof):%Y%m%d}" if asof else f"live{pd.Timestamp.today():%Y%m%d}"
    return d / f"{etf}_{tag}.{kind}"


def fetch(etf: str, asof: str | None = None, offline: bool = False) -> dict:
    """Download (or read cached) raw files for one fund and date. asof=None = latest file, stamped today."""
    meta = ETF[etf]
    out = {}
    kinds = ["json", "csv"] if meta["site"] == "ca" else ["json"]
    for kind in kinds:
        p = path_for(etf, asof, kind)
        if not p.exists():
            if offline:
                raise FileNotFoundError(p)
            url = ETF_URL[f"{meta['site']}_{kind}"].format(pid=meta["pid"], etf=etf)
            if asof:
                url += f"&asOfDate={pd.Timestamp(asof):%Y%m%d}"
            b = http_get(url)
            if kind == "json" and not b.lstrip(b"\xef\xbb\xbf").startswith(b"{"):
                raise RuntimeError(f"{etf} {asof}: JSON endpoint returned non-JSON")
            p.write_bytes(b)
        out[kind] = p
    return out


def parse(etf: str, asof: str | None = None, offline: bool = True) -> pd.DataFrame:
    files = fetch(etf, asof, offline=offline)
    meta = ETF[etf]
    j = json.loads(files["json"].read_text(encoding="utf-8-sig"))
    rows = j.get("aaData", [])
    if meta["site"] == "ch":
        recs = []
        for r in rows:
            if len(r) < 13:
                continue
            o = len(r) - 13            # UK-domiciled funds (ISF) carry an extra "Type" field after the name
            recs.append(dict(ticker=str(r[0]).strip(), name=r[1], asset_class=r[3 + o], mv=_num(_raw(r[4 + o])),
                             weight=_num(_raw(r[5 + o])) / 100.0, shares=_num(_raw(r[7 + o])), isin=r[8 + o],
                             price=_num(_raw(r[9 + o])), location=r[10 + o], exchange=r[11 + o], mkt_ccy=r[12 + o]))
        df = pd.DataFrame(recs)
        df = df[(df.asset_class == "Equity") & (df.shares > 0)].copy()
        asof_hdr = pd.Timestamp(asof) if asof else pd.Timestamp.today().normalize()
    else:
        js = pd.DataFrame([dict(ticker=str(r[0]).strip(), isin=r[5], location=r[6], exchange=r[7], mkt_ccy=r[8])
                           for r in rows if len(r) >= 9])
        txt = files["csv"].read_text(encoding="utf-8-sig")
        first = txt.splitlines()[0]
        asof_hdr = pd.Timestamp(re.search(r'"(.+)"', first).group(1))
        x = pd.read_csv(StringIO(txt), skiprows=2, thousands=",", keep_default_na=False, na_values=[""])
        x = x[(x["Asset Class"] == "Equity") & (pd.to_numeric(x["Price"], errors="coerce") > 0)].copy()
        df = pd.DataFrame(dict(ticker=x["Ticker"].astype(str).str.strip(), name=x["Name"],
                               mv=pd.to_numeric(x["Market Value"], errors="coerce"),
                               weight=pd.to_numeric(x["Weight (%)"], errors="coerce") / 100.0,
                               shares=pd.to_numeric(x["Shares"], errors="coerce"),
                               price=pd.to_numeric(x["Price"], errors="coerce")))
        df = df.merge(js.drop_duplicates("ticker"), on="ticker", how="left")
        df["location"] = df.location.fillna("Canada")
        df["exchange"] = df.exchange.fillna("Toronto Stock Exchange")
        df["mkt_ccy"] = df.mkt_ccy.fillna("CAD")
    df["etf"] = etf
    df["asof"] = asof_hdr
    df["country"] = df.exchange.map(EXCH_COUNTRY)
    miss = df.country.isna()
    if miss.any():   # fall back to ISIN prefix (listing country usually = ISIN country for these funds)
        df.loc[miss, "country"] = df.loc[miss, "isin"].astype(str).str[:2].replace({"JE": "GB", "GG": "GB", "IM": "GB"})
    # several funds hold a name through two lines (e.g. share classes); keep lines separately keyed by ISIN
    df = df.groupby(["etf", "asof", "isin"], as_index=False).agg(
        ticker=("ticker", "first"), name=("name", "first"), shares=("shares", "sum"), price=("price", "first"),
        mv=("mv", "sum"), weight=("weight", "sum"), location=("location", "first"), exchange=("exchange", "first"),
        mkt_ccy=("mkt_ccy", "first"), country=("country", "first"))
    return df


def index_share_changes(etf: str, d0: str, d1: str, index_mv: float, offline: bool = True,
                        min_rel: float = 0.0) -> pd.DataFrame:
    """Per-name change in index shares between holdings dates d0 and d1 (see module docstring).

    index_mv: index float cap at d0 in the fund's currency (units, not millions).
    Returns columns: isin, ticker, name, country, mkt_ccy, price0, price1, H0, H1, F0, F1, dF, rel, change_type, k0, k1.
    change_type: add / delete / share_update (|rel| >= min_rel) / none.
    """
    a = parse(etf, d0, offline=offline)
    b = parse(etf, d1, offline=offline)
    aum0 = a.mv.sum()
    k0 = aum0 / index_mv
    m = a.merge(b, on="isin", how="outer", suffixes=("0", "1"))
    both = m.shares0.notna() & m.shares1.notna()
    ratio = (m.shares1 / m.shares0)[both]
    g = float(np.median(ratio))
    k1 = k0 * g
    out = pd.DataFrame(dict(
        isin=m["isin"], ticker=m.ticker1.fillna(m.ticker0), name=m.name1.fillna(m.name0),
        country=m.country1.fillna(m.country0), mkt_ccy=m.mkt_ccy1.fillna(m.mkt_ccy0),
        price0=m.price0, price1=m.price1, H0=m.shares0.fillna(0.0), H1=m.shares1.fillna(0.0)))
    out["F0"] = out.H0 / k0
    out["F1"] = out.H1 / k1
    out["dF"] = out.F1 - out.F0
    out["rel"] = np.where(out.F0 > 0, out.F1 / np.where(out.F0 > 0, out.F0, 1) - 1, np.inf)
    out["change_type"] = np.select([out.H0 == 0, out.H1 == 0, out.rel.abs() >= min_rel],
                                   ["add", "delete", "share_update"], "none")
    out.loc[(out.change_type == "share_update") & (out.rel.abs() < 1e-9), "change_type"] = "none"
    out["k0"], out["k1"], out["flow_ratio"] = k0, k1, g
    return out


def weights_now(etf: str, asof: str | None = None, offline: bool = False) -> pd.DataFrame:
    """Current holdings (latest file if asof is None) with weights, for capping and as constituent lists."""
    fetch(etf, asof, offline=offline)
    return parse(etf, asof, offline=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download iShares holdings into data/holdings/ (polite, cached).")
    ap.add_argument("etfs", nargs="+")
    ap.add_argument("--asof", nargs="*", default=[None])
    a = ap.parse_args()
    for e in a.etfs:
        for d in a.asof:
            h = parse(e, d, offline=False)
            print(e, d, len(h), "lines", f"AUM {h.mv.sum() / 1e9:,.2f}bn", "max weight", round(h.weight.max(), 4))
