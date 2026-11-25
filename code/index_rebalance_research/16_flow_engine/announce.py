"""Turn an announced (or hand-entered) list of index changes into engine input rows.

    python3 announce.py --close 2026-12-18 --index STOXX600 --file lists/stoxx_dec2026.csv --source "STOXX release 1 Dec"
    python3 announce.py --close 2026-12-18 --index FTSE100 --add GB00B17BBQ50 --delete GB00BGDT3G23 --source "..."
    python3 announce.py --close 2026-12-18 --index SPTSX_COMP --sp-release path/or/url.html     (S&P DJI release)

List file columns (CSV, header required; see README "Announcement formats"):
  side         add | delete | migration_in | migration_out | share_update | iwf_update | capping
  isin         preferred key (ticker + country accepted)
  ticker, country, name
  index_shares optional F for adds/deletes (provider pro-forma); or old_index_shares,new_index_shares for updates
  price, prob  optional
Missing index shares are filled as follows (and said so in `notes`):
  delete / migration_out   F from today's iShares holdings of that index (H / k)
  add / migration_in       F of the same company in a sibling index fund of the same provider (FTSE 100<->250,
                           DAX<->MDAX<->TecDAX, S&P/TSX 60<->Composite); else the TSX shadow float (S&P/TSX) or
                           yfinance floatShares (flagged: not the provider's float)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import inputs as I
import live as LV
from common import INDEX

SIBLINGS = {"FTSE100": ["FTSE250"], "FTSE250": ["FTSE100"], "DAX": ["MDAX", "TECDAX"], "MDAX": ["DAX", "TECDAX"],
            "TECDAX": ["DAX", "MDAX"], "SDAX": ["MDAX", "TECDAX"], "SPTSX_60": ["SPTSX_COMP"],
            "SPTSX_COMP": ["SPTSX_60"], "STOXX600": []}


def _members(index):
    if not INDEX.get(index, {}).get("etf"):
        return pd.DataFrame(columns=["isin", "ticker", "F", "price", "name", "country", "mkt_ccy"])
    h, _, _ = LV.members_today(index)
    return h


def resolve(lst: pd.DataFrame, index: str, status: str, source: str) -> pd.DataFrame:
    mem = _members(index)
    sib = {s: _members(s) for s in SIBLINGS.get(index, []) if INDEX.get(s, {}).get("etf")}
    rows = []
    ts = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    for r in lst.itertuples(index=False):
        d = r._asdict()
        side = d["side"].strip().lower()
        isin, tic, cty = d.get("isin"), d.get("ticker"), d.get("country")
        price = d.get("price", np.nan)
        F_old, F_new, note = d.get("old_index_shares", np.nan), d.get("new_index_shares", np.nan), ""
        F = d.get("index_shares", np.nan)

        def find(df):
            if isinstance(isin, str) and len(df) and (df["isin"] == isin).any():
                return df[df["isin"] == isin].iloc[0]
            if isinstance(tic, str) and len(df) and (df.ticker == tic).any():
                return df[df.ticker == tic].iloc[0]
            return None
        m = find(mem)
        rec = m
        if side in ("delete", "migration_out"):
            if not np.isfinite(F if F == F else np.nan) and m is not None:
                F, note = float(m.F), "F from today's holdings of the index fund"
            F_old, F_new = F, 0.0
        elif side in ("add", "migration_in"):
            if not (isinstance(F, float) and np.isfinite(F)):
                for s, df in sib.items():
                    x = find(df)
                    if x is not None:
                        F, note = float(x.F), f"F from {s} fund holdings (same provider float)"
                        rec = x
                        break
            if not (isinstance(F, float) and np.isfinite(F)):
                if not (isinstance(tic, str) and isinstance(cty, str)):
                    raise SystemExit(f"{isin}: not held by any fund of this provider - give ticker and country (and "
                                     "index_shares from the provider file if you have it)")
                F, note = LV.estimate_float(str(tic), str(cty))
                note = f"F estimated: {note}"
                if cty == "CA":
                    info = LV.tsx_screen_info(str(tic))
                    isin = isin if isinstance(isin, str) else info.get("isin")
                    price = price if isinstance(price, float) and np.isfinite(price) else info.get("price", np.nan)
                    d["name"] = d.get("name") if isinstance(d.get("name"), str) else info.get("name")
                    d["currency"] = d.get("currency") if isinstance(d.get("currency"), str) else "CAD"
            F_old, F_new = 0.0, F
        if rec is not None:          # fill identifiers and price from the holdings record
            isin = isin if isinstance(isin, str) else rec["isin"]
            tic = tic if isinstance(tic, str) else rec.ticker
            cty = cty if isinstance(cty, str) else rec.country
            price = price if isinstance(price, float) and np.isfinite(price) else rec.price
            if not isinstance(d.get("name"), str):
                d["name"] = rec["name"]
            if not isinstance(d.get("currency"), str):
                d["currency"] = rec.mkt_ccy
        rows.append(dict(change_type=side, isin=isin, ticker=tic, country=cty, name=d.get("name"),
                         currency=d.get("currency"),
                         old_index_shares=F_old, new_index_shares=F_new, price=price,
                         prob=d.get("prob", 1.0) if d.get("prob", 1.0) == d.get("prob", 1.0) else 1.0,
                         status=status, source=source, source_ts=ts, notes=note))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", required=True)
    ap.add_argument("--index", required=True, choices=list(INDEX))
    ap.add_argument("--file")
    ap.add_argument("--add", default="")
    ap.add_argument("--delete", default="")
    ap.add_argument("--country", default="")
    ap.add_argument("--sp-release", help="S&P DJI release HTML (path or URL), S&P/TSX Composite tables")
    ap.add_argument("--status", default="announced", choices=["announced", "final", "predicted"])
    ap.add_argument("--source", default="")
    a = ap.parse_args()
    if a.file:
        lst = pd.read_csv(a.file)
    elif a.sp_release:
        import sys
        from common import SHADOW_TSX, http_get
        sys.dont_write_bytecode = True    # read-only import: leave no __pycache__ in the shadow folder
        sys.path.insert(0, str(SHADOW_TSX))
        from score import parse_release            # read-only import of the shadow's parser
        txt = http_get(a.sp_release).decode("utf-8", "ignore") if a.sp_release.startswith("http") \
            else open(a.sp_release, encoding="utf-8", errors="ignore").read()
        adds, dels = parse_release(txt)
        lst = pd.DataFrame([dict(side="add", ticker=t, country="CA") for t in adds] +
                           [dict(side="delete", ticker=t, country="CA") for t in dels])
        a.source = a.source or f"S&P DJI release {a.sp_release}"
    else:
        def ids(s):
            return [x.strip() for x in s.split(",") if x.strip()]
        lst = pd.DataFrame([dict(side="add", isin=x if len(x) == 12 else None, ticker=None if len(x) == 12 else x,
                                 country=a.country or None) for x in ids(a.add)] +
                           [dict(side="delete", isin=x if len(x) == 12 else None, ticker=None if len(x) == 12 else x,
                                 country=a.country or None) for x in ids(a.delete)])
    for c in ["isin", "ticker", "country", "name", "currency", "index_shares", "old_index_shares",
              "new_index_shares", "price", "prob"]:
        if c not in lst:
            lst[c] = np.nan
    rows = resolve(lst, a.index, a.status, a.source)
    p = I.write(rows, a.close, a.index, mode="append")
    print(p)
    print(rows[["change_type", "isin", "ticker", "old_index_shares", "new_index_shares", "price", "notes"]].to_string())


if __name__ == "__main__":
    main()
