#!/usr/bin/env python3
"""Save iBorrowDesk's free rolling one-year borrow history for our index-event names.

iBorrowDesk (https://www.iborrowdesk.com) republishes Interactive Brokers' public shortable-stock
feed. Its undocumented JSON API returns, per symbol, about 259 daily rows (one rolling year; the
oldest day drops off every day) of fee, rebate and available shares, with open/high/low variants,
plus ~1 week of 15-minute snapshots in `real_time`:
    GET https://www.iborrowdesk.com/api/ticker/{SYMBOL}      (tested 2026-10-09)
The `cusip` field in the response is in fact the IBKR contract id (conid).

Politeness: no account, no login, neutral User-Agent, no personal identifier in any header,
at most one request every 1.5 s (default 1.6 s), exponential backoff on 429/5xx/network errors
(honours Retry-After), and a circuit breaker that stops the run after repeated failures.

Steps (default: all three):
  build   -> universe.csv: the requested names by group, mapped to iBorrowDesk symbols.
             Group 1 tsx     : XIC holdings, S&P/TSX Composite review changes since Sep 2024,
                               12_studies events 2025-26, current shadow predictions.
             Group 2 russell : June 2026 recon from Datastream month-end lists (R2), R2000 adds,
                               deletes and R1000<->R2000 migrants, plus FTSE's final R3000
                               add/delete PDFs (13_shadow/russell_dec2026/raw/ftse_2026-06-26).
             Group 3 stoxx   : STOXX Europe 600 changes since 2025-09 on UK, Swedish and Italian
                               lines (R2 comp_global_daily.g_idxcst_his, gvkeyx 150376).
             R2 extracts are cached in inputs/ so later builds don't need R2 (--refresh-inputs
             re-queries). Symbol mapping uses the latest IBKR shortstock snapshot in
             ../ibkr_archive/ (see README "Symbol mapping").
  fetch   -> raw/{SYMBOL}.json (exact response body) and manifest.csv. Resumable: symbols that
             already have a raw file are skipped unless --refresh; on --refresh the old file is
             first moved to raw/history/{SYMBOL}.{fetch date}.json so no older day is lost.
  report  -> coverage.csv and a printed per-group summary.

Usage (run from <repo>/code so `import r2` resolves; R2 only needed
when inputs/ is missing or --refresh-inputs is given):
  PYTHONPATH=<repo>/code python3 \
      index_rebalance_research/14_borrow_data/iborrowdesk/pull_iborrowdesk.py [build|fetch|report|all]
      [--refresh] [--refresh-inputs] [--groups tsx,russell,stoxx] [--interval 1.6] [--limit N]
Exit code: 0 on success, 2 if the circuit breaker stopped the fetch.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]                       # index_rebalance_research/
RAW = HERE / "raw"
HIST = RAW / "history"
INPUTS = HERE / "inputs"
UNIVERSE = HERE / "universe.csv"
MANIFEST = HERE / "manifest.csv"
COVERAGE = HERE / "coverage.csv"
LOG = HERE / "fetch.log"

API = "https://www.iborrowdesk.com/api/{}"          # ticker/{SYMBOL}, search/{QUERY}
UA = "Mozilla/5.0 (compatible; borrow-history-research/1.0)"   # neutral; no contact details
MIN_INTERVAL = 3.0          # seconds between request starts (hard floor 1.5). 1.6 s drew HTTP 444 blocks
                            # after ~135 requests on 2026-10-09, so the default is slower.
MAX_TRIES = 5
BREAKER = 8                 # consecutive non-404 failures -> stop the run
BLOCK_CODES = (403, 444)    # server refusing us: no retries; pause, slow down, then stop if it persists
COOLDOWN = 900              # seconds to pause after a block (x number of blocks so far)
MAX_COOLDOWNS = 4           # per run; after that the run stops (resume later)

# ---- inputs (read-only, outside this folder)
XIC = RESEARCH / "13_shadow/tsx_dec2026/raw/xic/XIC_holdings_20261009.csv"
TSX_CHANGES = RESEARCH / "13_shadow/tsx_dec2026/review_changes.csv"
TSX_EVENTS = RESEARCH / "12_studies/tsx_composite/events.csv"
TSX_PRED = RESEARCH / "13_shadow/tsx_dec2026/runs/2026-12/2026-10-09_H-10w/predictions.csv"
FTSE_DIR = RESEARCH / "13_shadow/russell_dec2026/raw/ftse_2026-06-26"
IBKR_ARCHIVE = RESEARCH / "14_borrow_data/ibkr_archive"

# IBKR country file -> iBorrowDesk suffix. Verified 2026-10-09 by fetching and matching the
# returned conid: RY.CA, AP.UN.CA, BBD.B.CA | AAPL, BRK.B | BARC.LN, BA..LN | ERIC.B.LN,
# VOLV.B.LN (Swedish lines are filed under .LN) | ENI.AV, PRY.AV (Italian lines are under .AV;
# the old .IM pages, e.g. UCG.IM, stopped in 2017).
SUFFIX = {"canada": ".CA", "usa": "", "british": ".LN", "swedish": ".LN", "italy": ".AV"}
COUNTRY_CCY = {"canada": "CAD", "usa": "USD", "british": "GBP", "swedish": "SEK", "italy": "EUR"}
GROUP_ORDER = ["tsx", "russell", "stoxx"]

STOP = {"INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "CO", "COMPANY", "THE", "GROUP", "HOLDINGS",
        "HOLDING", "HLDGS", "SA", "SPA", "AB", "AG", "NV", "SE", "TRUST", "REIT", "CLASS", "CL", "A", "B", "SHS",
        "COM", "NEW", "INTL", "INTERNATIONAL", "OF", "AND", "UNITS", "UNIT", "TR", "LP", "PUBL", "SUB", "VTG",
        "SUBORDINATE", "VOTING", "ORD", "REG", "THERAPEUTICS", "INVT", "INVESTMENT", "REAL", "ESTATE", "FUND"}


def log(msg: str) -> None:
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def tokens(name: str | None) -> set[str]:
    if not name:
        return set()
    t = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper().replace("&", " ")).split()
    return {w for w in t if w not in STOP and len(w) > 1}


GENERIC = {"BANCA", "BANCO", "BANK", "DI", "DE", "DEL", "LA", "LE", "ET", "CIE", "PPTYS", "PROPERTIES", "RESOURCES",
           "MINING", "ENERGY", "COMMUNICATIONS", "FINANCIAL", "CAPITAL", "SYSTEMS", "TECHNOLOGIES", "PARTNERS"}
LEGAL = {"PLC", "SPA", "S", "P", "A", "AB", "PUBL", "LTD", "LIMITED", "CORP", "CO", "THE", "NV", "INC", "SA", "AG"}


def strict_agree(a: str | None, b: str | None) -> bool:
    """Stricter test for search hits: share a non-generic word, and the source name's first
    non-generic word appears in the hit."""
    ta, tb = tokens(a) - GENERIC, tokens(b) - GENERIC
    if not ta or not tb:
        return False
    first = next((w for w in re.sub(r"[^A-Z0-9 ]", " ", str(a).upper()).split() if w in ta), None)
    return bool(ta & tb) and (first in tb or any(first[:4] == w[:4] for w in tb))


def names_agree(a: str | None, b: str | None) -> bool | None:
    """Loose company-name check: share a non-generic word, or the names (significant words run
    together) start with the same 5 letters ("NUVISTA" / "NU VISTA", "INNOV.SLTN." / "INNOVATIVE")."""
    ta, tb = tokens(a) - GENERIC, tokens(b) - GENERIC
    if not ta or not tb:
        return None
    if ta & tb:
        return True
    run = lambda n, t: "".join(w for w in re.sub(r"[^A-Z0-9 ]", " ", str(n).upper()).split() if w in t)
    ca, cb = run(a, ta), run(b, tb)
    return len(ca) >= 5 and len(cb) >= 5 and ca[:5] == cb[:5]


# ------------------------------------------------------------------ IBKR snapshot lookup
def load_ibkr() -> tuple[dict, str]:
    """{country: [row dicts]} from the newest IBKR shortstock snapshot in ../ibkr_archive."""
    days = sorted(p for p in IBKR_ARCHIVE.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}$", p.name))
    if not days:
        raise SystemExit(f"no IBKR snapshot in {IBKR_ARCHIVE}; run ../ibkr_shortstock_archiver.py first")
    day = days[-1]
    out = {}
    for c in SUFFIX:
        files = sorted(day.glob(f"{c}_*.txt.gz"))
        if not files:
            raise SystemExit(f"missing {c} file in {day}")
        rows = []
        with gzip.open(files[-1], "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\r\n").split("|")
                if len(p) >= 9:
                    rows.append(dict(sym=p[0], cur=p[1], name=p[2], conid=p[3], isin=p[4], country=c))
        out[c] = rows
    return out, day.name


def ibd_symbol(ibkr_sym: str, country: str) -> str:
    return ibkr_sym.replace(" ", ".") + SUFFIX[country]


# ------------------------------------------------------------------ group 1: S&P/TSX Composite
def build_tsx(ib) -> list[dict]:
    by_sym = {r["sym"]: r for r in ib["canada"]}
    reqs = []   # (src_ticker, subgroup, name)

    text = XIC.read_text(encoding="utf-8-sig").splitlines()
    hdr = next(i for i, l in enumerate(text) if l.startswith("Ticker,"))
    for r in csv.DictReader(io.StringIO("\n".join(text[hdr:]))):
        tic = (r.get("Ticker") or "").strip()
        if r.get("Asset Class") != "Equity" or r.get("Exchange") != "Toronto Stock Exchange":
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9]*(\.[A-Z]{1,2})?", tic):     # drops residual lines like 2299955D
            continue
        reqs.append((tic, "xic_member_20261009", r.get("Name")))

    for r in csv.DictReader(TSX_CHANGES.open()):
        reqs.append((r["symbol"].strip(), f"review_{r['side']}_{r['review']}", r.get("name") or None))

    for r in csv.DictReader(TSX_EVENTS.open()):
        if r["eff_date"] >= "2025-01-01":
            tic = r["tic"].strip().rstrip(".")                          # Compustat writes FIL. for FIL
            reqs.append((tic, f"event_{r['side']}_{r['etype']}_{r['eff_date']}", r["conm"]))

    for r in csv.DictReader(TSX_PRED.open()):
        reqs.append((r["symbol"].strip(), f"shadow_{r['status']}_2026-12", r["issuer"]))

    out = []
    for tic, sub, name in reqs:
        ib_row = by_sym.get(tic)
        out.append(dict(group="tsx", subgroup=sub, src_ticker=tic, src_name=name or "", isin="",
                        ibkr_file="canada" if ib_row else "", ibkr_symbol=ib_row["sym"] if ib_row else "",
                        ibkr_conid=ib_row["conid"] if ib_row else "", ibkr_name=ib_row["name"] if ib_row else "",
                        ibd_symbol=tic + ".CA",
                        map_rule="tsx symbol = ibkr canada symbol; +'.CA'" if ib_row else
                        "not in today's IBKR canada file (delisted/renamed?); tried tsx symbol +'.CA'"))
    return out


# ------------------------------------------------------------------ group 2: Russell June 2026
def russell_inputs(refresh: bool) -> tuple[list[dict], dict]:
    path = INPUTS / "russell_jun2026_datastream.csv"
    meta_path = INPUTS / "russell_jun2026_datastream.meta.json"
    if path.exists() and not refresh:
        return list(csv.DictReader(path.open())), json.loads(meta_path.read_text())
    import pandas as pd
    import r2
    with r2.session() as con:
        L = r2.sql(con, """
            SELECT DISTINCT indexlistintcode::INT code, date_, infocode::BIGINT infocode
            FROM tr_ds_equities.wrds_ds_indexconstmerged_mth
            WHERE indexlistintcode IN (6967, 6968) AND date_ BETWEEN DATE '2026-05-01' AND DATE '2026-06-30'""")
        L["date_"] = pd.to_datetime(L.date_)
        pre = L.loc[L.date_.dt.month == 5, "date_"].max()
        post = L.loc[L.date_.dt.month == 6, "date_"].max()
        S = lambda c, d: set(L.loc[(L.code == c) & (L.date_ == d), "infocode"])
        r1a, r2a, r1b, r2b = S(6967, pre), S(6968, pre), S(6967, post), S(6968, post)
        cats = {"r2000_add": r2b - r2a - r1a, "r2000_delete": r2a - r2b - r1b,
                "migrant_r1000_to_r2000": r1a & r2b, "migrant_r2000_to_r1000": r2a & r1b}
        ic = sorted(set().union(*cats.values()))
        cols = "infocode::BIGINT infocode, ticker, dsqtname, isin, startdate, enddate, isprimqt"
        N = r2.sql(con, f"SELECT {cols} FROM tr_ds_equities.wrds_ds_names WHERE infocode IN ({','.join(map(str, ic))})")
        miss = sorted(set(ic) - set(N.infocode))
        if miss:
            N2 = r2.sql(con, f"SELECT {cols} FROM tr_ds_equities.wrds_ds_names_full "
                             f"WHERE infocode IN ({','.join(map(str, miss))})")
            N = pd.concat([N, N2], ignore_index=True)
    N["startdate"], N["enddate"] = pd.to_datetime(N.startdate), pd.to_datetime(N.enddate)
    N["cover"] = (N.startdate <= post) & (N.enddate >= post)
    N = N.sort_values(["infocode", "cover", "enddate"]).groupby("infocode").tail(1).set_index("infocode")
    rows = []
    for cat, s in cats.items():
        for i in sorted(s):
            n = N.loc[i] if i in N.index else None
            get = lambda col: ("" if n is None or pd.isna(n[col]) else str(n[col]))   # n.isin would be Series.isin
            rows.append(dict(category=cat, infocode=i, ticker=get("ticker"), name=get("dsqtname"), isin=get("isin")))
    INPUTS.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(source="tr_ds_equities.wrds_ds_indexconstmerged_mth 6967 (R1000) / 6968 (R2000)",
                pre=str(pre.date()), post=str(post.date()), counts={k: len(v) for k, v in cats.items()},
                queried=dt.datetime.now().isoformat(timespec="seconds"))
    meta_path.write_text(json.dumps(meta, indent=1))
    return list(csv.DictReader(path.open())), meta


def ftse_list(p: Path) -> list[dict]:
    if not p.exists() or not shutil.which("pdftotext"):
        return []
    txt = subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, check=True).stdout
    ind = ("Technology|Financials|Health Care|Energy|Industrials|Basic Materials|Consumer Discretionary|"
           "Consumer Staples|Telecommunications|Utilities|Real Estate")
    row = re.compile(r"^\s*(?P<company>\S.*?)\s{2,}(?P<symbol>[A-Z0-9][A-Z0-9.\-/]*)\s{2,}(?:" + ind + r")\s*$")
    return [m.groupdict() for m in map(row.match, txt.splitlines()) if m]


def build_russell(ib, refresh_inputs: bool) -> list[dict]:
    ds, meta = russell_inputs(refresh_inputs)
    usa = ib["usa"]
    by_sym = {}
    for r in usa:
        by_sym.setdefault(r["sym"], []).append(r)

    def find(ticker: str, isin: str, name: str):
        cands = []
        for t in {ticker, ticker.replace(".", " "), ticker.replace("/", " "), ticker.replace("-", " ")}:
            cands += by_sym.get(t, [])
        if isin:
            hit = [r for r in cands if r["isin"][-5:] == isin[-5:]]      # IBKR masks US ISINs to the last 5
            if hit:
                return hit[0], "ds ticker = ibkr usa symbol, ISIN tail agrees"
        hit = [r for r in cands if names_agree(r["name"], name)]
        if hit:
            return hit[0], "ds ticker = ibkr usa symbol, name agrees"
        if isin:
            hit = [r for r in usa if r["isin"][-5:] == isin[-5:] and names_agree(r["name"], name)]
            if len(hit) == 1:
                return hit[0], "ISIN tail + name match in ibkr usa (ticker differs)"
        return None, "not in today's IBKR usa file (delisted/renamed?); tried ds ticker as is"

    out = []
    for r in ds:
        sub = r["category"] + f"_{meta['pre']}_to_{meta['post']}"
        tic = (r["ticker"] or "").strip()
        if not tic:
            out.append(dict(group="russell", subgroup=sub, src_ticker="", src_name=r["name"], isin=r["isin"],
                            ibkr_file="", ibkr_symbol="", ibkr_conid="", ibkr_name="", ibd_symbol="",
                            map_rule="no Datastream ticker; not requested"))
            continue
        hit, rule = find(tic, r["isin"], r["name"])
        out.append(dict(group="russell", subgroup=sub, src_ticker=tic, src_name=r["name"], isin=r["isin"],
                        ibkr_file="usa" if hit else "", ibkr_symbol=hit["sym"] if hit else "",
                        ibkr_conid=hit["conid"] if hit else "", ibkr_name=hit["name"] if hit else "",
                        ibd_symbol=ibd_symbol(hit["sym"], "usa") if hit else tic.replace(" ", ".").replace("/", "."),
                        map_rule=rule))
    for side in ["additions", "deletions"]:
        for r in ftse_list(FTSE_DIR / f"ru3000-{side}-20260626.pdf"):
            hit, rule = find(r["symbol"], "", r["company"])
            sym = r["symbol"]
            out.append(dict(group="russell", subgroup=f"ftse_final_r3000_{side}_20260626", src_ticker=sym,
                            src_name=r["company"], isin="", ibkr_file="usa" if hit else "",
                            ibkr_symbol=hit["sym"] if hit else "", ibkr_conid=hit["conid"] if hit else "",
                            ibkr_name=hit["name"] if hit else "",
                            ibd_symbol=ibd_symbol(hit["sym"], "usa") if hit else sym.replace("/", ".").replace("-", "."),
                            map_rule=rule.replace("ds ticker", "ftse symbol")))
    return out


# ------------------------------------------------------------------ group 3: STOXX Europe 600
def stoxx_inputs(refresh: bool) -> list[dict]:
    path = INPUTS / "stoxx600_changes_since_2025-09.csv"
    if path.exists() and not refresh:
        return list(csv.DictReader(path.open()))
    import r2
    with r2.session() as con:
        d = r2.sql(con, """
            SELECT h.gvkey, h.iid, h."from" AS dt_from, h.thru AS dt_thru, s.isin, s.sedol, s.exchg, s.excntry,
                   c.conm, c.fic
            FROM comp_global_daily.g_idxcst_his h
            JOIN comp_global_daily.g_security s USING (gvkey, iid)
            LEFT JOIN comp_global_daily.g_company c USING (gvkey)
            WHERE h.gvkeyx = '150376' AND (h."from" >= DATE '2025-09-01' OR h.thru >= DATE '2025-09-01')
            ORDER BY h."from", h.gvkey""")
        last = r2.sql(con, """SELECT max("from") mf, max(thru) mt FROM comp_global_daily.g_idxcst_his
                              WHERE gvkeyx = '150376'""")
    d["data_max_from"], d["data_max_thru"] = str(last.mf[0]), str(last.mt[0])
    INPUTS.mkdir(exist_ok=True)
    d.to_csv(path, index=False)
    return list(csv.DictReader(path.open()))


def build_stoxx(ib, refresh_inputs: bool) -> list[dict]:
    cmap = {"GBR": "british", "SWE": "swedish", "ITA": "italy"}
    out = []
    for r in stoxx_inputs(refresh_inputs):
        c = cmap.get(r["excntry"])
        if not c:
            continue
        sides = []
        if r["dt_from"] >= "2025-09-01":
            sides.append(f"add_{r['dt_from']}")
        if r["dt_thru"] and r["dt_thru"] >= "2025-09-01":
            sides.append(f"delete_after_{r['dt_thru']}")
        hits = [x for x in ib[c] if x["isin"] == r["isin"]]
        hits.sort(key=lambda x: x["cur"] != COUNTRY_CCY[c])               # local-currency line first
        hit = hits[0] if hits else None
        for s in sides:
            out.append(dict(group="stoxx", subgroup=f"{c}_{s}", src_ticker=r["isin"], src_name=r["conm"],
                            isin=r["isin"], ibkr_file=c if hit else "", ibkr_symbol=hit["sym"] if hit else "",
                            ibkr_conid=hit["conid"] if hit else "", ibkr_name=hit["name"] if hit else "",
                            ibd_symbol=ibd_symbol(hit["sym"], c) if hit else "",
                            map_rule=f"ISIN -> ibkr {c} symbol; +'{SUFFIX[c]}'" if hit else
                            f"ISIN not in today's IBKR {c} file"))
    return out


def resolve_by_search(rows: list[dict], interval: float, ib: dict) -> None:
    """STOXX rows whose ISIN is not in today's IBKR file (delisted, or missing from IBKR's file):
    query iBorrowDesk's search endpoint by company name (cached in inputs/), keep results with the
    country's suffix whose name agrees. Fetch then re-checks the name on the ticker page."""
    cache_p = INPUTS / "ibd_search_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    isin_of = {ibd_symbol(x["sym"], c): x["isin"] for c in ib for x in ib[c]}   # iBorrowDesk symbol -> IBKR ISIN
    F = None
    for r in rows:
        if r["group"] != "stoxx" or r["ibd_symbol"]:
            continue
        c = r["subgroup"].split("_")[0]
        suf = SUFFIX[c]
        words = re.sub(r"[^A-Z0-9 ]", " ", r["src_name"].upper()).split()
        legal_free = " ".join(w for w in words if w not in LEGAL)
        toks = [w for w in words if w not in STOP and w not in GENERIC]
        picked, tried = None, []
        for q in dict.fromkeys([legal_free, " ".join(toks[:2]), toks[0] if toks else ""]):
            if not q:
                continue
            if q not in cache:
                F = F or Fetcher(interval)
                st, body, _, err = F.get(q, kind="search")
                cache[q] = json.loads(body).get("results", []) if st == 200 else []
                cache_p.write_text(json.dumps(cache, indent=1))
            tried.append(q)
            ok = [x for x in cache[q] if x["symbol"].endswith(suf) and strict_agree(r["src_name"], x["name"])
                  and (not r["isin"] or isin_of.get(x["symbol"], r["isin"]) == r["isin"])]          # symbol held by another ISIN -> reject
            ok.sort(key=lambda x: ".OLD." in x["symbol"])
            if ok:
                picked = ok[0]
                break
        if picked:
            r.update(ibd_symbol=picked["symbol"],
                     map_rule=f"ISIN not in today's IBKR {c} file; iBorrowDesk search '{tried[-1]}' -> "
                              f"{picked['symbol']} ({picked['name']}); name check only")
        else:
            r["map_rule"] += f"; search {tried} found no {suf} line"


def build(args) -> None:
    ib, day = load_ibkr()
    rows = []
    if "tsx" in args.groups:
        rows += build_tsx(ib)
    if "russell" in args.groups:
        rows += build_russell(ib, args.refresh_inputs)
    if "stoxx" in args.groups:
        rows += build_stoxx(ib, args.refresh_inputs)
        resolve_by_search(rows, args.interval, ib)
    if UNIVERSE.exists():                                   # keep groups not rebuilt this time
        rows += [r for r in csv.DictReader(UNIVERSE.open()) if r["group"] not in args.groups]
    seen, uniq = set(), []
    for r in rows:
        k = (r["group"], r["subgroup"], r["src_ticker"], r["ibd_symbol"])
        if k not in seen:
            seen.add(k)
            r.setdefault("ibkr_snapshot", day)
            uniq.append(r)
    uniq.sort(key=lambda r: GROUP_ORDER.index(r["group"]))
    with UNIVERSE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0]))
        w.writeheader()
        w.writerows(uniq)
    for g in GROUP_ORDER:
        gr = [r for r in uniq if r["group"] == g]
        syms = {r["ibd_symbol"] for r in gr if r["ibd_symbol"]}
        log(f"build {g}: {len(gr)} rows, {len(syms)} unique iBorrowDesk symbols "
            f"({sum(1 for r in gr if r['ibkr_conid'])} rows matched to today's IBKR file)")


# ------------------------------------------------------------------ fetch
class Fetcher:
    def __init__(self, interval: float):
        self.interval = max(1.5, interval)
        self.next_at = 0.0

    def get(self, sym: str, kind: str = "ticker") -> tuple[int, bytes, int, str]:
        url = API.format(kind + "/" + urllib.request.quote(sym, safe=".-_"))
        err = ""
        for attempt in range(1, MAX_TRIES + 1):
            wait = self.next_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.next_at = time.monotonic() + self.interval
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            retry_after = 0.0
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, r.read(), attempt, ""
            except urllib.error.HTTPError as e:
                body = e.read()
                if e.code == 404:
                    return 404, body, attempt, ""
                if e.code in BLOCK_CODES:
                    return e.code, body, attempt, f"HTTP {e.code} (blocked)"
                err = f"HTTP {e.code}"
                ra = e.headers.get("Retry-After") if e.headers else None
                retry_after = float(ra) if ra and ra.strip().isdigit() else 0.0
                if e.code == 429:
                    self.interval = min(self.interval * 1.5, 10.0)       # slow down for the rest of the run
                if e.code < 500 and e.code != 429:
                    return e.code, body, attempt, err
            except Exception as e:                                       # timeouts, resets, DNS
                err = f"{type(e).__name__}: {e}"[:200]
            if attempt < MAX_TRIES:
                back = max(retry_after, 10 * 2 ** (attempt - 1))         # 10, 20, 40, 80 s
                log(f"  {sym}: {err}; backoff {back:.0f}s (attempt {attempt}/{MAX_TRIES})")
                time.sleep(min(back, 300))
        return -1, b"", MAX_TRIES, err


def raw_path(sym: str) -> Path:
    return RAW / (sym.replace("/", "_") + ".json")


def summarize(sym: str, status: int, body: bytes | None, fetched_at: str, attempts, err: str, exp: dict) -> dict:
    m = dict(symbol=sym, group=exp["group"], http_status=status, n_daily=0, first_date="", last_date="",
             fetched_at=fetched_at, groups=exp["groups"], subgroups=exp["subgroups"],
             expected_name=exp["name"], returned_name="", returned_symbol="", country="",
             conid_expected=exp["conid"], conid_returned="", match="", available_stale="", updated="",
             n_real_time=0, attempts=attempts, error=err)
    if status == 200 and body:
        try:
            d = json.loads(body)
        except ValueError:
            m["error"] = "invalid JSON"
            return m
        daily = d.get("daily") or []
        dates = sorted(x.get("date") for x in daily if x.get("date"))
        m.update(n_daily=len(daily), first_date=dates[0] if dates else "", last_date=dates[-1] if dates else "",
                 returned_name=d.get("name") or "", returned_symbol=d.get("symbol") or "",
                 country=d.get("country") or "", conid_returned=d.get("cusip") or "",
                 available_stale=d.get("available_stale"), updated=d.get("updated") or "",
                 n_real_time=len(d.get("real_time") or []))
        if exp["conid"]:
            m["match"] = "conid" if str(d.get("cusip")) == str(exp["conid"]) else "CONID_MISMATCH"
        else:
            ok = names_agree(exp["name"], d.get("name"))
            m["match"] = "name" if ok else ("NAME_MISMATCH" if ok is False else "unverified")
    return m


def expected_by_symbol() -> dict:
    rows = list(csv.DictReader(UNIVERSE.open()))
    exp = {}
    for r in rows:
        s = r["ibd_symbol"]
        if not s:
            continue
        e = exp.setdefault(s, dict(groups=[], subgroups=[], name="", conid=""))
        if r["group"] not in e["groups"]:
            e["groups"].append(r["group"])
        e["subgroups"].append(f"{r['group']}:{r['subgroup']}")
        e["name"] = e["name"] or r["ibkr_name"] or r["src_name"]
        e["conid"] = e["conid"] or r["ibkr_conid"]
    for s, e in exp.items():
        e["groups"].sort(key=GROUP_ORDER.index)
        e["group"] = e["groups"][0]
        e["groups"] = ";".join(e["groups"])
        e["subgroups"] = ";".join(dict.fromkeys(e["subgroups"]))
    return dict(sorted(exp.items(), key=lambda kv: (GROUP_ORDER.index(kv[1]["group"]), kv[0])))


def write_manifest(rows: dict) -> None:
    if not rows:
        return
    tmp = MANIFEST.with_suffix(".tmp")
    fields = list(next(iter(rows.values())).keys())
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(rows, key=lambda s: (GROUP_ORDER.index(rows[s]["group"]), s)):
            w.writerow(rows[s])
    tmp.replace(MANIFEST)


def fetch(args) -> int:
    RAW.mkdir(exist_ok=True)
    exp = expected_by_symbol()
    exp = {s: e for s, e in exp.items() if e["group"] in args.groups or
           any(g in args.groups for g in e["groups"].split(";"))}
    old = {r["symbol"]: r for r in csv.DictReader(MANIFEST.open())} if MANIFEST.exists() else {}
    man = dict(old)
    todo = [s for s in exp if args.refresh or not (raw_path(s).exists() or old.get(s, {}).get("http_status") == "404")]
    if args.limit:
        todo = todo[: args.limit]
    log(f"fetch: {len(exp)} symbols requested, {len(todo)} to fetch, interval {max(1.5, args.interval)}s "
        f"(~{len(todo) * max(1.5, args.interval) / 60:.0f} min)")
    if args.start_delay:
        log(f"start delay {args.start_delay}s (cooling down after an earlier block)")
        time.sleep(args.start_delay)
    F = Fetcher(args.interval)
    fails = cooldowns = 0
    k = 0
    while k < len(todo):
        s = todo[k]
        status, body, attempts, err = F.get(s)
        if status in BLOCK_CODES:
            cooldowns += 1
            write_manifest(man)
            if cooldowns > MAX_COOLDOWNS:
                log(f"[{k + 1}/{len(todo)}] {s} {err}; {MAX_COOLDOWNS} cooldowns used, stopping. Re-run later to resume.")
                return 2
            F.interval = min(F.interval * 1.5, 15.0)
            log(f"[{k + 1}/{len(todo)}] {s} {err}; pausing {COOLDOWN * cooldowns}s, then interval {F.interval:.1f}s "
                f"(cooldown {cooldowns}/{MAX_COOLDOWNS})")
            time.sleep(COOLDOWN * cooldowns)
            continue                                                    # retry the same symbol
        k += 1
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        if status == 200:
            p = raw_path(s)
            if p.exists():                                              # keep older windows
                HIST.mkdir(exist_ok=True)
                prev = (old.get(s, {}).get("fetched_at") or dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat())[:10]
                shutil.move(str(p), str(HIST / f"{p.stem}.{prev.replace('-', '')}.json"))
            tmp = p.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(p)
        m = summarize(s, status, body if status == 200 else None, now, attempts, err, exp[s])
        man[s] = m
        fails = 0 if status in (200, 404) else fails + 1
        if k % 25 == 0 or status not in (200, 404) or m["match"] in ("CONID_MISMATCH", "NAME_MISMATCH"):
            log(f"[{k}/{len(todo)}] {s} {status} rows={m['n_daily']} {m['first_date']}..{m['last_date']} {m['match']} {err}")
        if k % 25 == 0:
            write_manifest(man)
        if fails >= BREAKER:
            log(f"circuit breaker: {fails} consecutive failures; stopping. Re-run later to resume.")
            write_manifest(man)
            return 2
    # refresh group labels for rows not re-fetched (universe may have changed)
    for s, e in exp.items():
        if s in man:
            man[s].update(group=e["group"], groups=e["groups"], subgroups=e["subgroups"])
    write_manifest(man)
    log(f"fetch done: {len(todo)} requests")
    return 0


# ------------------------------------------------------------------ report
def category(r: dict) -> str:
    """Subgroup without its dates: review_add_2025-12 -> review_add, r2000_add_2026-05-29_to_... -> r2000_add."""
    return re.split(r"_(?:after_)?\d{4}", r["subgroup"])[0]


def stats(rows: list[dict], man: dict) -> dict:
    names = {(r["src_ticker"] or r["src_name"]) for r in rows}
    syms = sorted({r["ibd_symbol"] for r in rows if r["ibd_symbol"]})
    got = [man[s] for s in syms if man.get(s, {}).get("http_status") == "200"]
    found = [m for m in got if int(m["n_daily"]) > 0 and "MISMATCH" not in m["match"]]
    days = [int(m["n_daily"]) for m in found]
    return dict(source_names=len(names),
                unmapped_names=len({(r["src_ticker"] or r["src_name"]) for r in rows if not r["ibd_symbol"]}),
                symbols_requested=len(syms), http_200=len(got),
                http_404=sum(1 for s in syms if man.get(s, {}).get("http_status") == "404"),
                other_errors=sum(1 for s in syms if man.get(s, {}).get("http_status") not in (None, "200", "404")),
                found_with_rows=len(found),
                verified_conid=sum(1 for m in found if m["match"] == "conid"),
                verified_name=sum(1 for m in found if m["match"] == "name"),
                unverified=sum(1 for m in found if m["match"] == "unverified"),
                mismatched=sum(1 for m in got if "MISMATCH" in m["match"]),
                zero_rows_in_window=sum(1 for m in got if int(m["n_daily"]) == 0),
                full_window_ge250=sum(1 for d in days if d >= 250),
                median_days=statistics.median(days) if days else 0,
                fresh_last_date_ge_2026_10_01=sum(1 for m in found if m["last_date"] >= "2026-10-01"))


def report(args) -> None:
    uni = list(csv.DictReader(UNIVERSE.open()))
    man = {r["symbol"]: r for r in csv.DictReader(MANIFEST.open())} if MANIFEST.exists() else {}
    out = []
    for g in GROUP_ORDER:
        gr = [r for r in uni if r["group"] == g]
        if not gr:
            continue
        out.append(dict(group=g, category="ALL", **stats(gr, man)))
        for c in dict.fromkeys(category(r) for r in gr):
            out.append(dict(group=g, category=c, **stats([r for r in gr if category(r) == c], man)))
    with COVERAGE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        log(f"coverage {r['group']:8s} {r['category']:28s} requested={r['symbols_requested']:4d} "
            f"found={r['found_with_rows']:4d} (conid {r['verified_conid']}, name {r['verified_name']}, "
            f"unverified {r['unverified']}) 404={r['http_404']} mismatch={r['mismatched']} "
            f"zero_rows={r['zero_rows_in_window']} unmapped={r['unmapped_names']} median_days={r['median_days']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", default="all", choices=["build", "fetch", "report", "all"])
    ap.add_argument("--groups", default="tsx,russell,stoxx")
    ap.add_argument("--refresh", action="store_true", help="re-fetch symbols that already have a raw file")
    ap.add_argument("--refresh-inputs", action="store_true", help="re-query R2 for the Russell/STOXX inputs")
    ap.add_argument("--interval", type=float, default=MIN_INTERVAL, help="seconds between requests (floor 1.5)")
    ap.add_argument("--limit", type=int, default=0, help="fetch at most N symbols (testing)")
    ap.add_argument("--start-delay", type=int, default=0, help="seconds to wait before the first request")
    args = ap.parse_args()
    args.groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    rc = 0
    if args.step in ("build", "all"):
        build(args)
    if args.step in ("fetch", "all"):
        rc = fetch(args)
    if args.step in ("report", "all"):
        report(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
