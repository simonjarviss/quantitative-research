"""Downloaders for the European December 2026 shadow (paper only; public data; no accounts, no identifiers).

Every web request goes through `http_get` or `yf_call`: neutral browser User-Agent, at least config.MIN_GAP_S (1.6 s)
between requests (process-wide), retries with back-off. Every raw response is saved under raw/<fetch date>/.

Sources
  STOXX public selection lists  stoxx.com > Resources > Reports > Selection Lists. Files with prefix "slpublic" are
                                unrestricted (page text). STOXX Europe 600 = slpublic_sxxp_YYYYMMDD.csv (first trading
                                day of the month; Rank (FINAL) at the previous month-end cut-off, membership, RIC).
                                DAX family = slpublic_{daxk,mkdx,sdxk,tdxk}_YYYYMMDD.csv (3rd trading day 22:00 CET;
                                one ranking of all FSE Regulated Market companies with ISIN, membership, comments;
                                tdxk = Technology-industry ranking for TecDAX).
  iShares UCITS holdings        16_flow_engine/holdings.py (parser reused; cache redirected to raw/<date>/holdings/)
  FTSE Russell notices JSON     research.ftserussell.com ... GetNotices?id=UK (family UK = FTSE 100/250)
  STOXX RSS                     stoxx.com/feed/ (last ~10 press releases; DAX/STOXX review results)
  Yahoo Finance (yfinance)      daily prices (chart endpoint, one request per symbol), info (shares, float), search
  R2 (read-only)                FTSE All-Share / 100 / 250 Datastream daily lists to 2026-06-23 with ISIN and
                                Datastream shares (numshrs) for the UK universe and the June 2026 calibration
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

_last = [0.0]


def _wait():
    w = C.MIN_GAP_S - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()


def http_get(url: str, params: dict | None = None, tries: int = 3, timeout: int = 60) -> bytes:
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    err = None
    for k in range(tries):
        _wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": C.UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(4 * (k + 1))
    raise RuntimeError(f"download failed: {url[:160]}: {err}")


def rawdir(fetch_date: str) -> Path:
    d = C.RAW / fetch_date
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------- selection lists
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def sl_index(kind: str, d0: pd.Timestamp, d1: pd.Timestamp, out: Path) -> list[dict]:
    """Listing of selection-list files between months d0..d1 (<= 12 months). kind = 'STOXX' | 'DAX'."""
    cmd = "getStoxxSLData" if kind == "STOXX" else "getDaxSLData"
    p = {C.SL_PARAM + "cmd": cmd, C.SL_PARAM + "fromdate": f"{MONTHS[d0.month - 1]} {d0.year}",
         C.SL_PARAM + "todate": f"{MONTHS[d1.month - 1]} {d1.year}", C.SL_PARAM + "symbols": ""}
    b = http_get(C.SL_PAGE, p)
    f = out / f"sl_index_{kind}_{d0:%Y%m}_{d1:%Y%m}.json"
    f.write_bytes(b)
    rows = json.loads(json.loads(b)["data"])
    res = []
    for o in rows:
        href = re.findall(r"href='([^']+)'", o.get("slpublic", "") or "")
        if href:
            res.append(dict(symbol=o["symbol"].lower(), name=o.get("fullName"), updated=o.get("LastUpdate"),
                            path=href[0], fname=href[0].split("/")[-1]))
    return res


def get_lists(family: str, months: list[str], out: Path, listing: list[dict] | None = None) -> dict:
    """Download the public selection lists of `family` for the given YYYYMM months (latest file per month and
    symbol). Returns {(symbol, YYYYMMDD): path}. Files already in `out` are not fetched again."""
    kind, syms = C.SL_SYMBOL[family]
    out.mkdir(parents=True, exist_ok=True)
    if listing is None:
        ms = sorted(pd.Timestamp(m + "01") for m in months)
        listing = sl_index(kind, ms[0], ms[-1], out)
    got = {}
    for s in syms:
        for m in months:
            c = [r for r in listing if r["symbol"] == s and re.search(rf"_{m}\d\d\.csv$", r["fname"])]
            if not c:
                continue
            r = sorted(c, key=lambda r: r["updated"])[-1]           # latest publication for that month
            date = re.search(r"_(\d{8})\.csv$", r["fname"]).group(1)
            p = out / r["fname"]
            if not p.exists():
                url = "https://www.stoxx.com/" + r["path"].replace("documents/stoxxnet/Documents/", "document/")
                b = http_get(url)
                if not b[:40].decode("latin-1").startswith("Creation_Date"):
                    raise RuntimeError(f"{r['fname']}: not a selection-list CSV")
                p.write_bytes(b)
            got[(s, date)] = p
    return got


def read_list(p: Path) -> pd.DataFrame:
    x = pd.read_csv(p, sep=";", dtype=str, encoding="latin-1")
    x.columns = [c.strip() for c in x.columns]
    x["rank"] = pd.to_numeric(x["Rank (FINAL)"], errors="coerce")
    x["rank_prev"] = pd.to_numeric(x["Rank (PREVIOUS)"], errors="coerce")
    x["member_code"] = x["Index Membership"].fillna("").str.strip()
    x["Comment"] = x["Comment"].fillna("").str.strip()
    x["list_date"] = pd.to_datetime(x["Creation_Date"].iloc[0], format="%Y%m%d")
    return x


# ------------------------------------------------------------------------------------- iShares holdings
def _holdings_module(out: Path):
    sys.dont_write_bytecode = True               # read-only import: leave no __pycache__ in 16_flow_engine
    if str(C.ENGINE) not in sys.path:
        sys.path.insert(0, str(C.ENGINE))
    import holdings as H                         # noqa: E402  (engine parser, unchanged)
    H.HOLD = out                                 # cache under our raw folder, never the engine's data/
    return H


def get_holdings(out: Path, offline: bool = False) -> dict[str, pd.DataFrame]:
    hd = out / "holdings"
    hd.mkdir(parents=True, exist_ok=True)
    H = _holdings_module(hd)
    res = {}
    for etf in C.FUNDS:
        stamp = out.name.replace("-", "")
        p = hd / etf / f"{etf}_live{stamp}.json"
        if not p.exists() and not offline:
            url = H.ETF_URL["ch_json"].format(pid=H.ETF[etf]["pid"])
            p.parent.mkdir(parents=True, exist_ok=True)
            b = http_get(url)
            if not b.lstrip(b"\xef\xbb\xbf").startswith(b"{"):
                raise RuntimeError(f"{etf}: holdings endpoint returned non-JSON")
            p.write_bytes(b)
        res[etf] = parse_holdings(H, etf, p)
    return res


def parse_holdings(H, etf: str, p: Path) -> pd.DataFrame:
    """holdings.parse() on an explicit file (the engine's parser keys files by date; we keep our own names)."""
    j = json.loads(p.read_text(encoding="utf-8-sig"))
    recs = []
    for r in j.get("aaData", []):
        if len(r) < 13:
            continue
        o = len(r) - 13
        recs.append(dict(ticker=str(r[0]).strip(), name=r[1], asset_class=r[3 + o], mv=H._num(H._raw(r[4 + o])),
                         weight=H._num(H._raw(r[5 + o])) / 100.0, shares=H._num(H._raw(r[7 + o])), isin=r[8 + o],
                         price=H._num(H._raw(r[9 + o])), location=r[10 + o], exchange=r[11 + o], mkt_ccy=r[12 + o]))
    df = pd.DataFrame(recs)
    df = df[(df.asset_class == "Equity") & (df.shares > 0)].copy()
    df["etf"] = etf
    df["country"] = df.exchange.map(H.EXCH_COUNTRY)
    miss = df.country.isna()
    df.loc[miss, "country"] = df.loc[miss, "isin"].astype(str).str[:2]
    df = df.groupby(["etf", "isin"], as_index=False).agg(
        ticker=("ticker", "first"), name=("name", "first"), shares=("shares", "sum"), price=("price", "first"),
        mv=("mv", "sum"), weight=("weight", "sum"), location=("location", "first"), exchange=("exchange", "first"),
        mkt_ccy=("mkt_ccy", "first"), country=("country", "first"))
    return df


# ------------------------------------------------------------------------------------- notices, RSS
def get_ftse_notices(out: Path, since: str = "2026-05-01", max_pages: int = 8) -> pd.DataFrame:
    rows = []
    for page in range(1, max_pages + 1):
        b = http_get(C.FTSE_NOTICES.format(page=page))
        (out / f"ftse_notices_UK_p{page}.json").write_bytes(b)
        j = json.loads(b)
        items = j if isinstance(j, list) else (j.get("data") or j.get("Data") or {})
        if isinstance(items, dict):
            items = items.get("NoticesList") or items.get("Notices") or items.get("Items") or []
        if not items:
            break
        rows += items
        dates = pd.to_datetime([i.get("AnnouncedDate") or i.get("announcedDate") for i in items], errors="coerce")
        if len(dates) and dates.min() < pd.Timestamp(since):
            break
    df = pd.DataFrame(rows)
    return df


def get_stoxx_rss(out: Path) -> Path:
    p = out / "stoxx_feed.xml"
    if not p.exists():
        p.write_bytes(http_get(C.STOXX_RSS))
    return p


# ------------------------------------------------------------------------------------- Yahoo
def yf_call(fn, *a, tries: int = 3, **k):
    """Call a yfinance function with the shared >= 1.6 s gap; back off on rate limits."""
    err = None
    for i in range(tries):
        _wait()
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            err = e
            msg = str(e)
            if "Too Many Requests" in msg or "Rate" in msg:
                time.sleep(60 * (i + 1))
            else:
                time.sleep(3)
    raise RuntimeError(err)


def yf_prices(symbols: list[str], out: Path, start: str = C.PRICE_START, end: str | None = None,
              tag: str = "prices", budget_min: float = 90) -> pd.DataFrame:
    """Daily OHLCV per symbol (one chart request each), resumable: raw/<date>/yahoo/<tag>.jsonl."""
    import yfinance as yf
    yd = out / "yahoo"
    yd.mkdir(exist_ok=True)
    fj = yd / f"{tag}.jsonl"
    done = set()
    if fj.exists():
        for line in fj.read_text().splitlines():
            try:
                done.add(json.loads(line)["symbol"])
            except Exception:  # noqa: BLE001
                pass
    end = end or (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    todo = [s for s in dict.fromkeys(symbols) if s and s not in done]
    t0 = time.time()
    with open(fj, "a") as fh:
        for i, s in enumerate(todo):
            if time.time() - t0 > budget_min * 60:
                print(f"  yahoo prices: time budget reached after {i} of {len(todo)}", flush=True)
                break
            try:
                t = yf.Ticker(s)
                h = yf_call(t.history, start=start, end=end, auto_adjust=False, actions=False)
                meta = getattr(t, "history_metadata", {}) or {}
                rec = dict(symbol=s, currency=meta.get("currency"), exchange=meta.get("exchangeName"),
                           name=meta.get("longName") or meta.get("shortName"), n=int(len(h)),
                           date=[d.strftime("%Y-%m-%d") for d in h.index],
                           close=h["Close"].round(6).tolist() if len(h) else [],
                           adjclose=h["Adj Close"].round(6).tolist() if len(h) else [],
                           high=h["High"].round(6).tolist() if len(h) else [],
                           low=h["Low"].round(6).tolist() if len(h) else [],
                           volume=h["Volume"].astype(float).tolist() if len(h) else [])
            except Exception as e:  # noqa: BLE001
                rec = dict(symbol=s, n=0, error=str(e)[:200])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 100 == 0:
                print(f"  yahoo prices {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return read_prices(out, tag)


def read_prices(out: Path, tag: str = "prices") -> pd.DataFrame:
    """Reads <tag>.jsonl plus any targeted retry files <tag>_retry_<date>.jsonl (later files win per symbol)."""
    fjs = [out / "yahoo" / f"{tag}.jsonl"] + sorted((out / "yahoo").glob(f"{tag}_retry_*.jsonl"))
    fjs = [f for f in fjs if f.exists()]
    if not fjs:
        return pd.DataFrame(columns=["symbol", "date", "close", "adjclose", "high", "low", "volume", "currency"])
    recs = {}
    for fj in fjs:
        for line in fj.read_text().splitlines():
            r = json.loads(line)
            if r.get("n") or r["symbol"] not in recs:
                recs[r["symbol"]] = r
    rows = []
    for r in recs.values():
        if not r.get("n"):
            continue
        d = pd.DataFrame(dict(date=pd.to_datetime(r["date"]), close=r["close"], adjclose=r["adjclose"],
                              high=r["high"], low=r["low"], volume=r["volume"]))
        d["symbol"], d["currency"], d["exchange"] = r["symbol"], r.get("currency"), r.get("exchange")
        d["yname"] = r.get("name")
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def yf_info(symbols: list[str], out: Path, tag: str = "info", budget_min: float = 45) -> pd.DataFrame:
    """Yahoo quoteSummary info (shares, float, currency, name), resumable: raw/<date>/yahoo/<tag>.jsonl."""
    import yfinance as yf
    yd = out / "yahoo"
    yd.mkdir(exist_ok=True)
    fj = yd / f"{tag}.jsonl"
    done = set()
    if fj.exists():
        done = {json.loads(l)["symbol"] for l in fj.read_text().splitlines() if l.strip()}
    keep = ["symbol", "longName", "shortName", "currency", "quoteType", "exchange", "sharesOutstanding",
            "impliedSharesOutstanding", "floatShares", "marketCap", "heldPercentInsiders", "heldPercentInstitutions",
            "country", "industry", "sector", "regularMarketPrice", "isin"]
    t0 = time.time()
    todo = [s for s in dict.fromkeys(symbols) if s and s not in done]
    with open(fj, "a") as fh:
        for i, s in enumerate(todo):
            if time.time() - t0 > budget_min * 60:
                print(f"  yahoo info: time budget reached after {i} of {len(todo)}", flush=True)
                break
            try:
                inf = yf_call(lambda: yf.Ticker(s).info) or {}
                rec = {k: inf.get(k) for k in keep}
                rec["symbol"] = s
            except Exception as e:  # noqa: BLE001
                rec = dict(symbol=s, error=str(e)[:200])
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
    rows = [json.loads(l) for l in fj.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows).drop_duplicates("symbol", keep="last")


def yf_search(queries: list[str], out: Path, tag: str = "search") -> dict:
    """Yahoo symbol search (ISIN or name), resumable: raw/<date>/yahoo/<tag>.jsonl -> {query: [quotes]}."""
    import yfinance as yf
    yd = out / "yahoo"
    yd.mkdir(exist_ok=True)
    fj = yd / f"{tag}.jsonl"
    res = {}
    if fj.exists():
        for l in fj.read_text().splitlines():
            r = json.loads(l)
            res[r["q"]] = r["quotes"]
    with open(fj, "a") as fh:
        for q in dict.fromkeys(queries):
            if not q or q in res:
                continue
            try:
                s = yf_call(lambda: yf.Search(q, max_results=10, news_count=0))
                quotes = [{k: x.get(k) for k in ("symbol", "exchange", "shortname", "longname", "quoteType",
                                                 "exchDisp")} for x in (s.quotes or [])]
            except Exception as e:  # noqa: BLE001
                quotes = [dict(error=str(e)[:200])]
            res[q] = quotes
            fh.write(json.dumps(dict(q=q, quotes=quotes)) + "\n")
            fh.flush()
    return res


# ------------------------------------------------------------------------------------- R2 (read-only)
def r2_ftse(out: Path) -> dict[str, pd.DataFrame]:
    """FTSE All-Share (3455), FTSE 100 (3670), FTSE 250 (3671) Datastream daily spells touching 2026-03..06 with ISIN,
    plus Datastream shares (numshrs) and close for those lines in Apr-Jun 2026 (UK universe; June calibration)."""
    f1, f2 = out / "r2_ftse_lists.parquet", out / "r2_ftse_shares.parquet"
    if f1.exists() and f2.exists():
        return dict(lists=pd.read_parquet(f1), shares=pd.read_parquet(f2))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # code/, where the r2 package lives
    import r2
    con = r2.connect()
    lists = r2.sql(con, """SELECT c.indexlistintcode::INT AS list, c.infocode::BIGINT AS infocode, c.startdate, c.enddate,
                                  n.isin, n.dsqtname, n.isocurrcode, n.dscode
                           FROM tr_ds_equities.ds2constdly c
                           LEFT JOIN (SELECT infocode, any_value(isin) isin, any_value(dsqtname) dsqtname,
                                             any_value(isocurrcode) isocurrcode, any_value(dscode) dscode
                                      FROM tr_ds_equities.wrds_ds_names GROUP BY infocode) n ON n.infocode = c.infocode
                           WHERE c.indexlistintcode IN (3455, 3670, 3671)
                             AND c.enddate >= DATE '2026-03-01'""")
    lists = lists.dropna(subset=["infocode"])
    ics = sorted(set(lists.infocode.astype(int)))
    con.execute("CREATE OR REPLACE TEMP TABLE wic AS SELECT unnest(?)::BIGINT AS infocode", [ics])
    sh = r2.sql(con, """SELECT d.infocode::BIGINT infocode, d.marketdate, d.numshrs::DOUBLE numshrs,
                               d."close"::DOUBLE close_, d.mktcap::DOUBLE mktcap
                        FROM tr_ds_equities.wrds_ds2dsf d JOIN wic USING (infocode)
                        WHERE d.marketdate BETWEEN DATE '2026-04-01' AND DATE '2026-06-30'""")
    lists.to_parquet(f1, index=False)
    sh.to_parquet(f2, index=False)
    return dict(lists=lists, shares=sh)
