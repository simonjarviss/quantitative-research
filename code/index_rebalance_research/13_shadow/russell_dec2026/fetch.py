"""Raw-data fetchers (public, no accounts). Everything is saved under raw/<fetch-date>/ before parsing.

Headers: a neutral browser User-Agent only; no e-mail or personal identifier is ever sent.
Sources: iShares holdings CSVs (IWB/IWM/IWV), nasdaqtrader symbol directories, the Nasdaq screener JSON,
and Yahoo Finance through yfinance (info = shares/float/country; download = daily prices).
"""
from __future__ import annotations
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

import config as C

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
INFO_FIELDS = ["quoteType", "shortName", "longName", "country", "sharesOutstanding", "impliedSharesOutstanding",
               "floatShares", "marketCap", "exchange", "industry", "sector", "firstTradeDateMilliseconds",
               "heldPercentInsiders", "currency", "financialCurrency", "regularMarketPrice", "averageVolume"]


def yf_symbol(s: str) -> str:
    """Listing / iShares / FTSE symbol -> Yahoo symbol (BRK.B, 'BRK B' -> BRK-B)."""
    return str(s).strip().replace(" ", ".").replace(".", "-").upper()


def http_get(url: str, accept: str | None = None, timeout: int = 90, tries: int = 3) -> requests.Response:
    h = {"User-Agent": C.UA}
    if accept:
        h["Accept"] = accept
    err = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"GET failed {url}: {err}")


# ---------------------------------------------------------------- iShares holdings
def fetch_holdings(outdir: Path) -> dict:
    out = {}
    for fund, (pid, slug) in C.ISHARES.items():
        body = None
        for url in (C.ISHARES_URL.format(pid=pid, slug=slug), C.ISHARES_AJAX.format(pid=pid, slug=slug, fund=fund)):
            try:
                r = http_get(url)
            except RuntimeError:
                continue
            txt = r.content.decode("utf-8-sig", errors="replace")
            if "Ticker," in txt[:3000] and "<html" not in txt[:500].lower():
                body = r.content
                break
        if body is None:
            raise RuntimeError(f"{fund}: no CSV from iShares (both URLs returned HTML / errors)")
        asof, _ = parse_holdings_bytes(body)
        p = outdir / f"{fund}_holdings_asof{asof:%Y-%m-%d}.csv"
        p.write_bytes(body)
        out[fund] = p
    return out


def parse_holdings_bytes(b: bytes):
    raw = b.decode("utf-8-sig", errors="replace").splitlines()
    asof = None
    for l in raw[:12]:
        if l.startswith("Fund Holdings as of"):
            asof = pd.to_datetime(l.split(",", 1)[1].strip().strip('"'))
    i = next(k for k, l in enumerate(raw) if l.startswith("Ticker,"))
    body = [l for l in raw[i:] if l.count(",") >= 10]          # drop the footer
    df = pd.read_csv(io.StringIO("\n".join(body)), thousands=",", dtype={"Ticker": str})
    return asof, df


def load_holdings(rawdir: Path) -> tuple[pd.DataFrame, dict]:
    frames, asofs = [], {}
    for fund in C.ISHARES:
        ps = sorted(rawdir.glob(f"{fund}_holdings_asof*.csv"))
        if not ps:
            raise FileNotFoundError(f"no {fund} holdings in {rawdir}")
        asof, df = parse_holdings_bytes(ps[-1].read_bytes())
        df["fund"] = fund
        asofs[fund] = asof
        frames.append(df)
    h = pd.concat(frames, ignore_index=True)
    h = h[h["Asset Class"] == "Equity"].copy()
    for c in ["Market Value", "Quantity", "Price", "Weight (%)"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["Ticker"] = h["Ticker"].fillna("-").astype(str)
    h["sym"] = h["Ticker"].str.strip().str.replace(" ", ".")
    h["yf"] = h["Ticker"].map(yf_symbol)
    bad_exch = h["Exchange"].isin(["NO MARKET (E.G. UNLISTED)", "-", "Non-Nms Quotation Service (Nnqs)"])
    bad_name = h["Name"].str.contains(r"\bCVR\s*$|- CVR\b|\bESCROW\b|\bRIGHTS\b|\bVESTING\b|\bRESTRICTION\b|\bCONTRA\b|\bPRVT\b",
                                      case=False, regex=True)
    h["residual"] = bad_exch | bad_name | (h["Ticker"] == "-") | (h["Market Value"] < 1000)
    return h, asofs


# ---------------------------------------------------------------- nasdaqtrader symbol directories
def fetch_symdir(outdir: Path) -> dict:
    out = {}
    for k, url in C.SYMDIR.items():
        r = http_get(url)
        p = outdir / f"{k}.txt"
        p.write_bytes(r.content)
        out[k] = p
    return out


def load_symdir(rawdir: Path) -> pd.DataFrame:
    n = pd.read_csv(rawdir / "nasdaqlisted.txt", sep="|", dtype=str)
    n = n[~n.Symbol.fillna("").str.startswith("File Creation")]
    n = pd.DataFrame({"sym": n.Symbol, "name": n["Security Name"], "exch": "Q", "etf": n.ETF,
                      "test": n["Test Issue"], "fin_status": n["Financial Status"], "src": "nasdaqlisted"})
    o = pd.read_csv(rawdir / "otherlisted.txt", sep="|", dtype=str)
    o = o[~o["ACT Symbol"].fillna("").str.startswith("File Creation")]
    o = pd.DataFrame({"sym": o["ACT Symbol"], "name": o["Security Name"], "exch": o.Exchange, "etf": o.ETF,
                      "test": o["Test Issue"], "fin_status": None, "src": "otherlisted"})
    d = pd.concat([n, o], ignore_index=True).dropna(subset=["sym"])
    d["name"] = d["name"].fillna("").str.strip()
    d["yf"] = d.sym.map(yf_symbol)
    return d.drop_duplicates("sym")


# ---------------------------------------------------------------- Nasdaq screener (cross-check / fallback)
def fetch_screener(outdir: Path) -> Path:
    r = http_get(C.NASDAQ_SCREENER, accept="application/json")
    p = outdir / "nasdaq_screener.json"
    p.write_bytes(r.content)
    return p


def load_screener(rawdir: Path) -> pd.DataFrame:
    p = rawdir / "nasdaq_screener.json"
    if not p.exists():
        return pd.DataFrame(columns=["yf", "nd_cap", "nd_px", "nd_country", "nd_ipoyear"])
    rows = json.loads(p.read_text())["data"]["rows"]
    s = pd.DataFrame(rows)
    s["yf"] = s.symbol.map(lambda x: yf_symbol(str(x).replace("/", ".").replace("^", ".P")))
    s["nd_cap"] = pd.to_numeric(s.marketCap, errors="coerce")
    s["nd_px"] = pd.to_numeric(s.lastsale.str.replace("$", "", regex=False), errors="coerce")
    s["nd_country"] = s.country.replace("", None)
    s["nd_ipoyear"] = pd.to_numeric(s.ipoyear, errors="coerce")
    return s[["yf", "nd_cap", "nd_px", "nd_country", "nd_ipoyear"]].drop_duplicates("yf")


# ---------------------------------------------------------------- Yahoo info (shares, float, country)
class _Gate:
    """Shared throttle: a minimum gap between Yahoo info calls across threads, and a global pause on HTTP 429."""
    def __init__(self, gap: float):
        import threading
        self.lock, self.gap, self.next_ok, self.pause_until, self.n429, self.streak = threading.Lock(), gap, 0.0, 0.0, 0, 0

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                t = max(self.next_ok, self.pause_until)
                if now >= t:
                    self.next_ok = now + self.gap
                    return
            time.sleep(min(t - now, 5))

    def limited(self):
        with self.lock:
            now = time.time()
            if now < self.pause_until:          # already paused by another thread: don't escalate
                return self.pause_until - now
            self.n429 += 1
            self.streak += 1
            back = min(45 * 2 ** (self.streak - 1), 300)
            self.pause_until = now + back
            return back

    def ok(self):
        with self.lock:
            self.streak = 0


def _one_info(t: str, gate: _Gate, deadline: float) -> dict:
    import yfinance as yf
    last = ""
    for k in range(6):
        if time.time() > deadline:
            return {"t": t, "ok": False, "err": "time budget", "ts": time.time()}
        gate.wait()
        try:
            i = yf.Ticker(t).info or {}
            gate.ok()
            d = {f: i.get(f) for f in INFO_FIELDS}
            d.update(t=t, ok=bool(i.get("quoteType")), ts=time.time())
            return d
        except Exception as e:
            last = repr(e)[:200]
            if "Rate" in last or "429" in last or "Too Many" in last:
                gate.limited()
            else:
                time.sleep(2 * (k + 1))
    return {"t": t, "ok": False, "err": last, "ts": time.time()}


def fetch_info(tickers, outdir: Path, workers: int = 4, gap: float = 0.4, max_minutes: float = 45,
               log=print) -> pd.DataFrame:
    """Throttled Yahoo info with resume: results stream (line-flushed) to raw/<date>/yf_info.jsonl.
    Stops after max_minutes; callers fall back to the Nasdaq screener for anything missing."""
    p = outdir / "yf_info.jsonl"
    done = {}
    if p.exists():
        for l in p.read_text().splitlines():
            try:
                d = json.loads(l)
                if d.get("ok"):
                    done[d["t"]] = d
            except json.JSONDecodeError:
                pass
    todo = [t for t in dict.fromkeys(tickers) if t not in done]
    log(f"  yahoo info: {len(done)} cached, {len(todo)} to fetch ({workers} threads, >= {gap}s apart, budget {max_minutes:.0f} min)")
    t0 = time.time()
    deadline = t0 + 60 * max_minutes
    gate = _Gate(gap)
    with open(p, "a", buffering=1) as fh, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_one_info, t, gate, deadline): t for t in todo}
        for k, f in enumerate(as_completed(futs), 1):
            d = f.result()
            fh.write(json.dumps(d, default=str) + "\n")
            if d.get("ok"):
                done[d["t"]] = d
            if k % 250 == 0:
                log(f"    {k}/{len(todo)}  {time.time() - t0:.0f}s  rate-limit pauses {gate.n429}")
    log(f"  yahoo info done: {sum(t in done for t in tickers)}/{len(set(tickers))} ok, {gate.n429} rate-limit pauses, "
        f"{time.time() - t0:.0f}s")
    df = pd.DataFrame(list(done.values()))
    return df[df.t.isin(set(tickers))].reset_index(drop=True) if len(df) else pd.DataFrame(columns=["t"] + INFO_FIELDS)


def load_info(rawdir: Path) -> pd.DataFrame:
    p = rawdir / "yf_info.jsonl"
    rows = {}
    if not p.exists():
        return pd.DataFrame(columns=["t"] + INFO_FIELDS)
    for l in p.read_text().splitlines():
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue
        if d.get("ok"):
            rows[d["t"]] = d
    return pd.DataFrame(list(rows.values())) if rows else pd.DataFrame(columns=["t"] + INFO_FIELDS)


# ---------------------------------------------------------------- Yahoo daily prices
def fetch_prices(tickers, start, end, outdir: Path, tag: str, chunk: int = 400, log=print) -> dict:
    """Batch yf.download -> raw/<date>/prices_<tag>_{close,adjclose,volume}.parquet (wide, columns = Yahoo symbols).
    end is inclusive here (passed to yfinance as end + 1 day)."""
    import yfinance as yf
    tickers = list(dict.fromkeys(tickers))
    s, e = pd.Timestamp(start).date(), (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    parts = {"Close": [], "Adj Close": [], "Volume": []}

    def grab(tk):
        df = yf.download(tk, start=s, end=e, auto_adjust=False, actions=False, threads=True,
                         progress=False, group_by="column")
        if df is None or df.empty:
            return
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_product([df.columns, tk[:1]])
        for f in parts:
            if f in df.columns.get_level_values(0):
                parts[f].append(df[f])

    t0 = time.time()
    for i in range(0, len(tickers), chunk):
        grab(tickers[i:i + chunk])
        log(f"    prices {min(i + chunk, len(tickers))}/{len(tickers)}  {time.time() - t0:.0f}s")
    out = {}
    for f, L in parts.items():
        w = pd.concat(L, axis=1) if L else pd.DataFrame()
        w = w.loc[:, ~w.columns.duplicated()]
        out[f] = w
    # retry tickers with no data at all, in small chunks
    have = out["Close"].columns[out["Close"].notna().any()] if len(out["Close"]) else []
    miss = [t for t in tickers if t not in set(have)]
    if miss and len(miss) < len(tickers):
        log(f"    retrying {len(miss)} tickers with no prices")
        for f in parts:
            parts[f] = [out[f].drop(columns=[c for c in miss if c in out[f].columns])]
        for i in range(0, len(miss), 50):
            grab(miss[i:i + 50])
        for f, L in parts.items():
            w = pd.concat(L, axis=1)
            out[f] = w.loc[:, ~w.columns.duplicated()]
    for f, w in out.items():
        w = w.sort_index()
        ix = pd.to_datetime(w.index)
        w.index = (ix.tz_localize(None) if ix.tz is not None else ix).normalize()
        out[f] = w
        w.to_parquet(outdir / f"prices_{tag}_{f.replace(' ', '').lower()}.parquet")
    return out


def load_prices(rawdir: Path, tag: str) -> dict:
    out = {}
    for f in ["Close", "Adj Close", "Volume"]:
        p = rawdir / f"prices_{tag}_{f.replace(' ', '').lower()}.parquet"
        out[f] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out
