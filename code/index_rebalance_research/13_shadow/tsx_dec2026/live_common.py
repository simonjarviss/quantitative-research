"""Shared helpers for the live S&P/TSX Composite shadow predictions (paper-only research).

Data sources (all free, no account, neutral User-Agent, no personal identifiers in any request):
  XIC holdings CSV (BlackRock Canada)   current members + S&P float shares (XIC shares held / k)
  TMX MiG "TSX & TSXV Listed Companies"  universe, issuer O/S shares, listing date, HQ region, US interlisting
  TSX company directory JSON            share-class lines per issuer
  yfinance (.TO and US tickers)          daily close and volume (TSX-only for .TO lines), sharesOutstanding, floatShares
  R2 study files (read-only, ../../12_studies/tsx_composite/data)
                                         Datastream/Worldscope float proxies (to 2026-07), consolidated/TSX volume
                                         ratio per name (to 2026-06-24), Compustat domicile and US tickers,
                                         SDC pending deals (to 2026-06-26), backtest R-screens (committee memory)

Rules: ../../11_methodology/specs/spdji/sp_tsx_composite.md (SPTSX-M June 2025 archived edition; the
11 Sep 2026 S&P/TSX Indices Update is UNREAD). Same screen as ../../12_studies/tsx_composite/04_predict.py
with the current-era parameters (common.rules_for for reviews >= 2025-06).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CACHE = HERE / "cache"
RUNS = HERE / "runs"
BT = HERE.parent.parent / "12_studies" / "tsx_composite"          # backtest (read-only)
BT_DATA = BT / "data"
REVIEW_CHANGES = HERE / "review_changes.csv"
MA_FILE = HERE / "ma_pending.csv"
LOG = HERE / "shadow_log.md"

# Neutral browser-like User-Agent. Never put an e-mail or other personal identifier in headers.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

XIC_URL = ("https://www.blackrock.com/ca/investors/en/products/239837/ishares-core-sptsx-capped-composite-index-etf/"
           "1464253357814.ajax?fileType=csv&fileName=XIC_holdings&dataType=fund")
MIG_URL = "https://www.tsx.com/en/resource/571"
TSXDIR_URL = "https://www.tsx.com/json/company-directory/search/tsx/%5E*"

# Current-era rule parameters (SPTSX-M June 2025, pp.5-6; params/sp_tsx_composite.yaml)
RULES = dict(size_entry=0.0004, size_stay=0.00025, turn_entry=0.50, turn_stay=0.25,
             ca_only_entry=0.25, ca_only_stay=0.125, price_min=1.0, vwap_days=10, cap_switch=0.10,
             seasoning_months=6, ban_reviews=4)

# US tickers that differ from both Compustat (R2) and the TSX root (Yahoo symbols). Extend when a run flags one.
US_TICKER_OVERRIDES = {"GRT.UN": "GRP-U"}
US_EXCH_MIG = {"NYSE", "NYSE Mkt", "NasdaqGS", "NasdaqCM", "NasdaqGM", "NYSE Arca"}
NON_OPERATING_SECTORS = {"ETP", "CDR", "Closed-End Funds", "SPAC"}
NON_OPERATING_SPTYPE = {"Exchange Traded Funds", "CDR", "Split Shares", "Fund of Equities", "Commodity Funds",
                        "Fund of Debt", "Fund of Multi-Asset/Other", "Exchange Traded Receipt", "Fixed Income",
                        "Other"}
# Share-class suffixes kept as equity lines. Everything else (.PR.x, .PF.x, .DB.x, .WT.x, .U, .NT, .RT, .IR, .R,
# .NO, .WS ...) is a preferred, debenture, warrant, USD line, note, right or receipt and is excluded.
CLASS_OK = re.compile(r"^[A-Z0-9]+(\.(A|B|C|X|Y|Z|K|UN|SV|VS|MV|NV|LV))?$")


# ----------------------------------------------------------------------------------------------- utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, timeout: int = 60, retries: int = 2) -> bytes:
    last = None
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status} for {url}")
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def yf_symbol(tsx: str) -> str:
    """TSX symbol -> Yahoo symbol: 'TECK.B' -> 'TECK-B.TO', 'REI.UN' -> 'REI-UN.TO'."""
    return tsx.replace(".", "-") + ".TO"


def norm_tic(t: str) -> str:
    """Compustat Canadian ticker ('AEM.', 'TECK.B') -> TSX symbol ('AEM', 'TECK.B')."""
    return str(t).rstrip(".")


# ------------------------------------------------------------------------------------------------ calendar
_CAL = {}


def sessions(exch: str = "XTSE") -> pd.DatetimeIndex:
    if exch not in _CAL:
        import exchange_calendars as xc
        c = xc.get_calendar(exch, start="2015-01-01", end="2028-12-31")
        _CAL[exch] = pd.DatetimeIndex(c.sessions.tz_localize(None) if c.sessions.tz is not None else c.sessions)
    return _CAL[exch]


def nth_weekday(y, m, weekday, n):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def last_on_or_before(cal, d):
    return cal[cal.searchsorted(pd.Timestamp(d), side="right") - 1]


def review_dates(review: str) -> dict:
    """Timeline of a quarterly review (spec §5.3, §10 Step 0).

    R = 10 TSX business days before the first Friday (SPTSX-M p.9); R_alt also skips NYSE holidays (spec §11.4).
    W10 = the 10 TSX sessions before R. A = first Friday (practice Sep 2024-Jun 2026), A_latest = T - 5 sessions.
    T = 3rd-Friday close; eff = next session; C = share/IWF cut-off (3rd Friday - 35 days, SPDJI-PP p.16).
    H3 = last session on or before the 3rd Friday minus 3 calendar months (backtest definition).
    """
    y, m = int(review[:4]), int(review[5:7])
    cal = sessions("XTSE")
    both = cal.intersection(sessions("XNYS"))
    f1 = nth_weekday(y, m, 4, 1)
    i = cal.searchsorted(f1)
    R = cal[i - 10]
    j = both.searchsorted(f1)
    R_alt = both[j - 10]
    t_nom = nth_weekday(y, m, 4, 3)
    T = last_on_or_before(cal, t_nom)
    k = cal.get_loc(T)
    return dict(review=review, F1=f1, R=R, R_alt=R_alt, W10_start=cal[i - 20], W10_end=cal[i - 11],
                A=last_on_or_before(cal, f1), A_latest=cal[k - 5], T=T, eff=cal[k + 1],
                C=t_nom - pd.Timedelta(days=35), H3=last_on_or_before(cal, t_nom - pd.DateOffset(months=3)),
                seasoning_cutoff=pd.Timestamp(R.year, R.month, 1) - pd.DateOffset(months=RULES["seasoning_months"]))


def prev_reviews(review: str, n: int) -> list[str]:
    y, m = int(review[:4]), int(review[5:7])
    out = []
    for _ in range(n):
        m -= 3
        if m <= 0:
            y, m = y - 1, m + 12
        out.append(f"{y}-{m:02d}")
    return out


def next_review_after(d: pd.Timestamp) -> str:
    """First quarterly review whose reference date is on or after d."""
    y, m = d.year, d.month
    m = ((m - 1) // 3 + 1) * 3
    for _ in range(6):
        rv = f"{y}-{m:02d}"
        if review_dates(rv)["R"] >= d:
            return rv
        m += 3
        if m > 12:
            y, m = y + 1, m - 12
    raise ValueError(d)


# --------------------------------------------------------------------------------------------- raw sources
def fetch_xic(asof: pd.Timestamp, offline: bool = False) -> Path:
    """XIC holdings as of the last TSX session before `asof` (the file is published T-1).

    For a run on the as-of date itself, the live file is downloaded (it is dated the previous session) and
    stamped with the download date. For a historical as-of date, the asOfDate parameter is used.
    """
    d = RAW / "xic"
    d.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    if asof >= today - pd.Timedelta(days=1):
        p = d / f"XIC_holdings_{today:%Y%m%d}.csv"
        url = XIC_URL
    else:
        cal = sessions()
        prev = cal[cal.searchsorted(asof) - 1]          # holdings as of the session before asof
        p = d / f"XIC_holdings_asof{prev:%Y%m%d}.csv"
        url = XIC_URL + f"&asOfDate={prev:%Y%m%d}"
    if not p.exists():
        if offline:
            raise FileNotFoundError(p)
        p.write_bytes(http_get(url))
    return p


def parse_xic(p: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    first = open(p, encoding="utf-8-sig").readline()
    asof = pd.Timestamp(re.search(r'"(.+)"', first).group(1))
    x = pd.read_csv(p, skiprows=2, thousands=",", keep_default_na=False, na_values=[""])  # 'NA' = National Bank
    e = x[(x["Asset Class"] == "Equity") & (x["Price"] > 0)].copy()                       # drops the odd 0-price row
    e = e.rename(columns={"Ticker": "symbol", "Name": "xic_name", "Shares": "xic_shares", "Price": "xic_price",
                          "Market Value": "xic_mv", "Weight (%)": "xic_weight_pct"})
    e["symbol"] = e.symbol.astype(str).str.strip()
    return asof, e[["symbol", "xic_name", "xic_shares", "xic_price", "xic_mv", "xic_weight_pct"]].reset_index(drop=True)


def fetch_mig(offline: bool = False) -> Path:
    d = RAW / "mig"
    d.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    p = d / f"tsx_mig_listed_{today:%Y%m%d}.xlsx"
    if not p.exists():
        if offline:
            cands = sorted(d.glob("tsx_mig_listed_*.xlsx"))
            if not cands:
                raise FileNotFoundError(p)
            return cands[-1]
        p.write_bytes(http_get(MIG_URL))
    return p


def parse_mig(p: Path) -> tuple[str, pd.DataFrame]:
    xl = pd.ExcelFile(p)
    sheet = [s for s in xl.sheet_names if s.startswith("TSX Issuers")][0]
    raw = pd.read_excel(xl, sheet, header=None, nrows=30)
    h = next(i for i in range(30) if "Root\nTicker" in [str(v) for v in raw.iloc[i].tolist()]
             or "Root Ticker" in [str(v) for v in raw.iloc[i].tolist()])
    df = pd.read_excel(xl, sheet, header=h, keep_default_na=False, na_values=[""])
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    mc = next(c for c in df.columns if c.startswith("Market Cap"))
    os_ = next(c for c in df.columns if c.startswith("O/S Shares"))
    vol = next(c for c in df.columns if c.startswith("Volume YTD"))
    asof_txt = mc.replace("Market Cap (C$)", "").strip()
    out = pd.DataFrame(dict(root=df["Root Ticker"].astype(str).str.strip(), issuer=df["Name"], mig_mcap=df[mc],
                            mig_os=df[os_], interlisted1=df["Interlisted I"], interlisted2=df["Interlisted II"],
                            sector=df["Sector"], hq_region=df["HQ Region"], hq_loc=df["HQ Location"],
                            listing_type=df["Listing Type"], listing_date=pd.to_datetime(
                                df["Listing Date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"),
                            tsxv_grad=df["TSX Venture Grad"].eq("Y") if "TSX Venture Grad" in df else False,
                            sp_flag=df["S&P/TSX Index"].astype(str).replace("nan", ""),
                            sp_type=df["SP_Type"], mig_vol_ytd=df[vol]))
    out["us_listed_mig"] = out.interlisted1.isin(US_EXCH_MIG) | out.interlisted2.isin(US_EXCH_MIG)
    out["operating"] = ~out.sector.isin(NON_OPERATING_SECTORS) & ~out.sp_type.isin(NON_OPERATING_SPTYPE)
    return asof_txt, out


def fetch_tsxdir(offline: bool = False) -> Path:
    d = RAW / "tsxdir"
    d.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().normalize()
    p = d / f"tsx_directory_{today:%Y%m%d}.json"
    if not p.exists():
        if offline:
            cands = sorted(d.glob("tsx_directory_*.json"))
            if not cands:
                raise FileNotFoundError(p)
            return cands[-1]
        p.write_bytes(http_get(TSXDIR_URL))
    return p


def parse_tsxdir(p: Path) -> pd.DataFrame:
    j = json.loads(p.read_text())
    rows = []
    for r in j["results"]:
        for ins in r.get("instruments", []):
            rows.append(dict(root=r["symbol"], issuer_dir=r["name"], symbol=ins["symbol"], line_name=ins["name"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- R2 study (read-only)
def r2_links() -> pd.DataFrame:
    """One row per Compustat Canadian line in the backtest panel, keyed by current TSX symbol.

    Adds: last priced date (to resolve reused tickers), consolidated/TSX Canadian volume ratio over the last
    252 R2 sessions, US exchange ticker (Compustat US line, exchg 11/12/14), domicile fields.
    """
    cache = CACHE / "r2_links.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    L = pd.read_parquet(BT_DATA / "panel_lines.parquet")
    z = np.load(BT_DATA / "panel.npz")
    P, VT, VC = z["P"], z["VT"], z["VC"]
    cal = pd.DatetimeIndex(pd.read_parquet(BT_DATA / "tsx_calendar.parquet").date)
    has = ~np.isnan(P)
    last_pos = np.where(has.any(0), len(cal) - 1 - has[::-1].argmax(0), -1)
    s = max(0, len(cal) - 252)
    vt, vc = VT[s:], VC[s:]
    both = ~np.isnan(vt) & ~np.isnan(vc) & (vt > 0)
    num = np.where(both, np.fmax(vc, vt), 0).sum(0)
    den = np.where(both, vt, 0).sum(0)
    ratio = np.where((both.sum(0) >= 60) & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    lines = pd.read_parquet(BT_DATA / "lines.parquet")
    us = lines[lines.exchg.astype(str).isin(["11", "12", "14"]) & (lines.secstat == "A") & ~lines.iid.str.endswith("C")]
    us = us.sort_values("iid").drop_duplicates("gvkey")
    out = pd.DataFrame(dict(lid=L.lid, gvkey=L.gvkey, symbol=L.tic.map(norm_tic), conm=L.conm, cusip=L.cusip,
                            isin=L["isin"], fic=L.fic, loc=L["loc"], gsubind=L.gsubind, line=np.arange(len(L)),
                            last_px=cal[np.clip(last_pos, 0, None)], has_px=last_pos >= 0, cons_ratio=ratio))
    out["us_tic_r2"] = out.gvkey.map(dict(zip(us.gvkey, us.tic)))
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def r2_float_at(asof: pd.Timestamp) -> pd.DataFrame:
    """Latest Datastream / Worldscope float proxy per panel line at or before asof (backtest float_monthly)."""
    fm = pd.read_parquet(BT_DATA / "float_monthly.parquet")
    fm["month"] = pd.to_datetime(fm.month)
    fm = fm[fm.month <= asof].sort_values("month")
    g = fm.groupby("line").last()
    return g[["month", "ff_ds", "ff_ws"]].rename(columns={"month": "ff_month"}).reset_index()


def sdc_pending(asof: pd.Timestamp) -> pd.DataFrame:
    """SDC control deals (>= 50% sought) live at asof, same filter as 05_filters.py. SDC ends 2026-06-26."""
    s = pd.read_parquet(BT_DATA / "sdc_ma_canada.parquet")
    s = s[(s.psought.fillna(s.pctacq) >= 50) & s.status.isin(["Completed", "Pending", "Unconditional"])].copy()
    for c in ["dateann", "dateeff", "datewith"]:
        s[c] = pd.to_datetime(s[c])
    live = (s.dateann <= asof) & (s.datewith.isna() | (s.datewith > asof)) & (
        (s.dateeff > asof) | (s.dateeff.isna() & (s.status == "Pending") & (s.dateann >= asof - pd.Timedelta(days=365))))
    return s[live][["master_cusip", "tmanames", "amanames", "dateann", "dateeff", "status", "deal_value"]]


# ------------------------------------------------------------------------------------------------ yfinance
def yf_history(symbols: list[str], start: str, end: str, tag: str, offline: bool = False) -> dict:
    """Daily Close / Adj Close / Volume for many symbols, cached per run tag (incremental).

    Close is split-adjusted only (auto_adjust=False); .TO volume is TSX-only (checked against Compustat
    cshtrd, ratio 1.000 on 40 members, Jan-Jun 2026). Rate-limited symbols are retried with back-off; symbols
    that return no rows for another reason (delisted, renamed) go to cache/yf_missing_<tag>.txt.
    """
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"yf_hist_{tag}.parquet"
    pm = CACHE / f"yf_missing_{tag}.txt"
    d = pd.read_parquet(p) if p.exists() else None
    have = set(d.columns.get_level_values(1)) if d is not None else set()
    missing = set(pm.read_text().split()) if pm.exists() else set()
    need = sorted(set(symbols) - have - missing)
    if need and offline:
        raise FileNotFoundError(f"{p} lacks {need[:5]}")

    def pull(chunk):
        import yfinance.shared as yfs
        x = yf.download(chunk, start=start, end=end, auto_adjust=False, progress=False, threads=True,
                        group_by="column")
        errs = dict(getattr(yfs, "_ERRORS", {}) or {})
        limited = {s for s, e in errs.items() if "Rate" in str(e) or "Too Many" in str(e)}
        if x is None or x.empty:
            return None, set(chunk) if not errs else limited
        if not isinstance(x.columns, pd.MultiIndex):
            x = pd.concat({k: x[[k]].rename(columns={k: chunk[0]}) for k in ["Close", "Adj Close", "Volume"]}, axis=1)
        x = x[["Close", "Adj Close", "Volume"]]
        ok = [s for s in chunk if s in x["Close"] and x["Close"][s].notna().any()]
        return (x.loc[:, (slice(None), ok)] if ok else None), limited - set(ok)

    frames, failed = [], []
    todo = list(need)
    for attempt in range(5):
        retry = []
        for i in range(0, len(todo), 100):
            chunk = todo[i:i + 100]
            got, limited = pull(chunk)
            okset = set() if got is None else set(got.columns.get_level_values(1))
            if got is not None:
                frames.append(got)
            retry += [s for s in chunk if s in limited]
            failed += [s for s in chunk if s not in okset and s not in limited]   # genuinely no data
            time.sleep(2)
        if not retry:
            break
        time.sleep(45 * (attempt + 1))                           # Yahoo rate limit: back off, retry those only
        todo = retry
    else:
        retry = todo
    if frames:
        new = pd.concat(frames, axis=1)
        new.index = pd.DatetimeIndex(new.index).tz_localize(None).normalize()
        d = new if d is None else pd.concat([d, new], axis=1)
        d = d.loc[:, ~d.columns.duplicated()].sort_index()
        d.to_parquet(p)
    if failed:
        pm.write_text("\n".join(sorted(missing | set(failed))))
    if d is None or retry:
        raise RuntimeError(f"Yahoo still rate-limiting {len(retry)} symbols; re-run later (cache keeps what was fetched)")
    return {k: d[k] for k in ["Close", "Adj Close", "Volume"]}


def yf_info(symbols: list[str], tag: str, offline: bool = False, workers: int = 4) -> pd.DataFrame:
    """sharesOutstanding, floatShares, country, quoteType per Yahoo symbol (cached per run tag)."""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"yf_info_{tag}.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["yf"])
    ok_old = old[old["quoteType"].notna()] if "quoteType" in old else old.iloc[0:0]
    old = ok_old                                                    # rows that failed earlier are retried
    need = sorted(set(symbols) - set(old.yf))
    if need and offline:
        raise FileNotFoundError(f"{p} lacks {need[:5]}")
    keys = ["sharesOutstanding", "floatShares", "impliedSharesOutstanding", "country", "quoteType", "longName",
            "currency", "exchange"]

    def one(s):
        for k in range(3):
            try:
                i = yf.Ticker(s).info or {}
                if i.get("quoteType") or k == 2:
                    return dict(yf=s, **{c: i.get(c) for c in keys})
            except Exception as e:  # noqa: BLE001
                if "404" in str(e) or "Not Found" in str(e):
                    return dict(yf=s)
            time.sleep(5 * (k + 1))
        return dict(yf=s)

    if need:
        with ThreadPoolExecutor(workers) as ex:
            new = pd.DataFrame(list(ex.map(one, need)))
        old = pd.concat([old, new], ignore_index=True)
        for c in ["sharesOutstanding", "floatShares", "impliedSharesOutstanding"]:
            if c in old:
                old[c] = pd.to_numeric(old[c], errors="coerce")
        for c in ["country", "quoteType", "longName", "currency", "exchange"]:
            if c in old:
                old[c] = old[c].astype("string")
        old.to_parquet(p, index=False)
    return old[old.yf.isin(symbols)].reset_index(drop=True)


# -------------------------------------------------------------------------------- review history and M&A
def load_review_changes() -> pd.DataFrame:
    if not REVIEW_CHANGES.exists():
        return pd.DataFrame(columns=["review", "side", "symbol", "name", "source"])
    return pd.read_csv(REVIEW_CHANGES, dtype=str, keep_default_na=False)


def load_ma(asof: pd.Timestamp) -> tuple[pd.DataFrame, set]:
    """Hand-maintained M&A list (ma_pending.csv).

    Returns (live rows, cancel set): a row is live if announced <= asof and not closed/terminated by asof and
    its status is not terminated/not_control; the cancel set holds symbols whose SDC flag should be ignored
    (deal terminated or not a change of control by asof)."""
    if not MA_FILE.exists():
        return pd.DataFrame(columns=["symbol"]), set()
    m = pd.read_csv(MA_FILE, dtype=str, keep_default_na=False, comment="#")
    ann = pd.to_datetime(m.announced, errors="coerce")
    closed = pd.to_datetime(m.closed_or_terminated.replace("", None), errors="coerce")
    dead = m.status.isin(["terminated", "not_control"]) & (closed.isna() | (closed <= asof))
    live = (ann <= asof) & (closed.isna() | (closed > asof)) & ~m.status.isin(["terminated", "not_control"])
    return m[live].reset_index(drop=True), set(m.loc[dead, "symbol"])
