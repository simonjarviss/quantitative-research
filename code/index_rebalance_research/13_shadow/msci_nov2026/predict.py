"""MSCI November 2026 Index Review: shadow prediction of MSCI Canada and MSCI Europe Standard changes
(additions, deletions and Standard <-> Small Cap migrations). Paper only; public data; no accounts.

    python3 predict.py [--asof YYYY-MM-DD] [--fetch-date YYYY-MM-DD] [--tag TAG] [--no-fetch]
    python3 predict.py --replay 2026-05          # calibration replay on the May 2026 review (writes replay/)

As-of = the holdings date (iShares UCITS files are requested for that date; default = the last business day
before today). Steps per market (Canada; DM Europe as one market, eu §0):
  1. company table on the as-of date (build.company_table): segment from CSCA / IMEA (Standard) and WSML (Small
     Cap), full company cap in USD incl. unlisted classes, float cap and FIF, ATVR, relative volatility, seasoning;
  2. the previous review's cutoff C from its actual changes at its price-cutoff prices (model.bracket), moved to the
     as-of date by the Standard members' price change around C (model.scale_factor);
  3. known-excluded non-members (qualified on size at the previous review and were not added) and Small Cap names
     that were above 1.5 C last time but stayed (blocked by float / EPI / liquidity);
  4. rules and probabilities (model.assign; random walk of the stock vs the market to the expected price cutoff,
     plus cutoff and data error), NOC fill (model.noc_fill);
  5. outputs in predictions/ (or replay/) and an append-only block with SHA-256 hashes in shadow_log.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import build as B
import config as C
import fetch as F
import model as M

T0 = time.time()
STD_FUND = {"CA": "CSCA", "EU": "IMEA"}
YF_SUFFIX = {"London Stock Exchange": ".L", "Nyse Euronext - Euronext Paris": ".PA", "Xetra": ".DE",
             "Deutsche Boerse Xetra": ".DE", "SIX Swiss Exchange": ".SW", "Nasdaq Omx Nordic": ".ST",
             "Borsa Italiana": ".MI", "Euronext Amsterdam": ".AS", "Bolsa De Madrid": ".MC",
             "Omx Nordic Exchange Copenhagen A/S": ".CO", "Nasdaq Omx Helsinki Ltd.": ".HE",
             "Nyse Euronext - Euronext Brussels": ".BR", "Oslo Bors Asa": ".OL", "Wiener Boerse Ag": ".VI",
             "Irish Stock Exchange - All Market": ".IR", "Nyse Euronext - Euronext Lisbon": ".LS",
             "Toronto Stock Exchange": ".TO", "NASDAQ": "", "New York Stock Exchange Inc.": ""}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archive_old(p: Path) -> Path:
    """Never overwrite a logged prediction: move an existing file to <dir>/superseded/ first."""
    if p.exists():
        d = p.parent / "superseded"
        d.mkdir(exist_ok=True)
        ts = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
        p.rename(d / f"{p.stem}__{ts}{p.suffix}")
    return p


def yf_symbol(ticker: str, exchange: str) -> str | None:
    if not isinstance(ticker, str) or exchange not in YF_SUFFIX:
        return None
    t = ticker.strip().rstrip(".").replace(" ", "-").replace("/", "-")
    if exchange == "Toronto Stock Exchange":
        t = t.replace(".", "-")
    return t + YF_SUFFIX[exchange]


def yf_close(symbols: list[str], start: str, end: str, tag: str, offline: bool) -> pd.DataFrame:
    """Daily closes from Yahoo, cached per tag (cache/yf_close_<tag>.parquet); only missing symbols are pulled."""
    p = C.CACHE / f"yf_close_{tag}.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    need = sorted(set(s for s in symbols if s) - set(have.columns))
    if need and not offline:
        import yfinance as yf
        frames = []
        for i in range(0, len(need), 80):
            x = yf.download(need[i:i + 80], start=start, end=end, auto_adjust=False, progress=False, threads=True)
            if x is None or x.empty:
                continue
            cl = x["Close"] if isinstance(x.columns, pd.MultiIndex) else x[["Close"]].rename(columns={"Close": need[i]})
            frames.append(cl)
            time.sleep(2)
        if frames:
            new = pd.concat(frames, axis=1)
            new.index = pd.DatetimeIndex(new.index).tz_localize(None).normalize()
            have = new if have.empty else have.join(new, how="outer")
            have = have.loc[:, ~have.columns.duplicated()]
            have.to_parquet(p)
    return have


FX_PAIRS = ["EURUSD=X", "GBPUSD=X", "CADUSD=X", "CHFUSD=X", "SEKUSD=X", "NOKUSD=X", "DKKUSD=X"]


def fx_table(fetch_date: str, offline: bool) -> pd.DataFrame:
    """USD per unit of each currency (Yahoo daily closes), cached per fetch date."""
    p = C.CACHE / f"yf_fx_{pd.Timestamp(fetch_date):%Y%m%d}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    if offline:
        q = sorted(C.CACHE.glob("yf_fx_*.parquet"))
        return pd.read_parquet(q[-1])
    import yfinance as yf
    fx = yf.download(FX_PAIRS, start="2025-06-01", end=str((pd.Timestamp(fetch_date) + pd.Timedelta(days=1)).date()),
                     auto_adjust=False, progress=False)["Close"]
    fx.index = pd.DatetimeIndex(fx.index).tz_localize(None).normalize()
    fx.to_parquet(p)
    return fx


def usd_per(ccy: str, date, fx: pd.DataFrame) -> float:
    ccy = {"GBX": "GBP", "GBp": "GBP"}.get(str(ccy), str(ccy))
    if ccy == "USD":
        return 1.0
    col = f"{ccy}USD=X"
    if col not in fx:
        return np.nan
    s = fx.loc[:pd.Timestamp(date), col].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


# ---------------------------------------------------------------------------------------------- per date
def holdings_at(date: str, rawdir: Path, offline: bool) -> dict:
    out = {}
    for f in ("CSCA", "IMEA", "WSML", "SWDA"):
        p = F.ch_path(rawdir, f, date)
        if not p.exists():
            if offline:
                raise FileNotFoundError(p)
            p = F.fetch_ch(f, date, rawdir)
        out[f] = F.parse_ch(p, f)
    j = out["IMEA"].merge(out["SWDA"][["isin", "price"]], on="isin", suffixes=("", "_usd"))
    out["usd_per_eur"] = float((j.price_usd / j.price).median())
    return out


def in_market(df: pd.DataFrame, mkt: str) -> pd.Series:
    return (df.location == "Canada") if mkt == "CA" else df.location.isin(B.EU_LOC)


def table_at(mkt: str, date: str, H: dict, fx: pd.DataFrame, ca_close: pd.DataFrame | None) -> tuple:
    """Company table for one market on a holdings date (prices = holdings prices; non-members at the reference
    date of their line table)."""
    std_h = H[STD_FUND[mkt]]
    w = H["WSML"]
    sc_h = w[in_market(w, mkt)]
    other = set(H["SWDA"].loc[~in_market(H["SWDA"], mkt), "isin"]) | set(w.loc[~in_market(w, mkt), "isin"])
    d = pd.Timestamp(date)
    if mkt == "CA":
        pend = B.panel("CA")["D"][-1]
        lines = B.ca_lines_r2(d) if d <= pend else B.ca_lines_live(d, ca_close)
        m = std_h.merge(lines[["symbol", "px"]], left_on="ticker", right_on="symbol")
        usd_cad = float((m.price / m.px).median())
        T, LL = B.company_table("CA", std_h, sc_h, lines, {"CSCA": 1.0, "WSML": 1.0}, lambda r: usd_cad,
                                min(d, pd.Timestamp("2026-06-24")), f"ca_{d:%Y%m%d}", other)
        meta = dict(usd_per_cad=usd_cad, lines_source="R2 panel" if d <= pend else "TSX shadow 9 Oct screen + Yahoo closes")
    else:
        pend = B.panel("EU")["D"][-1]
        ref = min(d, pend)
        lines = B.eu_lines_r2(ref)
        ue_ref = float(fx.loc[:ref, "EURUSD=X"].dropna().iloc[-1])
        T, LL = B.company_table("EU", std_h, sc_h, lines, {"IMEA": H["usd_per_eur"], "WSML": 1.0},
                                lambda r: r.eur_per * ue_ref if np.isfinite(getattr(r, "eur_per", np.nan)) else np.nan,
                                min(d, pd.Timestamp("2026-06-24")), f"eu_{d:%Y%m%d}", other)
        meta = dict(usd_per_eur=H["usd_per_eur"], usd_per_eur_ref=ue_ref, lines_source=f"R2 panel {ref.date()}")
    return T, LL, std_h, sc_h, lines, meta


FUNDLIKE = (r"\bETF\b|UCITS|ISHARES|XTRACKERS|X-TRACK|X TRACK|LYXOR|AMUNDI|VANGUARD|SPDR|INVESCO MARKETS|POWERSHARES|"
            r"\bFUND\b|SICAV|ICAV|INV(ESTMENT)? TRUST|\bTRUST\b|ROLINCO|HBM HEALTHCARE|\bETC\b|\bETN\b|WISDOMTREE|"
            r"JPMORGAN ETFS|HSBC ETFS|UBS \(IRL\)|UBS ETF|DB X|STATE STREET|SSGA|GLOBAL X|FRANKLIN|VANECK|BLACKROCK|"
            r"\bTR$|INV TR|\bINVESTMENT CO|\bCAPITAL TR|PERSHING SQUARE")
DM_EU3 = {"GBR", "FRA", "DEU", "CHE", "NLD", "SWE", "ITA", "ESP", "DNK", "BEL", "FIN", "NOR", "AUT", "IRL", "PRT"}


def nonelig(T: pd.DataFrame, mkt: str) -> pd.Series:
    """Reason a non-member cannot be a candidate in this market (empty = may be eligible)."""
    r = pd.Series("", index=T.index)
    non = T.seg == "NON"
    r[non & T.other_country] = "MSCI holds it in another country (SWDA/WSML)"
    if mkt == "CA":
        r[non & (r == "") & ~T.main_isin.astype(str).str.startswith("CA")] = "not Canada-incorporated (App III)"
    else:
        r[non & (r == "") & ~T.excntry.astype(str).isin(DM_EU3)] = "listed outside DM Europe"
        r[non & (r == "") & ~T.tpci.astype(str).isin(["0", "1"])] = "not an ordinary/preferred share (Compustat tpci)"
        r[non & (r == "") & T.name.astype(str).str.upper().str.contains(FUNDLIKE, regex=True)] = "fund / ETF / trust"
    # the same company held by the funds under a new ISIN (ISIN change, e.g. Fresenius, Telecom Italia)
    from score import same_company
    mem = T.loc[T.seg != "NON", "name"].astype(str).tolist()
    for i in T.index[non & (r == "") & (T.full_cap_usd > 2e9)]:
        if any(same_company(T.at[i, "name"], n) >= 0.85 for n in mem):
            r[i] = "duplicate of a member (ISIN change)"
    return r


def epi_check(P: pd.DataFrame, mkt: str, asof, args, c, sig_c, h) -> pd.DataFrame:
    """Extreme Price Increase screen (gimi §2.3.6.3) for Small Cap / non-member candidates with p >= 0.05 and
    FIF < 0.75: excess return over the market (URTH) on each Mon-Fri window ending at the as-of date (Yahoo)."""
    cand = P[(P.seg != "STD") & (P.p_change >= 0.05) & (P.fif < C.EPI_FIF_EXEMPT)]
    if cand.empty or args.replay is not None:
        P["epi_checked"] = False
        return P
    syms = {i: (r.yf if isinstance(getattr(r, "yf", None), str) and r.yf else yf_symbol(r.main_ticker, r.exchange))
            for i, r in cand.iterrows()}
    px = yf_close(list(set(syms.values()) | {"URTH"}), str((asof - pd.Timedelta(days=380)).date()),
                  str((asof + pd.Timedelta(days=1)).date()), f"epi_{args.fetch_date}", args.no_fetch)
    P["epi_checked"] = False
    for i, sym in syms.items():
        if not sym or sym not in px or "URTH" not in px:
            continue
        s = px[[sym, "URTH"]].dropna()
        s = s[s.index <= asof]
        breach = False
        for n, thr in C.EPI:
            if len(s) > n:
                ex = (s[sym].iloc[-1] / s[sym].iloc[-1 - n]) - (s["URTH"].iloc[-1] / s["URTH"].iloc[-1 - n])
                breach |= ex > thr
        P.loc[i, "epi_breach"], P.loc[i, "epi_checked"] = breach, True
    if P.epi_breach.any():
        P = M.assign(P, c, sig_c, h, mkt)
    return P


def refresh_eu_non(P: pd.DataFrame, asof, args, c, sig_c, h) -> pd.DataFrame:
    """European non-members with p >= 0.05 carry R2 (30 Jun 2026) caps: look them up on Yahoo by ISIN, move the cap
    by the Yahoo price change 30 Jun -> as-of, and treat a name with no Yahoo listing / no recent price as
    probably delisted (M&A) (p x 0.1)."""
    import yfinance as yf
    cand = P[(P.seg == "NON") & (P.p_change >= 0.05)]
    if cand.empty:
        return P
    mp = C.CACHE / "yf_isin_map.json"
    m = json.loads(mp.read_text()) if mp.exists() else {}
    for i, r in cand.iterrows():
        isin = r.main_isin
        if isin not in m and not args.no_fetch:
            try:
                q = [x for x in yf.Search(isin, max_results=5).quotes if x.get("quoteType") == "EQUITY"]
                m[isin] = q[0]["symbol"] if q else None
            except Exception:  # noqa: BLE001
                m[isin] = None
            time.sleep(1.0)
    mp.write_text(json.dumps(m, indent=1))
    syms = {i: m.get(r.main_isin) for i, r in cand.iterrows()}
    px = yf_close([s for s in syms.values() if s], "2026-06-20", str((asof + pd.Timedelta(days=1)).date()),
                  f"eunon_{args.fetch_date}", args.no_fetch)
    for i, s in syms.items():
        P.loc[i, "yf"] = s
        x = px[s].dropna() if s and s in px else pd.Series(dtype=float)
        x = x[x.index <= asof]
        if x.empty or x.index[-1] < asof - pd.Timedelta(days=10):
            P.loc[i, "flags"] = (str(P.loc[i, "flags"] or "") + ";no_recent_yahoo_price(delisted?)").strip(";")
            P.loc[i, "nonelig"] = "no Yahoo listing / no recent price (delisted or taken over?)"
            continue
        ref = x[x.index <= pd.Timestamp("2026-06-30")]
        if len(ref):
            P.loc[i, "full_cap_usd"] *= float(x.iloc[-1] / ref.iloc[-1])
            P.loc[i, "float_usd"] *= float(x.iloc[-1] / ref.iloc[-1])
    return M.assign(P, c, sig_c, h, "EU")


US_CHECK = {"CA": "EWC", "EU": "EFA"}       # near-full-replication US funds of MSCI Canada / MSCI EAFE


def _tk(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(r"[ .\-/]", "", regex=True)


def us_crosscheck(T: pd.DataFrame, mkt: str, asof, std_h: pd.DataFrame, rawdir: Path, offline: bool) -> pd.DataFrame:
    """Cross-check the sampled UCITS Standard fund (CSCA / IMEA) against EWC / EFA on the same holdings date
    (added 2026-10-13, before any outcome). Two effects:
      1. float: a member the UCITS fund under-holds gets an understated fund-implied float (Redeia: EFA holds 4.5x
         IMEA's relative position). us_rel = (US qty / UCITS qty) / median over matched lines. The float, FIF and
         index shares are scaled by sqrt(us_rel) when the funds roughly agree (0.8-1.25: average the two), by us_rel
         when the UCITS fund clearly under-holds (> 1.25, capped at 10), and left alone when the US fund does (< 0.8).
      2. membership: a line the US fund holds but the UCITS fund does not is a Standard member (Telecom Italia,
         IT0005712671, dropped by IMEA/SWDA in Sep 2026 but held by EFA on 9 Oct), so it cannot be an addition.
    Skipped (us_rel = 1) when no US file dated `asof` is in raw/ (the US endpoint serves the latest file only)."""
    T["us_rel"], T["us_rel_up"], T["us_member"] = np.nan, 1.0, False
    fund = US_CHECK[mkt]
    p = rawdir / f"{fund}_us_asof{pd.Timestamp(asof):%Y%m%d}.csv"
    if not p.exists() and not offline:
        got, msg = F.fetch_us(fund, rawdir)
        log(f"{mkt}: fetched {fund}: {msg}")
    if not p.exists():
        # the US endpoint serves the latest file only: fall back to the newest one dated within 10 days before asof
        a = pd.Timestamp(asof)
        older = [q for q in C.HERE.glob(f"raw/*/{fund}_us_asof*.csv")
                 if a - pd.Timedelta(days=10) <= pd.Timestamp(q.stem[-8:]) <= a]
        if older:
            p = max(older, key=lambda q: q.stem[-8:])
            log(f"{mkt}: no {fund} file dated {a.date()}; using {p.relative_to(C.HERE)}")
    if not p.exists():
        log(f"{mkt}: no {fund} file dated within 10 days before {pd.Timestamp(asof).date()}; US cross-check skipped")
        T.attrs["us_check"] = f"skipped (no {fund} file dated within 10 days before {pd.Timestamp(asof).date()})"
        return T
    u = F.parse_us(p)
    u = u[u.Location.isin(B.EU_LOC)] if mkt == "EU" else u[u.Location == "Canada"]
    u = u.assign(tk=_tk(u.Ticker))
    h = std_h[in_market(std_h, mkt)].assign(tk=lambda d: _tk(d.ticker))
    j = h.merge(u[["tk", "Location", "Quantity"]], left_on=["tk", "location"], right_on=["tk", "Location"])
    j = j[(j.shares > 0) & (j.Quantity > 0)]
    j["rel"] = (j.Quantity / j.shares) / float((j.Quantity / j.shares).median())
    rel = j.drop_duplicates("isin").set_index("isin").rel
    T["us_rel"] = T.main_isin.map(rel)
    r = T.us_rel
    up = pd.Series(np.select([r > 1.25, r >= 0.8], [r.clip(upper=10.0), np.sqrt(r)], 1.0), index=T.index)
    up = up.fillna(1.0).where(T.seg == "STD", 1.0)
    T["us_rel_up"] = up
    T["float_usd"] = T.float_usd * up
    T["fif"] = (T.fif * up).clip(upper=1.0)
    # US-only lines -> members the UCITS fund skips; match non-members by name
    from score import same_company
    only = u[~u.tk.isin(h.tk) & (u["Weight (%)"].fillna(0) > 0)]
    cand = T.index[T.seg != "STD"]
    for nm in only.Name.astype(str):
        for i in cand:
            if same_company(nm, str(T.at[i, "name"])) >= 0.85:
                T.at[i, "us_member"] = True
    n_up = int((up > 1.25).sum())
    T.attrs["us_check"] = (f"{fund} {p.name}: {len(j)} lines matched; {n_up} members with UCITS under-holding > 25% "
                           f"(float raised); US-only members matched to non-members: "
                           f"{', '.join(T.loc[T.us_member, 'name'].astype(str)) or 'none'}")
    log(f"{mkt}: {T.attrs['us_check']}")
    return T


def known_excluded(P: pd.DataFrame) -> pd.DataFrame:
    """Hand-kept list of names MSCI has excluded for reasons we cannot see (known_excluded.csv: isin, p_cap,
    reason, source). Live runs only: the replay must not use later knowledge."""
    k = pd.read_csv(C.HERE / "known_excluded.csv", dtype=str)
    for r in k.itertuples():
        m = P.main_isin.astype(str) == r.isin
        if m.any():
            P.loc[m, "p_change"] = np.minimum(P.loc[m, "p_change"], float(r.p_cap))
            P.loc[m, "why"] = P.loc[m, "why"].astype(str) + f"; {r.reason} [{r.source}] -> p capped at {r.p_cap}"
    return P


def run(args) -> dict:
    replay = args.replay is not None
    if replay:
        rev = args.replay
        cur = C.PREV[rev]
        prev_key = {"2026-05": "2026-02", "2026-08": "2026-05"}[rev]
        asof = pd.Timestamp(cur["cutoff"])
        h = 0                                     # prices are the actual cutoff prices
        exp_cut = asof
        trade = pd.Timestamp(cur["trade"])
        liq_end = pd.Timestamp(asof.year, asof.month, 1) - pd.Timedelta(days=1)   # end of the previous month (Mar)
    else:
        prev_key = "2026-08"
        asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(args.fetch_date) - pd.offsets.BDay(1)
        trade = C.TRADE
        exp_cut = max(C.CUTOFF_LIKELY[1], asof)
        h = int(np.busday_count(asof.date(), exp_cut.date()))
        liq_end = min(C.LIQUIDITY_CUTOFF, pd.Timestamp("2026-06-30"))
    prev = C.PREV[prev_key]
    rawdir = F.rawdir(args.fetch_date)
    fx = fx_table(args.fetch_date, args.no_fetch)
    # ---- Canadian Yahoo closes (TSX shadow cache to 9 Oct + own refresh after it)
    tsx_cache = pd.read_parquet(C.TSX_SHADOW / "cache" / "yf_hist_20261009.parquet")["Close"]
    ca_close = tsx_cache
    if not replay and asof > tsx_cache.index.max():
        S = pd.read_csv(C.TSX_SHADOW / "runs/2026-12/2026-10-09_H-10w/screen_all.csv")
        fresh = yf_close(S.yf.dropna().tolist(), "2026-09-01", str((asof + pd.Timedelta(days=1)).date()),
                         f"ca_{args.fetch_date}", args.no_fetch)
        ca_close = pd.concat([tsx_cache, fresh[fresh.index > tsx_cache.index.max()]]).sort_index()
    log(f"as-of {asof.date()}, previous review {prev_key} (cutoff {prev['cutoff']}), h = {h} sessions to {exp_cut.date()}")
    Ha = holdings_at(str(asof.date()), rawdir, args.no_fetch)
    Hc = holdings_at(prev["cutoff"], rawdir, args.no_fetch)
    pre = {f: F.parse_ch(F.fetch_ch(f, prev["pre"], rawdir), f) for f in ("CSCA", "IMEA", "WSML")}
    post = {f: F.parse_ch(F.fetch_ch(f, prev["post"], rawdir), f) for f in ("CSCA", "IMEA", "WSML")}
    log("holdings loaded")
    results, metas = {}, {}
    for mkt in ("CA", "EU"):
        T, LL, std_h, sc_h, lines, meta = table_at(mkt, str(asof.date()), Ha, fx, ca_close)
        Tc, LLc, std_c, sc_c, lines_c, meta_c = table_at(mkt, prev["cutoff"], Hc, fx, ca_close)
        T, km = B.enrich(T, mkt, std_h, sc_h, lines, liq_end, min(asof, B.panel(mkt)["D"][-1]), trade - pd.DateOffset(months=3))
        Tc, kmc = B.enrich(Tc, mkt, std_c, sc_c, lines_c, liq_end, min(pd.Timestamp(prev["cutoff"]), B.panel(mkt)["D"][-1]),
                           pd.Timestamp(prev["trade"]) - pd.DateOffset(months=3))
        sf = STD_FUND[mkt]
        std_pre, std_post = set(pre[sf]["isin"]), set(post[sf]["isin"])
        sc_post = set(post["WSML"].loc[in_market(post["WSML"], mkt), "isin"])
        br = M.bracket(Tc, std_pre, std_post, sc_post, prev["gmsr"])
        f, nf = M.scale_factor(T, Tc, br["C"])
        drift = args.drift if args.drift is not None else (1.0 if replay else C.CUTOFF_DRIFT)
        c_now = br["C"] * f * drift
        # GMSR bounds on the as-of date: previous GMSR moved by the median DM (SWDA) price change
        sw = Ha["SWDA"].merge(Hc["SWDA"][["isin", "price"]], on="isin", suffixes=("", "_c"))
        dm = float(np.exp(np.median(np.log(sw.price / sw.price_c))))
        gmsr_now = prev["gmsr"] * dm
        c_now_b = float(np.clip(c_now, C.GMSR_RANGE[0] * gmsr_now, C.GMSR_RANGE[1] * gmsr_now))
        sig_c = C.SIGMA_CUTOFF[mkt] * (1.0 if br["consistent"] else 1.5)
        # known-excluded non-members and blocked Small Cap names from the previous review
        Tc_i = Tc.set_index("company")
        prev_cap = T.company.map(Tc_i.full_cap_usd)
        prev_seas = T.company.map(Tc_i.seasoned)
        T["prev_cap_usd"] = prev_cap
        T["known_excl"] = (T.seg == "NON") & (prev_cap >= br["C"]) & prev_seas.fillna(True).astype(bool)
        T["nonelig"] = nonelig(T, mkt)
        T["sc_blocked_prev"] = (T.seg == "SC") & (prev_cap >= C.BUF_HI * br["C"])
        T["epi_breach"] = False                                                   # filled for candidates below
        # float-test memory: Standard members already far below the existing-member float threshold at the previous
        # review and still kept (e.g. Redeia, Aug 2026): MSCI's float test evidently uses a larger float than the
        # fund-implied one (inclusion factor < FIF) -> the float test is not applied to them (I)
        prev_float = T.company.map(Tc_i.float_usd)
        T["float_exempt"] = (T.seg == "STD") & (prev_float < 0.75 * br["C"] / 3)
        if not replay:
            T = us_crosscheck(T, mkt, asof, std_h, rawdir, args.no_fetch)
        us_note = T.attrs.get("us_check", "not run (replay)")
        P = M.assign(T, c_now_b, sig_c, h, mkt)
        if mkt == "EU" and not replay:
            P = refresh_eu_non(P, asof, args, c_now_b, sig_c, h)
        P = epi_check(P, mkt, asof, args, c_now_b, sig_c, h)
        if not replay:
            um = P.get("us_member", pd.Series(False, index=P.index)).fillna(False).astype(bool)
            P.loc[um, "p_change"] = 0.0
            P.loc[um, "why"] = P.loc[um, "why"].astype(str) + f"; Standard member per {US_CHECK[mkt]} (UCITS fund does not hold it) -> no addition"
            P = known_excluded(P)
        P.loc[P.sc_blocked_prev, "p_change"] *= 0.3
        P.loc[P.sc_blocked_prev, "why"] += "; was >= 1.5 C at the previous review and stayed Small Cap (float/EPI/liquidity?)"
        P = M.noc_fill(P, c_now_b)
        P["market"] = mkt
        results[mkt] = P
        results[mkt + "_prev"] = Tc
        metas[mkt] = dict(bracket=br, scale=f, n_scale=nf, drift=drift, C_prev=br["C"], C_now=c_now, C_now_bounded=c_now_b,
                          gmsr_now=gmsr_now, dm_move=dm, sigma_c=sig_c, h=h, k=km, k_prev=kmc, meta=meta,
                          noc_target=P.attrs.get("noc_target"), exp_std=P.attrs.get("exp_std"), gap=P.attrs.get("gap"),
                          n_std=int((P.seg == "STD").sum()), n_sc=int((P.seg == "SC").sum()), us_check=us_note)
        log(f"{mkt}: C_prev {br['C'] / 1e9:.2f}bn [{br['lo'] / 1e9:.2f}, {br['hi'] / 1e9:.2f}] x {f:.3f} -> "
            f"{c_now_b / 1e9:.2f}bn; Std {metas[mkt]['n_std']}, expected changes {P.p_change[P.p_change > 0.05].sum():.1f}")
    return dict(results=results, metas=metas, asof=asof, h=h, exp_cut=exp_cut, rawdir=rawdir, replay=replay,
                Ha=Ha, prev_key=prev_key)


# ---------------------------------------------------------------------------------------------- outputs
CAT_LABEL = {"MIG_STD_TO_SC": "Standard -> Small Cap (migration)", "DELETE_FLOAT": "Standard -> out (float test)",
             "DELETE_LIQ": "Standard -> out (liquidity)", "MIG_SC_TO_STD": "Small Cap -> Standard (migration)",
             "ADD_STD": "outside IMI -> Standard (direct add)"}
SIDE = {"MIG_STD_TO_SC": -1, "DELETE_FLOAT": -1, "DELETE_LIQ": -1, "MIG_SC_TO_STD": 1, "ADD_STD": 1}


def country2(r) -> str:
    if isinstance(r.location, str) and r.location in B.EU_LOC:
        return B.EU_LOC[r.location]
    if r.market == "CA":
        return "CA"
    return B.ISO3.get(str(getattr(r, "fic", "")), str(getattr(r, "fic", ""))[:2])


def tidy(R: dict) -> pd.DataFrame:
    rows = []
    for mkt in ("CA", "EU"):
        P = R["results"][mkt].copy()
        m = R["metas"][mkt]
        P["C_usd"] = m["C_now_bounded"]
        k = m["k"]["k_std"]
        up = P["us_rel_up"].fillna(1.0) if "us_rel_up" in P else 1.0
        P["index_shares_std"] = np.where(P.seg == "STD", P.main_h / k * up, P.fif * P.main_so)
        rows.append(P)
    A = pd.concat(rows, ignore_index=True)
    A["country"] = [country2(r) for r in A.itertuples()]
    A["margin_pct"] = A.full_cap_usd / A.edge - 1
    A["side"] = A.category.map(SIDE).fillna(0).astype(int)
    A["usd_per_local"] = [usd_per(c, R["asof"], R["fx"]) for c in A.mkt_ccy]
    A["price_local"] = A.main_px_usd / A.usd_per_local
    A["yf_sym"] = [r.yf if isinstance(getattr(r, "yf", None), str) and r.yf else yf_symbol(r.main_ticker, r.exchange)
                   for r in A.itertuples()]
    A["asof"] = R["asof"].date()
    A["review"] = C.REVIEW if not R["replay"] else R["review"]
    return A


KEEP = ["asof", "review", "market", "country", "category", "risk", "side", "p_change", "name", "main_ticker", "yf_sym",
        "main_isin", "seg", "full_cap_usd", "C_usd", "edge", "margin_pct", "sigma", "float_usd", "float_thr", "fif",
        "fif_src", "atvr12", "atvr3", "fot3", "sig_rel", "seasoned", "known_excl", "nonelig", "float_exempt", "us_rel", "us_member",
        "sc_blocked_prev", "noc_fill", "epi_checked", "epi_breach", "prev_cap_usd", "index_shares_std", "main_h",
        "main_so", "price_local", "mkt_ccy", "main_px_usd", "exchange", "flags", "why", "isins", "company"]


def write_outputs(R: dict, args) -> list[Path]:
    A = tidy(R)
    stamp = f"{R['asof']:%Y-%m-%d}" + (f"_{args.tag}" if args.tag else "")
    outdir = C.REPLAY if R["replay"] else C.PRED
    keep = [c for c in KEEP if c in A]
    A = A.sort_values(["market", "p_change"], ascending=[True, False])
    files = []
    f_all = archive_old(outdir / f"pred_{stamp}_all.csv")
    A[keep].to_csv(f_all, index=False, float_format="%.6g")
    files.append(f_all)
    ch = A[(A.category.isin(SIDE)) & (A.p_change >= 0.05)].copy()
    ch = ch.sort_values(["market", "p_change"], ascending=[True, False])
    f_ch = archive_old(outdir / f"pred_{stamp}_changes.csv")
    ch[keep].to_csv(f_ch, index=False, float_format="%.6g")
    files.append(f_ch)
    pp = ch[ch.p_change >= 0.5][["asof", "market", "country", "category", "side", "name", "yf_sym", "main_isin",
                                 "price_local", "mkt_ccy", "main_px_usd", "p_change"]].copy()
    f_pp = archive_old(outdir / f"paper_portfolio_{stamp}.csv")
    pp.to_csv(f_pp, index=False, float_format="%.6g")
    files.append(f_pp)
    wl = A[(A.seg != "NON") & (A.p_change < 0.05) & (A.margin_pct.abs() < 0.20)].sort_values("margin_pct", key=abs)
    f_wl = archive_old(outdir / f"watchlist_{stamp}.csv")
    wl[keep].to_csv(f_wl, index=False, float_format="%.6g")
    files.append(f_wl)
    summ = dict(asof=str(R["asof"].date()), review=A.review.iloc[0], replay=R["replay"], tag=args.tag,
                fetch_date=args.fetch_date, run_utc=R["run_utc"], h_sessions=R["h"], expected_cutoff=str(R["exp_cut"].date()),
                previous_review=R["prev_key"], markets={})
    for mkt in ("CA", "EU"):
        m = R["metas"][mkt]
        x = ch[ch.market == mkt]
        summ["markets"][mkt] = dict(
            C_prev_usd=m["C_prev"], bracket=dict((k, v) for k, v in m["bracket"].items()), scale=m["scale"], drift=m["drift"],
            n_scale=m["n_scale"], C_now_usd=m["C_now"], C_now_bounded_usd=m["C_now_bounded"], gmsr_now=m["gmsr_now"],
            gmsr_range=[C.GMSR_RANGE[0] * m["gmsr_now"], C.GMSR_RANGE[1] * m["gmsr_now"]], dm_move=m["dm_move"],
            sigma_c=m["sigma_c"], k=m["k"], k_prev=m["k_prev"], meta=m["meta"], n_std=m["n_std"], n_sc=m["n_sc"],
            noc_target=m["noc_target"], expected_std=m["exp_std"],
            calls={c: int(((x.category == c) & (x.p_change >= 0.5)).sum()) for c in SIDE},
            expected={c: round(float(x.loc[x.category == c, "p_change"].sum()), 2) for c in SIDE})
    f_sum = archive_old(outdir / f"summary_{stamp}.json")
    f_sum.write_text(json.dumps(summ, indent=1, default=str))
    files.append(f_sum)
    R["A"], R["ch"], R["summ"] = A, ch, summ
    return files


def log_block(R: dict, args, files: list[Path]):
    rawdir = R["rawdir"]
    raw_files = sorted(p for p in rawdir.iterdir() if p.is_file())
    used = [p for p in raw_files if any(f"asof{d:%Y%m%d}" in p.name for d in R["dates_used"])]
    kind = "REPLAY (calibration, pseudo point-in-time)" if R["replay"] else "SHADOW (out-of-sample)"
    L = [f"\n## {'Replay ' + R['review'] if R['replay'] else 'Run'} {R['run_utc'][:19].replace('T', ' ')} UTC — "
         f"as-of {R['asof']:%Y-%m-%d}" + (f" ({args.tag})" if args.tag else "") + f" | {kind}", "",
         f"- Command: `python3 predict.py {' '.join(sys.argv[1:])}`; wall time {time.time() - T0:.0f}s.",
         f"- Holdings (iShares UCITS CH JSON, prices in fund currency) as of {R['asof']:%Y-%m-%d}; previous review "
         f"{R['prev_key']} (cutoff {C.PREV[R['prev_key']]['cutoff']}, pre {C.PREV[R['prev_key']]['pre']}, post "
         f"{C.PREV[R['prev_key']]['post']}). Expected price cutoff {R['exp_cut']:%Y-%m-%d} ({R['h']} sessions ahead).",
         f"- Raw inputs `raw/{args.fetch_date}/` used by this run:"]
    L += [f"  - `{p.name}` sha256 `{sha(p)}`" for p in used]
    for mkt in ("CA", "EU"):
        m = R["summ"]["markets"][mkt]
        br = m["bracket"]
        L += [f"- **{'MSCI Canada' if mkt == 'CA' else 'MSCI Europe (DM Europe, one market)'}**: previous cutoff "
              f"C = USD {br['C'] / 1e9:.2f}bn (bracket {br['lo'] / 1e9:.2f}–{br['hi'] / 1e9:.2f}bn; "
              f"{'consistent' if br['consistent'] else 'INCONSISTENT'}{'; ' + br['note'] if br['note'] else ''}) × "
              f"{m['scale']:.3f} (median price move of {m['n_scale']} Standard members near C) × drift {m['drift']:.2f} = "
              f"**USD {m['C_now_bounded_usd'] / 1e9:.2f}bn**; GMSR range {m['gmsr_range'][0] / 1e9:.2f}–"
              f"{m['gmsr_range'][1] / 1e9:.2f}bn; edges: Standard stays ≥ {C.BUF_LO * m['C_now_bounded_usd'] / 1e9:.2f}bn, "
              f"Small Cap up ≥ {C.BUF_HI * m['C_now_bounded_usd'] / 1e9:.2f}bn; existing float ≥ "
              f"{C.FLOAT_EXIST * m['C_now_bounded_usd'] / 1e9:.2f}bn. Standard companies {m['n_std']}, NOC target "
              f"{m['noc_target']}, expected Standard after review {m['expected_std']:.1f}.",
              f"  - previous review evidence: down {br['down']}; out {br['out']}; up {br['up']}; new {br['new']}; "
              f"smallest kept {br['smallest_kept']}; largest Small Cap kept {br['largest_sc_stay']}."]
    L += ["", "| Market | Category | Calls (p ≥ 0.5) | Σp (p ≥ 0.05) |", "|---|---|---|---|"]
    for mkt in ("CA", "EU"):
        m = R["summ"]["markets"][mkt]
        for c in SIDE:
            if m["expected"][c] > 0 or m["calls"][c] > 0:
                L.append(f"| {mkt} | {CAT_LABEL[c]} | {m['calls'][c]} | {m['expected'][c]} |")
    L += ["", "- Predicted changes with p ≥ 0.15 (p; full cap vs edge; why):"]
    ch = R["ch"]
    for r in ch[ch.p_change >= 0.15].itertuples():
        L.append(f"  - {r.market}/{r.country} {r.name} ({r.main_ticker}) {r.category} p={r.p_change:.2f}: {r.why}")
    L += ["", "| File | SHA-256 |", "|---|---|"]
    rel = (C.REPLAY if R["replay"] else C.PRED).name
    L += [f"| `{rel}/{p.name}` | `{sha(p)}` |" for p in files]
    L += ["", "- Known error sources: full caps from R2 shares (Jun 2026; Datastream company totals for unlisted "
          "classes; Canada MiG 31 Aug) x holdings prices, not MSCI's NOS on the cutoff date; FIF of members implied by "
          "iShares holdings (H/k), of others from Datastream/Worldscope proxies; market cutoffs reconstructed from the "
          "previous review (not published); ATVR from R2 volume to Jun 2026 (MSCI uses Sep 2026); WSML samples the "
          "Small Cap index; NOC dynamics, MSCI discretion, M&A between now and 30 Nov, the non-operating-company "
          "consultation (result due 16 Oct) and November country reclassifications are not modelled."]
    new = not C.LOG.exists() or C.LOG.stat().st_size == 0
    with open(C.LOG, "a") as fh:
        if new:
            fh.write("# Shadow log — MSCI November 2026 Index Review (MSCI Canada, MSCI Europe; paper only)\n\n"
                     "Append-only. Each `predict.py` / `score.py` run adds a block with SHA-256 hashes; never edit "
                     "earlier blocks. Runbook: README.md.\n")
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", help="holdings date YYYY-MM-DD (default: last business day before --fetch-date)")
    ap.add_argument("--fetch-date", default=dt.date.today().isoformat(), help="raw/<fetch-date>/ folder")
    ap.add_argument("--tag", default="", help="label, e.g. initial / cutoff-1019")
    ap.add_argument("--no-fetch", action="store_true", help="use files already in raw/<fetch-date>/ and caches")
    ap.add_argument("--replay", help="calibration replay of a past review, e.g. 2026-05 (outputs in replay/)")
    ap.add_argument("--drift", type=float, help="cutoff drift factor (default config.CUTOFF_DRIFT live, 1.0 replay)")
    args = ap.parse_args()
    run_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    R = run(args)
    R["run_utc"] = run_utc
    R["review"] = args.replay or C.REVIEW
    R["fx"] = fx_table(args.fetch_date, True)
    prev = C.PREV[R["prev_key"]]
    R["dates_used"] = [R["asof"]] + [pd.Timestamp(prev[k]) for k in ("cutoff", "pre", "post")]
    files = write_outputs(R, args)
    log_block(R, args, files)
    for p in files:
        log(f"  {p}  {sha(p)[:16]}")
    return R


if __name__ == "__main__":
    main()
