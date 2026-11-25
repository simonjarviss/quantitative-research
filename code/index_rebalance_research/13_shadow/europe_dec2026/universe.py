"""Universes and Yahoo symbols for the three families (paper only).

STOXX Europe 600   lines of the latest public selection list (slpublic_sxxp): RIC, country, currency, exchange,
                   membership (Large/Mid/Small = member), official Rank (FINAL) at the last month-end cut-off.
                   Members are matched to EXSA holdings (ISIN, local ticker, index shares) by exchange and name.
DAX family         slpublic_mkdx (one ranking of all FSE Regulated Market companies: ISIN, RIC, membership DAXK /
                   MKDX / SDXK, rank, comments) and slpublic_tdxk (TecDAX technology ranking and membership).
FTSE UK            FTSE 100 = ISF holdings (EVRAZ, a frozen zero-value line, dropped); FTSE 250 = Wikipedia
                   "FTSE 250 Index" table (checked against MIDD, which samples); SmallCap candidates = R2 Datastream
                   FTSE All-Share list (3455) on 23 Jun 2026 minus today's FTSE 350, minus names deleted since.
Yahoo symbols      holdings tickers + exchange suffix; RIC heuristics; Yahoo search by ISIN or name as fallback.
                   The cumulative map lives in raw/symbol_map.csv (one row per key; method recorded).
"""
from __future__ import annotations

import difflib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import fetch as F

EXSA_SUFFIX = {"London Stock Exchange": ".L", "Nyse Euronext - Euronext Paris": ".PA", "Xetra": ".DE",
               "Deutsche Boerse Xetra": ".DE", "SIX Swiss Exchange": ".SW", "Nasdaq Omx Nordic": ".ST",
               "Borsa Italiana": ".MI", "Euronext Amsterdam": ".AS", "Bolsa De Madrid": ".MC",
               "Omx Nordic Exchange Copenhagen A/S": ".CO", "Oslo Bors Asa": ".OL",
               "Warsaw Stock Exchange/Equities/Main Market": ".WA", "Nasdaq Omx Helsinki Ltd.": ".HE",
               "Nyse Euronext - Euronext Brussels": ".BR", "Athens Exchange S.A. Cash Market": ".AT",
               "Wiener Boerse Ag": ".VI", "Irish Stock Exchange - All Market": ".IR",
               "Nyse Euronext - Euronext Lisbon": ".LS"}
LIST_SUFFIX = {"London SE": ".L", "NASDAQ Stockholm": ".ST", "Warsaw SE": ".WA", "XETRA": ".DE",
               "Deutsche Boerse": ".DE", "Euronext Paris": ".PA", "Oslo Bors": ".OL", "Six Swiss Exchange": ".SW",
               "Borsa Italiana": ".MI", "NASDAQ Helsinki": ".HE", "BME Exchange": ".MC", "ATHENS SE": ".AT",
               "Euronext Brussels": ".BR", "Euronext Amsterdam": ".AS", "NASDAQ Copenhagen": ".CO",
               "Wiener Boerse AG": ".VI", "Euronext Lisbon": ".LS", "Euronext Dublin": ".IR"}
RIC_SUFFIX = {".L": ".L", ".ST": ".ST", ".WA": ".WA", ".DE": ".DE", ".PA": ".PA", ".OL": ".OL", ".S": ".SW",
              ".MI": ".MI", ".HE": ".HE", ".MC": ".MC", ".AT": ".AT", ".BR": ".BR", ".AS": ".AS", ".CO": ".CO",
              ".VI": ".VI", ".LS": ".LS", ".I": ".IR", ".F": ".F"}
STOP = {"PLC", "AG", "SA", "NV", "N", "V", "ASA", "AB", "OYJ", "SE", "SPA", "S", "P", "A", "HOLDING", "HOLDINGS", "HLDG",
        "HLDGS", "GROUP", "GRP", "CORP", "CORPORATION", "LTD", "LIMITED", "THE", "CO", "INC", "COMPANY", "AS", "NAMEN",
        "REG", "REGISTERED", "CLASS", "B", "PREF", "PREFERENCE", "ORD", "SHS", "BEARER", "INH", "VZ", "AKT", "ADR",
        "TRUST", "FUND", "REIT", "INVESTMENT", "ET", "DE", "AND", "OF", "INTERNATIONAL", "INTL", "SPA.", "KGAA", "GMBH",
        "&", "BANK", "SOCIETE", "CIE"}


def norm_name(s) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper().replace("&", " AND "))
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks)


def sim(a, b) -> float:
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    j = len(ta & tb) / max(1, min(len(ta), len(tb)))
    pre = 1.0 if (a[:5] == b[:5] and len(a) >= 4) else 0.0
    return max(r, j, 0.85 * pre)


def exsa_symbol(ticker: str, exchange: str) -> str | None:
    suf = EXSA_SUFFIX.get(exchange)
    if suf is None:
        return None
    t = str(ticker).strip().rstrip(".")
    t = t.replace(" ", "-").replace("/", "-")
    if suf == ".L":
        t = t.replace(".", "-")
    return t + suf


def ric_symbol(ric: str) -> str | None:
    m = re.match(r"^(.*?)(\.[A-Z]+)$", str(ric))
    if not m:
        return None
    root, suf = m.group(1), m.group(2)
    ys = RIC_SUFFIX.get(suf)
    if ys is None:
        return None
    root = root.replace("_p", "")
    if ys in (".ST", ".CO") and re.search(r"[a-z]$", root):
        root = root[:-1] + "-" + root[-1].upper()
    elif ys == ".L" and root.endswith("a") and len(root) > 2:
        root = root[:-1] + "-A"
    else:
        root = re.sub(r"[a-z]+$", "", root)
    return root + ys


# ------------------------------------------------------------------------------------------ symbol map
MAP = C.RAW / "symbol_map.csv"


def load_map() -> pd.DataFrame:
    if MAP.exists():
        return pd.read_csv(MAP, dtype=str)
    return pd.DataFrame(columns=["key", "symbol", "method", "name", "added"])


def save_map(m: pd.DataFrame):
    m.drop_duplicates("key", keep="last").to_csv(MAP, index=False)


# ------------------------------------------------------------------------------------------ STOXX 600
RIC_ALIAS = {"ICAG.L": "IAG.L", "WDPP.BR": "WDP.BR", "OTEr.AT": "HTO.AT"}   # list RIC -> Yahoo, where name match fails


def stoxx_universe(sl: pd.DataFrame, exsa: pd.DataFrame, band=(380, 950)) -> pd.DataFrame:
    """One row per selection-list line: key = RIC. Members matched to EXSA by exchange suffix + name."""
    u = sl.copy()
    u["key"] = u["RIC"]
    u["member"] = u.member_code.isin(["Large", "Mid", "Small"])
    u["suffix"] = u.Exchange.map(LIST_SUFFIX)
    e = exsa.copy()
    e["symbol"] = [exsa_symbol(t, x) for t, x in zip(e.ticker, e.exchange)]
    e["suffix"] = e.exchange.map(EXSA_SUFFIX)
    # greedy best-score assignment within the same listing suffix (members only)
    pairs = []
    mem = u[u.member]
    for i, r in mem.iterrows():
        cand = e[e.suffix == r.suffix]
        for j, q in cand.iterrows():
            s = sim(r.Instrument_Name, q["name"])
            rs = RIC_ALIAS.get(r.RIC) or ric_symbol(r.RIC)
            if rs and rs.split(".")[0].replace("-", "") == str(q.symbol).split(".")[0].replace("-", ""):
                s += 0.5
            pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_u, used_e, match = set(), set(), {}
    for s, i, j in pairs:
        if i in used_u or j in used_e or s < 0.45:
            continue
        used_u.add(i); used_e.add(j); match[i] = (j, s)
    u["exsa_isin"] = None
    u["exsa_ticker"] = None
    u["exsa_score"] = np.nan
    for i, (j, s) in match.items():
        u.loc[i, ["exsa_isin", "exsa_ticker", "exsa_score"]] = [e.loc[j, "isin"], e.loc[j, "symbol"], s]
    u["in_band"] = u["rank"].between(*band) | u.member
    return u, e[~e.index.isin(used_e)]


# ------------------------------------------------------------------------------------------ DAX family
def dax_universe(main: pd.DataFrame, tec: pd.DataFrame, hold: dict) -> pd.DataFrame:
    u = main.copy()
    u["key"] = u["ISIN"].where(u["ISIN"].notna(), u["RIC"])
    lvl = {"DAXK": 3, "MKDX": 2, "SDXK": 1}
    u["lvl"] = u.member_code.map(lvl).fillna(0).astype(int)
    t = tec[["ISIN", "rank", "member_code", "Comment"]].rename(
        columns={"rank": "rank_tec", "member_code": "tec_code", "Comment": "tec_comment"})
    u = u.merge(t, on="ISIN", how="left")
    u["tech_listed"] = u.rank_tec.notna() | u.tec_code.notna()
    u["tec"] = u.tec_code.fillna("") == "TDXK"
    # holdings index shares and tickers
    for etf, idx in [("EXS1", "DAX"), ("EXS3", "MDAX"), ("EXS2", "TECDAX")]:
        h = hold[etf].set_index("isin")
        u[f"h_{etf}"] = u["ISIN"].map(h.shares)
        u[f"t_{etf}"] = u["ISIN"].map(h.ticker)
    tick = u.t_EXS1.fillna(u.t_EXS3).fillna(u.t_EXS2)
    u["symbol_hold"] = np.where(tick.notna(), tick.astype(str) + ".DE", None)
    return u


# ------------------------------------------------------------------------------------------ FTSE UK
def wiki_ftse(out: Path, page: str) -> pd.DataFrame:
    p = out / f"wikipedia_{page}.html"
    if not p.exists():
        p.write_bytes(F.http_get(f"https://en.wikipedia.org/api/rest_v1/page/html/{page}"))
    tabs = pd.read_html(io.BytesIO(p.read_bytes()))
    t = [x for x in tabs if x.shape[1] == 3 and x.shape[0] >= 95 and "Ticker" in [str(c) for c in x.columns]][0]
    t.columns = ["company", "ticker", "sector"]
    t["ticker"] = t.ticker.astype(str).str.upper().str.strip()
    return t


def lse_symbol(t: str) -> str:
    t = str(t).strip().upper().rstrip(".")
    return t.replace(".", "-").replace(" ", "-") + ".L"


def ftse_universe(out: Path, hold: dict, r2: dict, asof_lists=pd.Timestamp("2026-06-23")) -> pd.DataFrame:
    isf = hold["ISF"]
    isf = isf[(isf.name != "EVRAZ") & (isf.weight > 1e-5)].copy()
    w250 = wiki_ftse(out, "FTSE_250_Index")
    midd = hold["MIDD"]
    rows = []
    for r in isf.itertuples():
        rows.append(dict(key=r.isin, isin=r.isin, name=r.name, ticker=r.ticker, symbol=lse_symbol(r.ticker),
                         cur="FTSE100", src="ISF", ccy=r.mkt_ccy, hold_shares=r.shares, hold_price=r.price,
                         hold_weight=r.weight))
    mt = midd.assign(t=midd.ticker.str.upper().str.rstrip(".")).set_index("t")
    alias = {"HANA": "HAN", "BRGE": "BGEIT1"}
    for r in w250.itertuples():
        t = r.ticker.rstrip(".")
        m = mt.loc[alias.get(t, t)] if alias.get(t, t) in mt.index else None
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        rows.append(dict(key=(m["isin"] if m is not None else "LSE:" + t), isin=m["isin"] if m is not None else None,
                         name=r.company, ticker=t, symbol=lse_symbol(t), cur="FTSE250",
                         src="Wikipedia+MIDD" if m is not None else "Wikipedia",
                         ccy=m["mkt_ccy"] if m is not None else None,
                         hold_shares=m["shares"] if m is not None else np.nan,
                         hold_price=m["price"] if m is not None else np.nan,
                         hold_weight=m["weight"] if m is not None else np.nan))
    u = pd.DataFrame(rows)
    # SmallCap candidates from the R2 All-Share list (23 Jun 2026), not now in the FTSE 350
    L, S = r2["lists"], r2["shares"]
    d = asof_lists
    als = L[(L.startdate <= d) & (L.enddate >= d) & (L.list == 3455)].copy()
    s = S[S.marketdate == S.marketdate.max()][["infocode", "mktcap", "numshrs", "close_"]]
    als = als.merge(s, on="infocode", how="left")
    names350 = [norm_name(x) for x in u.name]
    isins350 = set(u["isin"].dropna())
    sc = []
    for r in als.itertuples():
        if r.isin in isins350:
            continue
        nm = norm_name(r.dsqtname)
        if any(difflib.SequenceMatcher(None, nm, x).ratio() > 0.86 for x in names350):
            continue
        sc.append(dict(key=r.isin or f"DS:{r.infocode}", isin=r.isin, name=r.dsqtname, ticker=None, symbol=None,
                       cur="SmallCap", src="R2 All-Share 2026-06-23", ccy=r.isocurrcode, hold_shares=np.nan,
                       hold_price=np.nan, hold_weight=np.nan, r2_mktcap_jun=r.mktcap, r2_numshrs_jun=r.numshrs,
                       infocode=r.infocode))
    # FTSE 350 rows: R2 Datastream shares (latest date held, units) by ISIN as the fallback when Yahoo has no count
    by_isin = als.dropna(subset=["isin"]).drop_duplicates("isin").set_index("isin")
    u["r2_numshrs_jun"] = u["isin"].map(by_isin.numshrs)
    u["r2_mktcap_jun"] = u["isin"].map(by_isin.mktcap)
    u["infocode"] = u["isin"].map(by_isin.infocode)
    u = pd.concat([u, pd.DataFrame(sc)], ignore_index=True)
    return u
