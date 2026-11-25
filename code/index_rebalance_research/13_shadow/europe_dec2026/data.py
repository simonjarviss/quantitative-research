"""Fetch-and-assemble step shared by predict.py, validate_june2026.py and score.py (paper only).

prepare(fetch_date) downloads (or reads from raw/<fetch_date>/ when already there):
  lists     STOXX 600 and DAX-family public selection lists, Mar 2026 -> latest (raw/<d>/lists/)
  holdings  EXSA, EXS1, EXS3, EXS2, ISF, MIDD (raw/<d>/holdings/)
  ftse      Wikipedia FTSE 100/250 tables, FTSE notices JSON, STOXX RSS, R2 FTSE lists (raw/<d>/)
  symbols   Yahoo symbols per line (raw/symbol_map.csv, cumulative; Yahoo search for the rest)
  yahoo     daily prices for every line in the ranking bands, FX and index levels (raw/<d>/yahoo/prices.jsonl);
            Yahoo info (shares in issue, float) for the FTSE universe and DAX lines without holdings
            (raw/<d>/yahoo/info.jsonl)
and returns a dict with the parsed frames.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import fetch as F
import universe as U

LIST_MONTHS = [f"2026{m:02d}" for m in range(3, 13)] + [f"2027{m:02d}" for m in range(1, 3)]


def latest_list(files: dict, sym: str, asof: pd.Timestamp) -> tuple[str, Path] | None:
    """Latest list of `sym` published on or before asof (file date = creation date)."""
    c = sorted((d, p) for (s, d), p in files.items() if s == sym and pd.Timestamp(d) <= asof)
    return c[-1] if c else None


def fill_members(sl: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """Review-month STOXX public lists (e.g. 1 Jun 2026) carry an empty "Index Membership" column. Take membership from
    the previous month's list (by RIC); members deleted since then are caught by the EXSA check in model_stoxx."""
    sl = sl.copy()
    sl["member_code"] = sl.RIC.map(dict(zip(prev.RIC, prev.member_code))).fillna("")
    sl["member_from"] = str(prev.list_date.iloc[0].date())
    return sl


def stoxx_list_with_members(files: dict, asof: pd.Timestamp):
    """Latest sxxp list on or before asof; if its membership column is empty, membership from the latest earlier list
    that has one."""
    d, p = latest_list(files, "sxxp", asof)
    sl = F.read_list(p)
    if not sl.member_code.isin(["Large", "Mid", "Small"]).any():
        for d2, p2 in sorted(((d2, p2) for (s, d2), p2 in files.items() if s == "sxxp" and d2 < d), reverse=True):
            prev = F.read_list(p2)
            if prev.member_code.isin(["Large", "Mid", "Small"]).any():
                print(f"[data] sxxp list {d} has no membership column; membership from the {d2} list", flush=True)
                return (d, p), fill_members(sl, prev)
    return (d, p), sl


def _months_to(asof: pd.Timestamp) -> list[str]:
    return [m for m in LIST_MONTHS if pd.Timestamp(m + "01") <= asof]


def get_all_lists(out: Path, asof: pd.Timestamp, offline: bool) -> dict:
    files = {}
    ld = out / "lists"
    ld.mkdir(exist_ok=True)
    for fam in ["stoxx600", "dax"]:
        if offline:
            for p in ld.glob("slpublic_*.csv"):
                m = re.match(r"slpublic_(\w+?)_(\d{8})\.csv", p.name)
                if m and m.group(1) in C.SL_SYMBOL[fam][1]:
                    files[(m.group(1), m.group(2))] = p
        else:
            months = _months_to(asof)[-12:]
            files.update(F.get_lists(fam, months, ld))
    # the 1 Jun 2026 STOXX list (June review list; a re-publication on 8 Jun also exists)
    p = ld / "slpublic_sxxp_20260601.csv"
    if not p.exists() and not offline:
        p.write_bytes(F.http_get("https://www.stoxx.com/document/Reports/STOXXSelectionList/2026/June/"
                                 "slpublic_sxxp_20260601.csv"))
    if p.exists():
        files[("sxxp", "20260601")] = p
    return files


def resolve_symbols(rows: pd.DataFrame, out: Path, offline: bool) -> pd.DataFrame:
    """rows: key, name, isin, suffix, sym_guess, sym_method. Adds `symbol` (validated later by price fetch).
    Lines without a guess are searched on Yahoo by ISIN, then by name, keeping a quote with the right suffix."""
    mp = U.load_map().set_index("key")
    rows = rows.copy()
    rows["symbol"] = rows.key.map(mp.symbol)
    rows["sym_method"] = rows.key.map(mp.method).fillna(rows.get("sym_method"))
    need = rows.symbol.isna() & rows.sym_guess.notna()
    rows.loc[need, "symbol"] = rows.loc[need, "sym_guess"]
    srch = rows[rows.symbol.isna()]
    if len(srch) and not offline:
        qs = [q for q in srch["isin"].dropna().tolist()] + [f"{n}" for n in srch.loc[srch["isin"].isna(), "name"]]
        res = F.yf_search(qs, out)
        for i, r in srch.iterrows():
            q = r["isin"] if isinstance(r["isin"], str) else r["name"]
            quotes = [x for x in res.get(q, []) if x.get("symbol")]
            pick = [x for x in quotes if str(x["symbol"]).endswith(r.suffix)] if isinstance(r.suffix, str) else quotes
            if pick:
                rows.loc[i, ["symbol", "sym_method"]] = [pick[0]["symbol"], f"yahoo search {q[:20]}"]
    return rows


def fix_failed(rows: pd.DataFrame, prices: pd.DataFrame, out: Path, offline: bool) -> pd.DataFrame:
    """Lines whose symbol returned no prices, or whose Yahoo name does not resemble the list name: Yahoo search
    by name (and ISIN), keep a quote with the right suffix."""
    rows = rows.copy()          # callers compare the result with their own frame to find new symbols
    meta = prices.groupby("symbol").agg(n=("close", "size"), ccy=("currency", "first"),
                                        yname=("yname", "first")).reset_index()
    ok = set(meta.loc[meta.n >= 20, "symbol"])
    yn = meta.set_index("symbol").yname
    # a heuristic symbol that exists but names a different company is treated as failed
    wrong = rows.sym_method.fillna("").str.contains("heuristic") & rows.symbol.isin(ok) & [
        U.sim(a, yn.get(s, "")) < 0.34 for a, s in zip(rows["name"], rows.symbol)]
    bad = rows[~rows.symbol.isin(ok) | wrong]
    if bad.empty or offline:
        return rows
    qs = []
    for r in bad.itertuples():
        if isinstance(r.isin, str):
            qs.append(r.isin)
        qs.append(re.sub(r"['\"]", "", str(r.name)))
    res = F.yf_search(qs, out)
    for i, r in bad.iterrows():
        for q in ([r["isin"]] if isinstance(r["isin"], str) else []) + [re.sub(r"['\"]", "", str(r["name"]))]:
            quotes = [x for x in res.get(q, []) if x.get("symbol")]
            pick = [x for x in quotes if isinstance(r.suffix, str) and str(x["symbol"]).endswith(r.suffix)]
            if pick and pick[0]["symbol"] != r.symbol:
                rows.loc[i, ["symbol", "sym_method"]] = [pick[0]["symbol"], f"yahoo search {q[:24]}"]
                break
    return rows


def prepare(fetch_date: str, asof: pd.Timestamp, offline: bool = False, price_budget_min: float = 120,
            info_budget_min: float = 40, calib: bool = True) -> dict:
    t0 = time.time()
    out = F.rawdir(fetch_date)
    files = get_all_lists(out, asof, offline)
    H = F.get_holdings(out, offline=offline)
    if not offline:
        if not list(out.glob("ftse_notices_UK_p*.json")):
            F.get_ftse_notices(out, since="2026-05-15")
        F.get_stoxx_rss(out)
    r2 = F.r2_ftse(out)
    # ---------------- STOXX
    sx, sl = stoxx_list_with_members(files, asof)
    su, _ = U.stoxx_universe(sl, H["EXSA"])
    lines = []
    for r in su.itertuples():
        if not (r.in_band or r.member or isinstance(r.exsa_ticker, str)):
            continue
        guess = r.exsa_ticker if isinstance(r.exsa_ticker, str) else (U.RIC_ALIAS.get(r.RIC) or U.ric_symbol(r.RIC))
        lines.append(dict(key=r.RIC, name=r.Instrument_Name, isin=r.exsa_isin, suffix=r.suffix, sym_guess=guess,
                          sym_method="EXSA ticker" if isinstance(r.exsa_ticker, str) else "RIC heuristic"))
    # EXSA members not on the list as members (added since the list) are keyed by ISIN
    e = H["EXSA"]
    for r in e.itertuples():
        lines.append(dict(key="ISIN:" + r.isin, name=r.name, isin=r.isin, suffix=U.EXSA_SUFFIX.get(r.exchange),
                          sym_guess=U.exsa_symbol(r.ticker, r.exchange), sym_method="EXSA ticker"))
    # calibration lists (June 2026 review): band lines of the Apr / May / Jun lists
    if calib:
        for d in ["20260401", "20260501", "20260601"]:
            if ("sxxp", d) in files:
                x = F.read_list(files[("sxxp", d)])
                x = x[x["rank"].between(380, 950) | x.member_code.isin(["Large", "Mid", "Small"])]
                for r in x.itertuples():
                    lines.append(dict(key=r.RIC, name=r.Instrument_Name, isin=None, suffix=U.LIST_SUFFIX.get(r.Exchange),
                                      sym_guess=U.RIC_ALIAS.get(r.RIC) or U.ric_symbol(r.RIC), sym_method="RIC heuristic"))
    # ---------------- DAX family
    dl = {}
    for s in ["mkdx", "tdxk"]:
        dl[s] = latest_list(files, s, asof)
    du = U.dax_universe(F.read_list(dl["mkdx"][1]), F.read_list(dl["tdxk"][1]), H)
    dax_lists = [F.read_list(dl["mkdx"][1])]
    if calib:
        for d in ["20260401", "20260505", "20260602", "20260701"]:
            if ("mkdx", d) in files:
                dax_lists.append(F.read_list(files[("mkdx", d)]))
    ric2isin = {}
    for x in dax_lists:
        ric2isin.update(dict(zip(x.RIC, x.ISIN)))
        for r in x[x["rank"].notna() | (x.member_code != "")].itertuples():
            hs = du.loc[du.ISIN == r.ISIN, "symbol_hold"]
            hs = hs.iloc[0] if len(hs) and isinstance(hs.iloc[0], str) else None
            lines.append(dict(key=r.ISIN, name=r.Instrument_Name, isin=r.ISIN, suffix=".DE", sym_guess=hs,
                              sym_method="iShares ticker" if hs else None))
    # German STOXX lines: RIC -> ISIN via the DAX lists, so they share the DAX symbol
    L = pd.DataFrame(lines)
    de = L.key.isin(ric2isin) & (L.suffix == ".DE")
    L.loc[de, "isin"] = L.loc[de, "key"].map(ric2isin)
    L.loc[de & (L.sym_method == "RIC heuristic"), "sym_guess"] = None
    # ---------------- FTSE UK
    fu = U.ftse_universe(out, H, r2)
    big_sc = fu[(fu.cur == "SmallCap") & (fu.r2_mktcap_jun.fillna(0) > 150e6)]
    fl = []
    for r in pd.concat([fu[fu.cur != "SmallCap"], big_sc]).itertuples():
        fl.append(dict(key=r.key, name=r.name, isin=r.isin, suffix=".L", sym_guess=r.symbol,
                       sym_method="LSE ticker" if isinstance(r.symbol, str) else None))
    if calib:   # June 2026 FTSE 100/250 members and large SmallCap names (R2 lists on 2 Jun 2026)
        Ls, S = r2["lists"], r2["shares"]
        d = pd.Timestamp("2026-06-02")
        j = Ls[(Ls.startdate <= d) & (Ls.enddate >= d) & (Ls.list == 3455)]
        s = S[S.marketdate == S[S.marketdate <= d].marketdate.max()].set_index("infocode").mktcap
        j = j[j.infocode.map(s).fillna(0) > 150e6]
        have = set(fu["isin"].dropna())
        for r in j.itertuples():
            if r.isin in have:
                continue
            fl.append(dict(key=r.isin or f"DS:{r.infocode}", name=r.dsqtname, isin=r.isin, suffix=".L", sym_guess=None,
                           sym_method=None))
    L = pd.concat([L, pd.DataFrame(fl)], ignore_index=True)
    L = L.drop_duplicates("key", keep="first").reset_index(drop=True)
    L = resolve_symbols(L, out, offline)
    print(f"[data] lines {len(L)}; with symbol {L.symbol.notna().sum()} ({time.time() - t0:.0f}s)", flush=True)
    # ---------------- Yahoo prices
    extra = list(C.FX_TICKERS.values()) + list(C.LEVEL_TICKERS.values())
    syms = extra + L.symbol.dropna().drop_duplicates().tolist()
    P = F.read_prices(out) if offline else F.yf_prices(syms, out, budget_min=price_budget_min)
    if not offline:
        L2 = fix_failed(L, P, out, offline)
        new = sorted(set(L2.symbol.dropna()) - set(L.symbol.dropna()))
        if new:
            P = F.yf_prices(new, out, budget_min=price_budget_min)
        L = L2
    # ---------------- Yahoo info (shares in issue) for the FTSE universe and DAX lines without holdings
    info_syms = L.loc[L.suffix.eq(".L") & L.key.isin(set(pd.DataFrame(fl).key)), "symbol"].dropna().tolist()
    info_syms += L.loc[L.key.isin(set(du.ISIN[du.symbol_hold.isna() & du["rank"].notna()])), "symbol"].dropna().tolist()
    # share / float sanity check of the largest names (top 10 STOXX 600 and DAX members by fund weight)
    top = H["EXSA"].nlargest(10, "weight")
    info_syms += [U.exsa_symbol(a, b) for a, b in zip(top.ticker, top.exchange)]
    info_syms += [f"{t}.DE" for t in H["EXS1"].nlargest(5, "weight").ticker]
    I = F.yf_info(info_syms, out, budget_min=info_budget_min) if not offline else _read_info(out)
    # persist the symbol map
    mp = U.load_map()
    add = L[L.symbol.notna()][["key", "symbol", "sym_method", "name"]].rename(columns={"sym_method": "method"})
    add["added"] = fetch_date
    U.save_map(pd.concat([mp, add], ignore_index=True))
    # derived file: rewrite only when the symbol resolution changed, keeping the earlier version in superseded/
    pl = out / "lines_symbols.csv"
    txt = L.to_csv(index=False)
    if not pl.exists() or pl.read_text() != txt:
        import lib
        lib.archive_old(pl)
        pl.write_text(txt)
    print(f"[data] prepared in {time.time() - t0:.0f}s; price symbols {P.symbol.nunique() if len(P) else 0}", flush=True)
    return dict(out=out, files=files, H=H, r2=r2, stoxx_list=sl, stoxx_list_date=sx[0], stoxx_u=su,
                dax_u=du, dax_list_date=dl["mkdx"][0], ftse_u=fu, lines=L, prices=P, info=I)


def _read_info(out: Path) -> pd.DataFrame:
    import json
    fj = out / "yahoo" / "info.jsonl"
    if not fj.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in fj.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows).drop_duplicates("symbol", keep="last")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--asof", default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    asof = pd.Timestamp(a.asof) if a.asof else pd.Timestamp.today().normalize()
    D = prepare(a.fetch_date, asof, offline=a.offline)
