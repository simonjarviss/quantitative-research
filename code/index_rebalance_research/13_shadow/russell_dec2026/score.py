"""Score a shadow prediction against FTSE Russell's lists and track its paper P&L versus IWM (paper only).

Usage
  # after the preliminary lists (Fri 13 Nov 2026, after the close): downloads the two PDFs if needed
  python3 score.py --asof 2026-10-09 --prelim-date 2026-11-13
  # same, with files saved by hand (PDF, CSV with a Symbol column, or one symbol per line)
  python3 score.py --asof 2026-10-30 --adds path/ru3000-additions-20261113.pdf --deletes path/ru3000-deletions-20261113.pdf
  # after the recon (holdings dated Mon 14 Dec or later): score all six categories incl. R1000<->R2000 migrations
  python3 score.py --asof 2026-10-30 --post-holdings 2026-12-15
  # paper P&L only, prediction close -> any later close
  python3 score.py --asof 2026-10-09 --pnl-to 2026-12-11

FTSE Russell public list format (checked on the June 2026 files):
  URL   https://www.lseg.com/content/dam/ftse-russell/en_us/documents/other/ru3000-additions-YYYYMMDD.pdf
        .../ru3000-deletions-YYYYMMDD.pdf (also rmicro-additions/-deletions). YYYYMMDD = list date: June 2026
        prelim 20260522, final 20260626; December 2026 prelim expected 20261113, updates 20261120/1127/1204,
        final 20261211. Linked from https://www.lseg.com/en/ftse-russell/russell-reconstitution
  PDF   "Russell 3000(R) Index - Additions" / "- Deletions"; columns Company (truncated to 24 chars), Symbol
        (class suffix with a dot, e.g. BH.A), Industry (ICB industry). R3000-level only: the public lists do not
        split R1000 / R2000, so migrations are scored from post-recon iShares holdings (--post-holdings).
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import fetch as F

IND = ("Technology|Financials|Health Care|Energy|Industrials|Basic Materials|Consumer Discretionary|"
       "Consumer Staples|Telecommunications|Utilities|Real Estate")
ROW = re.compile(r"^\s*(?P<company>\S.*?)\s{2,}(?P<symbol>[A-Z0-9][A-Z0-9.\-/]*)\s{2,}(?P<industry>" + IND + r")\s*$")


# ------------------------------------------------------------------ list parsing
def pdf_text(p: Path) -> str:
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True, check=True).stdout
    import pdfplumber
    with pdfplumber.open(p) as pdf:
        return "\n".join(pg.extract_text(layout=True) or "" for pg in pdf.pages)


def read_list(p: Path) -> pd.DataFrame:
    """FTSE Russell list -> DataFrame(company, symbol, industry, yf). Accepts the PDF, a CSV with a Symbol column,
    or a text file with one symbol per line."""
    p = Path(p)
    if p.suffix.lower() == ".pdf":
        rows = [m.groupdict() for m in map(ROW.match, pdf_text(p).splitlines()) if m]
        df = pd.DataFrame(rows, columns=["company", "symbol", "industry"])
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"ticker": "symbol", "name": "company"})
        for c in ["company", "industry"]:
            if c not in df:
                df[c] = None
    else:
        syms = [l.strip().split()[0] for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        df = pd.DataFrame({"company": None, "symbol": syms, "industry": None})
    df["yf"] = df.symbol.map(F.yf_symbol)
    if df.empty:
        raise SystemExit(f"no rows parsed from {p}")
    return df.drop_duplicates("yf")


def get_ftse_lists(date: str) -> tuple[Path, Path]:
    d = pd.Timestamp(date)
    out = C.RAW / f"ftse_{d:%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for side in ["additions", "deletions"]:
        p = out / f"ru3000-{side}-{d:%Y%m%d}.pdf"
        if not p.exists():
            url = C.FTSE_LIST_URL.format(kind="ru3000", side=side, yyyymmdd=f"{d:%Y%m%d}")
            try:
                r = F.http_get(url, tries=2)
            except RuntimeError as e:
                raise SystemExit(f"{e}\nDownload the lists by hand from https://www.lseg.com/en/ftse-russell/russell-reconstitution "
                                 f"and pass --adds/--deletes.")
            if not r.content.startswith(b"%PDF"):
                raise SystemExit(f"{url} did not return a PDF; download by hand and pass --adds/--deletes")
            p.write_bytes(r.content)
        paths.append(p)
    return paths[0], paths[1]


# ------------------------------------------------------------------ matching
def norm_name(s) -> str:
    s = re.sub(r"\(.*?\)|[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|holdings?|group|the|class|cl|a|b|c|adr|ads)\b", " ", s)
    return " ".join(s.split())[:18]


def match_to_companies(lst: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Map list rows to prediction companies by ticker (any share class), then by truncated name."""
    t2c = {t: cid for cid, ts in zip(A.cid, A.tickers) for t in str(ts).split(",")}
    lst = lst.copy()
    lst["cid"] = lst.yf.map(t2c)
    lst["how"] = np.where(lst.cid.notna(), "ticker", None)
    nm = A.assign(k=A.name.map(norm_name)).drop_duplicates("k").set_index("k").cid
    miss = lst.cid.isna() & lst.company.notna()
    lst.loc[miss, "cid"] = lst.loc[miss, "company"].map(norm_name).map(nm)
    lst.loc[miss & lst.cid.notna(), "how"] = "name"
    return lst


def prf(P: set, Aset: set, universe: set | None = None) -> dict:
    hit = len(P & Aset)
    d = dict(n_pred=len(P), n_actual=len(Aset), n_hit=hit,
             precision=hit / len(P) if P else np.nan, recall=hit / len(Aset) if Aset else np.nan)
    if universe is not None:
        vis = Aset & universe
        d["n_actual_in_universe"] = len(vis)
        d["recall_in_universe"] = len(P & vis) / len(vis) if vis else np.nan
    return d


# ------------------------------------------------------------------ paper P&L
def paper_pnl(pp: pd.DataFrame, asof: pd.Timestamp, to: pd.Timestamp | None):
    import yfinance as yf
    tick = sorted(set(pp.pv)) + ["IWM"]
    end = (to or pd.Timestamp(dt.date.today())) + pd.Timedelta(days=1)
    df = yf.download(tick, start=(asof - pd.Timedelta(days=7)).date(), end=end.date(), auto_adjust=False,
                     actions=False, threads=True, progress=False, group_by="column")
    adj, close = df["Adj Close"], df["Close"]
    adj.index = pd.to_datetime(adj.index).tz_localize(None).normalize() if adj.index.tz is not None else pd.to_datetime(adj.index)
    close.index = adj.index
    if asof not in adj.index:
        raise SystemExit(f"no prices on the prediction date {asof:%Y-%m-%d}")
    adj = adj.loc[:to] if to is not None else adj
    to_d = adj.index[adj["IWM"].notna()].max()
    a0 = adj.loc[asof]
    last = adj.ffill().loc[to_d]
    lastdate = adj.apply(lambda s: s.last_valid_index())
    r = last / a0 - 1
    pp = pp.copy()
    pp["ret"] = pp.pv.map(r)
    pp["iwm_ret"] = r["IWM"]
    pp["excess"] = pp.ret - pp.iwm_ret
    pp["side_pnl"] = pp.side * pp.excess
    pp["last_price_date"] = pp.pv.map(lastdate)
    pp["stale"] = pp.last_price_date < to_d
    return pp, to_d, float(r["IWM"])


def pnl_tables(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in C.CATS + ["BUY book (ADD_R2000+MIG_R1000_TO_R2000)", "SELL book (DEL_R2000+MIG_R2000_TO_R1000)"]:
        if c.startswith("BUY"):
            g = pp[pp.category.isin(["ADD_R2000", "MIG_R1000_TO_R2000"])]
        elif c.startswith("SELL"):
            g = pp[pp.category.isin(["DEL_R2000", "MIG_R2000_TO_R1000"])]
        else:
            g = pp[pp.category == c]
        if g.empty:
            continue
        g = g[g.ret.notna()]
        rows.append(dict(book=c, n=len(g), side=int(np.sign(g.side.mean())), mean_ret=g.ret.mean(),
                         mean_excess_vs_iwm=g.excess.mean(), median_excess=g.excess.median(),
                         side_pnl_mean=g.side_pnl.mean(), hit_rate=(g.side_pnl > 0).mean(), n_stale=int(g.stale.sum())))
    T = pd.DataFrame(rows)
    b = pp[pp.category.isin(["ADD_R2000", "MIG_R1000_TO_R2000"]) & pp.ret.notna()]
    s = pp[pp.category.isin(["DEL_R2000", "MIG_R2000_TO_R1000"]) & pp.ret.notna()]
    if len(b) and len(s):
        T = pd.concat([T, pd.DataFrame([dict(book="LONG-SHORT (BUY minus SELL, equal weight)", n=len(b) + len(s), side=0,
                                             mean_ret=b.ret.mean() - s.ret.mean(), mean_excess_vs_iwm=np.nan,
                                             median_excess=np.nan, side_pnl_mean=b.ret.mean() - s.ret.mean(),
                                             hit_rate=np.nan, n_stale=int(b.stale.sum() + s.stale.sum()))])],
                      ignore_index=True)
    return T


PCT_COLS = {"precision", "recall", "recall_in_universe", "mean_ret", "mean_excess_vs_iwm", "median_excess",
            "side_pnl_mean", "hit_rate"}


def md_table(df: pd.DataFrame) -> str:
    def f(c, v):
        if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            return ""
        if c in PCT_COLS:
            return f"{v:+.1%}" if c in {"mean_ret", "mean_excess_vs_iwm", "median_excess", "side_pnl_mean"} else f"{v:.1%}"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return f"{int(v)}"
        return str(v)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(c, v) for c, v in zip(cols, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", required=True, help="as-of date of the prediction to score")
    ap.add_argument("--tag", default="", help="run tag used by predict.py (files predictions/pred_<asof>_<tag>_all.csv)")
    ap.add_argument("--prelim-date", help="FTSE list date to download (e.g. 2026-11-13)")
    ap.add_argument("--adds", help="R3000 additions list (PDF / CSV / txt)")
    ap.add_argument("--deletes", help="R3000 deletions list (PDF / CSV / txt)")
    ap.add_argument("--post-holdings", help="raw/<date>/ with post-recon iShares holdings (fetched if missing)")
    ap.add_argument("--pnl-to", help="paper P&L end date (default: latest close)")
    ap.add_argument("--pnl-from", help="paper P&L start date (default: the prediction as-of close), e.g. the recon close "
                                       "2026-12-11 to measure the post-recon delete rebound")
    ap.add_argument("--no-pnl", action="store_true")
    args = ap.parse_args()
    run_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    asof = pd.Timestamp(args.asof)
    stem0 = f"{asof:%Y-%m-%d}" + (f"_{args.tag}" if args.tag else "")
    A = pd.read_csv(C.PRED / f"pred_{stem0}_all.csv", dtype={"cid": str})
    A["cid"] = A.cid.astype(str)
    universe = set(A.cid)
    label = []
    md = [f"# Score of the {stem0} prediction (run {run_utc:%Y-%m-%d %H:%M} UTC)", ""]
    acc_rows, detail = [], []

    # ---------------- FTSE preliminary / final R3000 lists
    adds_p, dels_p = (Path(args.adds) if args.adds else None), (Path(args.deletes) if args.deletes else None)
    if args.prelim_date and not (adds_p and dels_p):
        adds_p, dels_p = get_ftse_lists(args.prelim_date)
    if adds_p and dels_p:
        LA, LD = match_to_companies(read_list(adds_p), A), match_to_companies(read_list(dels_p), A)
        tag = args.prelim_date or adds_p.stem
        label.append(f"lists {tag}")
        actA, actD = set(LA.cid.dropna()), set(LD.cid.dropna())
        predA = set(A.loc[A.category.isin(["ADD_R2000", "ADD_R1000"]), "cid"])
        predD = set(A.loc[A.category.isin(["DEL_R2000", "DEL_R1000"]), "cid"])
        for name, P, S in [("R3000 adds (ADD_R2000 + ADD_R1000)", predA, actA), ("R3000 deletes (DEL_R2000 + DEL_R1000)", predD, actD)]:
            acc_rows.append(dict(source=f"FTSE list {tag}", group=name, **prf(P, S, universe)))
        for c, S in [("ADD_R2000", actA), ("ADD_R1000", actA), ("DEL_R2000", actD), ("DEL_R1000", actD)]:
            P = set(A.loc[A.category == c, "cid"])
            hit = len(P & S)
            acc_rows.append(dict(source=f"FTSE list {tag}", group=c, n_pred=len(P), n_actual=np.nan, n_hit=hit,
                                 precision=hit / len(P) if P else np.nan, recall=np.nan))
        # where did the misses sit?
        A_ix = A.set_index("cid")
        for side, Lx, Pset in [("add", LA, predA), ("delete", LD, predD)]:
            for r in Lx.itertuples():
                if r.cid is not None and r.cid in A_ix.index:
                    a = A_ix.loc[r.cid]
                    detail.append(dict(list=side, symbol=r.symbol, company=r.company, matched=r.how, predicted=r.cid in Pset,
                                       our_category=a.category, cur=a.cur, rank=a["rank"], p_hold=a.p_hold,
                                       excl_reason=a.excl_reason))
                else:
                    detail.append(dict(list=side, symbol=r.symbol, company=r.company, matched="not in our universe",
                                       predicted=False))
            for cid in Pset - set(Lx.cid.dropna()):
                a = A_ix.loc[cid]
                detail.append(dict(list=f"false {side}", symbol=a.pv, company=a["name"], matched="", predicted=True,
                                   our_category=a.category, cur=a.cur, rank=a["rank"], p_hold=a.p_hold))
        md += [f"FTSE lists: `{adds_p}` ({len(LA)} adds), `{dels_p}` ({len(LD)} deletes). "
               f"Unmatched list names (not in our {asof:%Y-%m-%d} universe): adds {int(LA.cid.isna().sum())}, deletes {int(LD.cid.isna().sum())}.", ""]

    # ---------------- post-recon holdings: all six categories
    if args.post_holdings:
        rd = C.RAW / args.post_holdings
        rd.mkdir(parents=True, exist_ok=True)
        if not list(rd.glob("IWM_holdings_asof*.csv")):
            F.fetch_holdings(rd)
        H, h_asof = F.load_holdings(rd)
        if min(h_asof.values()) <= C.RECON:
            print(f"WARNING: holdings dated {min(h_asof.values()):%Y-%m-%d} may still be pre-recon (recon close {C.RECON:%Y-%m-%d})")
        live = H[~H.residual]
        r1 = set(live.loc[live.fund == "IWB", "yf"])
        r2 = set(live.loc[live.fund == "IWM", "yf"]) - r1
        tk = A.tickers.astype(str).str.split(",")
        post = np.where(tk.map(lambda L: any(t in r1 for t in L)), "R1000",
                        np.where(tk.map(lambda L: any(t in r2 for t in L)), "R2000", "out"))
        A["post"] = post
        import predict as Pm
        A["actual"] = Pm.classify(A.cur, A.post)
        label.append(f"holdings {min(h_asof.values()):%Y-%m-%d}")
        for c in C.CATS:
            acc_rows.append(dict(source=f"iShares holdings {min(h_asof.values()):%Y-%m-%d}", group=c,
                                 **prf(set(A.loc[A.category == c, "cid"]), set(A.loc[A.actual == c, "cid"]), universe)))
        md += ["Actual post-recon membership from IWB/IWM holdings (companies absent from our as-of universe, e.g. later IPOs, "
               "are not counted).", ""]

    acc = pd.DataFrame(acc_rows)
    stem = f"score_{stem0}_{run_utc:%Y%m%dT%H%M}"
    files = []
    if len(acc):
        p = C.SCORES / f"{stem}_accuracy.csv"
        acc.to_csv(p, index=False)
        files.append(p)
        md += ["## Precision / recall", "", md_table(acc), ""]
    if detail:
        p = C.SCORES / f"{stem}_names.csv"
        pd.DataFrame(detail).to_csv(p, index=False)
        files.append(p)

    # ---------------- paper P&L
    if not args.no_pnl:
        pp = pd.read_csv(C.PRED / f"paper_portfolio_{stem0}.csv")
        to = pd.Timestamp(args.pnl_to) if args.pnl_to else None
        start = pd.Timestamp(args.pnl_from) if args.pnl_from else asof
        pp, to_d, iwm = paper_pnl(pp, start, to)
        T = pnl_tables(pp)
        p = C.SCORES / f"{stem}_pnl_names.csv"
        pp.to_csv(p, index=False, float_format="%.6g")
        files.append(p)
        p = C.SCORES / f"{stem}_pnl.csv"
        T.to_csv(p, index=False, float_format="%.6g")
        files.append(p)
        label.append(f"P&L {start:%Y-%m-%d} -> {to_d:%Y-%m-%d}")
        md += [f"## Paper P&L {start:%Y-%m-%d} close -> {to_d:%Y-%m-%d} close (total return, equal weight, gross; IWM {iwm:+.2%})", "",
               "side_pnl = side x (return - IWM): positive = the trade direction paid (long adds, short deletes). "
               "Names that stop trading are held at their last price (n_stale).", "", md_table(T), ""]
    p = C.SCORES / f"{stem}.md"
    p.write_text("\n".join(md) + "\n")
    files.insert(0, p)
    print("\n".join(md))

    L = [f"\n## Score {run_utc:%Y-%m-%d %H:%M:%S} UTC — prediction {stem0} ({'; '.join(label)})", ""]
    if len(acc):
        L += [md_table(acc[[c for c in ["source", "group", "n_pred", "n_actual", "n_hit", "precision", "recall"] if c in acc]]), ""]
    if not args.no_pnl:
        L += [md_table(T[["book", "n", "mean_ret", "mean_excess_vs_iwm", "side_pnl_mean", "hit_rate"]]), ""]
    L += ["| File | SHA-256 |", "|---|---|"] + [f"| `scores/{f.name}` | `{hashlib.sha256(f.read_bytes()).hexdigest()}` |" for f in files]
    with open(C.LOG, "a") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
