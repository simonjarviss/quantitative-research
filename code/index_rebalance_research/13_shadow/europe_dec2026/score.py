"""Score the logged December 2026 European shadow predictions against the announcements, and their paper P&L
(paper only; no orders).

Usage
  # 1. announcements: put each family's announced changes in a CSV and score every logged prediction
  #    columns: index (STOXX600|DAX|MDAX|SDAX|TECDAX|FTSE100|FTSE250), side (add|delete), name, isin (optional), ticker (optional)
  python3 score.py --family stoxx600 --actual actual/stoxx600_2026-12.csv      # STOXX release, Tue 1 Dec after the close
  python3 score.py --family ftse_uk  --actual actual/ftse_uk_2026-12.csv       # FTSE UK notice, Wed 2 Dec after the UK close
  python3 score.py --family dax      --actual actual/dax_2026-12.csv           # STOXX DAX release, Thu 3 Dec after 22:00 CET
  # 2. paper P&L of every logged prediction to the 18 Dec close and to T+5 (5th session after 18 Dec), vs the family index
  python3 score.py --family all --pnl
  # 3. backup truth after the trade: iShares holdings 17 -> 18 Dec (EXSA, EXS1, EXS3, EXS2, ISF; MIDD samples; no SDAX)
  python3 score.py --family all --from-holdings 2026-12-17 2026-12-18

How a family's actual moves are formed from index-level rows:
  STOXX600      add / delete.
  DAX family    per company, the highest index it leaves and joins: e.g. MDAX add + SDAX delete = SDAX->MDAX;
                TECDAX add / delete = TecDAX in / out.
  FTSE UK       FTSE100 add + FTSE250 delete = FTSE250->FTSE100; FTSE250 add alone = SmallCap->FTSE250; etc.
Matching to predictions: ISIN, then Yahoo/ticker root, then name similarity >= 0.8.
Each prediction file predictions/<family>/pred_<asof>_<tag>_changes.csv is scored on its deterministic calls
(`predicted`), on prob >= 0.5, and by Brier score over its listed rows. Results: scores/score_<family>_<date>.md/.csv,
appended to shadow_log.md with SHA-256 hashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import fetch as F
import lib
import universe as U

BENCH = {"stoxx600": "^STOXX", "dax": "^MDAXI", "ftse_uk": "^FTMC"}


def read_actual(p: Path) -> pd.DataFrame:
    x = pd.read_csv(p, dtype=str)
    x.columns = [c.strip().lower() for c in x.columns]
    for c in ("isin", "ticker", "name"):
        if c not in x:
            x[c] = ""
    x["index"] = x["index"].str.upper().str.replace("TECDAX", "TECDAX")
    x["side"] = x.side.str.lower().str.strip()
    return x.fillna("")


def company_moves(fam: str, a: pd.DataFrame) -> pd.DataFrame:
    """Index-level add/delete rows -> one move per company (family convention)."""
    a = a.copy()
    a["ck"] = np.where(a["isin"].str.len() == 12, a["isin"], a["name"].map(U.norm_name))
    out = []
    for ck, g in a.groupby("ck"):
        nm, isin, tk = g["name"].iloc[0], g["isin"].iloc[0], g["ticker"].iloc[0]
        s = set(zip(g["index"], g.side))
        if fam == "stoxx600":
            for idx, sd in s:
                out.append(dict(name=nm, isin=isin, ticker=tk, move="add" if sd == "add" else "delete"))
        elif fam == "dax":
            lvl = {"DAX": 3, "MDAX": 2, "SDAX": 1}
            frm = max([lvl[i] for i, sd in s if sd == "delete" and i in lvl], default=0)
            to = max([lvl[i] for i, sd in s if sd == "add" and i in lvl], default=0)
            nmv = {0: "none", 1: "SDAX", 2: "MDAX", 3: "DAX"}
            if to or frm:
                out.append(dict(name=nm, isin=isin, ticker=tk, move=f"{nmv[frm]}->{nmv[to]}"))
            for i, sd in s:
                if i == "TECDAX":
                    out.append(dict(name=nm, isin=isin, ticker=tk, move="TecDAX in" if sd == "add" else "TecDAX out"))
        else:
            a100 = ("FTSE100", "add") in s
            d100 = ("FTSE100", "delete") in s
            a250 = ("FTSE250", "add") in s
            d250 = ("FTSE250", "delete") in s
            mv = ("FTSE250->FTSE100" if a100 and d250 else "SmallCap->FTSE100" if a100 else
                  "FTSE100->FTSE250" if d100 and a250 else "FTSE100->SmallCap" if d100 else
                  "SmallCap->FTSE250" if a250 else "FTSE250->SmallCap" if d250 else None)
            if mv:
                out.append(dict(name=nm, isin=isin, ticker=tk, move=mv))
    return pd.DataFrame(out)


def match(pred: pd.DataFrame, act: pd.DataFrame) -> pd.Series:
    """For each predicted row: index of the matching actual company (same move), else NaN."""
    res = []
    for r in pred.itertuples():
        mv = r.category if "category" in pred and isinstance(getattr(r, "category", None), str) else r.move
        cand = act[act.move == mv]
        hit = np.nan
        for j, q in cand.iterrows():
            if isinstance(r.isin, str) and len(q["isin"]) == 12 and r.isin == q["isin"]:
                hit = j
                break
            root = str(r.symbol).split(".")[0].upper() if isinstance(r.symbol, str) else ""
            if q["ticker"] and root and q["ticker"].upper().replace(".", "-").rstrip("-") == root:
                hit = j
                break
            if U.sim(r.name, q["name"]) >= 0.8:
                hit = j
                break
        res.append(hit)
    return pd.Series(res, index=pred.index)


def score_family(fam: str, act: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    rows, L = [], []
    files = sorted((C.PRED / fam).glob("pred_*_changes.csv"))
    for f in files:
        p = pd.read_csv(f)
        if p.empty:
            continue
        if fam == "stoxx600":
            p["move"] = p.category
        p["hit"] = match(p, act)
        cats = sorted(set(p.move) | set(act.move))
        for c in cats + ["ALL"]:
            pp = p if c == "ALL" else p[p.move == c]
            aa = act if c == "ALL" else act[act.move == c]
            det = pp[pp.predicted.astype(bool)]
            p50 = pp[pp.prob >= 0.5]
            nd, nh = len(det), int(det.hit.notna().sum())
            n5, h5 = len(p50), int(p50.hit.notna().sum())
            na = len(aa)
            found = len(set(pp.hit.dropna()))
            rows.append(dict(file=f.name, move=c, n_actual=na, n_pred_det=nd, hits_det=nh,
                             precision_det=nh / nd if nd else np.nan, recall_det=nh / na if na else np.nan,
                             n_p50=n5, hits_p50=h5, precision_p50=h5 / n5 if n5 else np.nan,
                             recall_p50=h5 / na if na else np.nan, actual_found_in_rows=found,
                             brier=float(((pp.prob - pp.hit.notna().astype(float)) ** 2).mean()) if len(pp) else np.nan,
                             sum_prob=float(pp.prob.sum())))
    t = pd.DataFrame(rows)
    return t, L


def pnl_family(fam: str, out: Path) -> pd.DataFrame:
    """Paper P&L of each logged prediction's deterministic calls: prediction close -> 18 Dec close and -> T+5."""
    files = sorted((C.PRED / fam).glob("paper_portfolio_*.csv"))
    pp = pd.concat([pd.read_csv(f).assign(file=f.name) for f in files if f.stat().st_size > 5], ignore_index=True) \
        if files else pd.DataFrame()
    if pp.empty:
        return pp
    syms = sorted(set(pp.symbol.dropna())) + [BENCH[fam]]
    P = F.yf_prices(syms, out, start="2026-09-01", tag="score_prices")
    W, _ = lib.price_panel(P, "adjclose")
    res = []
    for r in pp.itertuples():
        s = r.symbol
        if s not in W:
            continue
        x = W[s].dropna()
        b = W[BENCH[fam]].dropna()
        d0 = pd.Timestamp(r.asof)
        for lab, d1 in [("to_T", C.TRADE), ("to_T+5", None)]:
            xs = x[x.index >= C.TRADE]
            if d1 is None:
                if len(xs) < 6:
                    continue
                d1 = xs.index[5]
            if x.index.max() < d1:
                continue
            p0 = x[x.index <= d0].iloc[-1]
            p1 = x[x.index <= d1].iloc[-1]
            b0, b1 = b[b.index <= d0].iloc[-1], b[b.index <= d1].iloc[-1]
            ret = p1 / p0 - 1
            br = b1 / b0 - 1
            res.append(dict(file=r.file, key=r.key, name=r.name, side=r.side, horizon=lab, d0=d0.date(), d1=d1.date(),
                            ret=ret, bench=br, signed=r.side * ret, signed_excess=r.side * (ret - br)))
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="all", choices=["all"] + C.FAMILIES)
    ap.add_argument("--actual", help="CSV of announced index-level changes (index, side, name, isin, ticker)")
    ap.add_argument("--from-holdings", nargs=2, metavar=("D0", "D1"), help="derive actual changes from iShares holdings")
    ap.add_argument("--pnl", action="store_true", help="paper P&L to the 18 Dec close and T+5")
    ap.add_argument("--fetch-date", default=dt.date.today().isoformat())
    a = ap.parse_args()
    run_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    out = F.rawdir(a.fetch_date)
    fams = C.FAMILIES if a.family == "all" else [a.family]
    block = [f"\n## Score {run_utc:%Y-%m-%d %H:%M:%S} UTC", "", f"- Command: `python3 score.py {' '.join(__import__('sys').argv[1:])}`."]
    files_out = []
    for fam in fams:
        act = None
        if a.actual:
            act = company_moves(fam, read_actual(Path(a.actual)))
            src = a.actual
        elif a.from_holdings:
            act = holdings_truth(fam, a.from_holdings, out)
            src = f"iShares holdings {a.from_holdings[0]} -> {a.from_holdings[1]}"
        if act is not None:
            t, _ = score_family(fam, act, src)
            p = C.SCORES / f"score_{fam}_{a.fetch_date}.csv"
            lib.archive_old(p)
            t.to_csv(p, index=False, float_format="%.4g")
            files_out.append(p)
            allr = t[t.move == "ALL"]
            for r in allr.itertuples():
                block.append(f"- {fam} `{r.file}`: deterministic precision {r.precision_det:.0%} recall {r.recall_det:.0%} "
                             f"({r.hits_det}/{r.n_pred_det} predicted, {r.n_actual} actual); prob>=0.5 precision "
                             f"{r.precision_p50:.0%} recall {r.recall_p50:.0%}; Brier {r.brier:.3f}; truth: {src}.")
        if a.pnl:
            pn = pnl_family(fam, out)
            if len(pn):
                p = C.SCORES / f"pnl_{fam}_{a.fetch_date}.csv"
                lib.archive_old(p)
                pn.to_csv(p, index=False, float_format="%.5g")
                files_out.append(p)
                g = pn.groupby(["file", "horizon", "side"]).agg(n=("key", "size"), mean_signed=("signed", "mean"),
                                                                 mean_excess=("signed_excess", "mean")).reset_index()
                for r in g.itertuples():
                    block.append(f"- P&L {fam} `{r.file}` {r.horizon} side {r.side:+d}: n {r.n}, mean signed return "
                                 f"{r.mean_signed:+.2%}, vs {BENCH[fam]} {r.mean_excess:+.2%} (gross, before costs).")
    block += ["", "| File | SHA-256 |", "|---|---|"] + [f"| `{p.relative_to(C.HERE)}` | `{lib.sha(p)}` |" for p in files_out]
    lib.append_log(block)
    print("\n".join(block))


def holdings_truth(fam: str, dates, out: Path) -> pd.DataFrame:
    """Adds/deletes from iShares holdings across the trade (asOfDate). SDAX has no fund; MIDD samples."""
    H = F._holdings_module(out / "holdings")
    funds = {"stoxx600": [("EXSA", "STOXX600")], "dax": [("EXS1", "DAX"), ("EXS3", "MDAX"), ("EXS2", "TECDAX")],
             "ftse_uk": [("ISF", "FTSE100"), ("MIDD", "FTSE250")]}[fam]
    rows = []
    for etf, idx in funds:
        hs = []
        for d in dates:
            p = out / "holdings" / etf / f"{etf}_asof{pd.Timestamp(d):%Y%m%d}.json"
            if not p.exists():
                url = H.ETF_URL["ch_json"].format(pid=H.ETF[etf]["pid"]) + f"&asOfDate={pd.Timestamp(d):%Y%m%d}"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(F.http_get(url))
            hs.append(F.parse_holdings(H, etf, p))
        a0, a1 = hs
        for r in a1[~a1["isin"].isin(set(a0["isin"]))].itertuples():
            rows.append(dict(index=idx, side="add", name=r.name, isin=r.isin, ticker=r.ticker))
        for r in a0[~a0["isin"].isin(set(a1["isin"]))].itertuples():
            rows.append(dict(index=idx, side="delete", name=r.name, isin=r.isin, ticker=r.ticker))
    return company_moves(fam, pd.DataFrame(rows, columns=["index", "side", "name", "isin", "ticker"]))


if __name__ == "__main__":
    main()
