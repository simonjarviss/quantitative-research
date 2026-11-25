"""Score the MSCI November 2026 shadow predictions (paper only; public data; no accounts).

    python3 score.py --replay 2026-05                       # calibration replay vs the May 2026 holdings diffs
    python3 score.py --announced 2026-11-11                 # after ~23:00 CET on 11 Nov: MSCI public lists
    python3 score.py --holdings-diff 2026-11-27 2026-11-30  # backup / confirmation: iShares UCITS diffs over T
    python3 score.py --pnl-to 2026-11-30 [--pnl-label T]    # paper P&L of every logged Nov run to that close
    python3 score.py --pnl-to 2026-12-07 --pnl-label T+5

Actual changes:
  * MSCI public lists (msci/README, manifest §3): https://www.msci.com/eqb/gimi/stdindex/MSCI_Nov26_STPublicList.pdf
    and .../smallcap/MSCI_Nov26_SCPublicList.pdf, parsed with `pdftotext -layout` (tested on the stored Aug 2026
    lists: Canada 0 adds / 2 deletes; Germany 0 / 2; UK 1 / 1; Sweden 0 / 4; Small Cap Canada 8 / 6). Names only
    (no ISIN): matched to our companies by normalised name tokens. A Standard deletion that appears as a Small Cap
    addition is a Standard -> Small Cap migration; a Standard addition that is a Small Cap deletion is the reverse.
  * Holdings diffs: CSCA / IMEA (Standard) across the trade close, WSML for the destination segment.
Scores: precision / recall of calls (p >= 0.5) per market and side, expected count (sum of p) vs actual, Brier
score over every company with a prediction, and the misses. Paper P&L: side x (stock USD return - benchmark USD
return), benchmark EWC (Canada) / IEUR (Europe), from the as-of close of the run to the given close; calls
equal-weighted, and all p >= 0.10 rows p-weighted. Every run appends a block with SHA-256 hashes to shadow_log.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build as B
import config as C
import fetch as F

DEL_CATS = {"MIG_STD_TO_SC", "DELETE_FLOAT", "DELETE_LIQ"}
ADD_CATS = {"MIG_SC_TO_STD", "ADD_STD"}
STOP = {"INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PLC", "SA", "S", "A", "AG", "NV", "N", "V", "SE", "AB", "ASA",
        "OYJ", "SPA", "P", "CO", "COMPANY", "THE", "CLASS", "CL", "B", "HOLDING", "HOLDINGS", "GROUP", "GRP", "NEW",
        "PUBL", "AKTIEBOLAG", "ORD", "REG", "SHS", "SV", "SUBORDINATE", "VOTING", "KGAA", "GMBH", "DE", "OF", "AND",
        "INTL", "INTERNATIONAL", "PREF", "PRF", "VTG", "NON", "CAD", "DR", "SDR", "BEARER", "REGISTERED"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def toks(name: str) -> list[str]:
    s = re.sub(r"\(.*?\)", " ", str(name).upper())
    s = re.sub(r"[^A-Z0-9 ]", " ", s.replace("&", " AND "))
    return [t for t in s.split() if t not in STOP and len(t) > 1]


def same_company(a: str, b: str) -> float:
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    if ta[0] != tb[0] and difflib.SequenceMatcher(None, ta[0], tb[0]).ratio() < 0.8:
        return 0.0
    j = len(set(ta) & set(tb)) / len(set(ta) | set(tb))
    return max(j, difflib.SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio())


# ---------------------------------------------------------------------------------------------- public lists
def parse_list(pdf: Path) -> pd.DataFrame:
    """MSCI public list PDF -> rows (country, side, name) for every 'MSCI <COUNTRY> INDEX' section."""
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    rows, country, del_col = [], None, None
    pages = txt.split("\f")
    for page in pages:
        lines = page.splitlines()
        hdr = [l for l in lines if re.match(r"^\s*Additions\s+Deletions\s*$", l)]
        page_del = hdr[0].index("Deletions") if hdr else del_col
        for l in lines:
            m = re.match(r"^\s*MSCI (.+?) INDEX\s*$", l)
            if m:
                country = m.group(1).strip()
                continue
            if re.match(r"^\s*Additions\s+Deletions", l):
                del_col = l.index("Deletions")
                continue
            if country is None or not l.strip() or l.strip().startswith("Page ") or "MSCI Global" in l:
                continue
            if re.match(r"^\s*(ASIA PACIFIC|EUROPE, MIDDLE EAST AND AFRICA|AMERICAS|Notice and Disclaimer)", l):
                if "Notice" in l:
                    country = None
                continue
            dc = del_col if del_col is not None else page_del
            chunks = [(mm.start(), mm.group(0).strip()) for mm in re.finditer(r"\S+(?: \S+)*", l)]
            if not chunks or dc is None:
                continue
            for st, ch in chunks:
                if ch == "None":
                    continue
                side = "delete" if abs(st - dc) < abs(st - 0) and st >= dc - 6 else "add"
                rows.append(dict(country=country, side=side, name=ch))
    return pd.DataFrame(rows)


def actual_from_lists(st: pd.DataFrame, sc: pd.DataFrame | None) -> pd.DataFrame:
    """Standard changes for Canada and DM Europe with the migration direction from the Small Cap list."""
    want = {"CANADA": "CA"} | {v: k for k, v in C.EU_COUNTRIES.items()}
    x = st[st.country.isin(want)].copy()
    x["market"] = np.where(x.country == "CANADA", "CA", "EU")
    x["iso"] = x.country.map(want)
    x["kind"] = np.where(x.side == "add", "ADD", "DELETE")
    if sc is not None and len(sc):
        s = sc[sc.country.isin(want)]
        for i, r in x.iterrows():
            other = s[(s.country == r.country) & (s.side == ("add" if r.side == "delete" else "delete"))]
            if any(same_company(r["name"], o) >= 0.6 for o in other.name):
                x.loc[i, "kind"] = "MIG_STD_TO_SC" if r.side == "delete" else "MIG_SC_TO_STD"
    return x.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- holdings diffs
def actual_from_holdings(pre: str, post: str, rawdir: Path, post2: str | None = None) -> pd.DataFrame:
    def load(f, d):
        return F.parse_ch(F.fetch_ch(f, d, rawdir), f)
    rows = []
    for mkt, sf in (("CA", "CSCA"), ("EU", "IMEA")):
        a, b = load(sf, pre), load(sf, post)
        if post2:                                  # trades booked a day late: union of the T and T+1 diffs
            b2 = load(sf, post2)
            gone = set(a["isin"]) - set(b["isin"]) | (set(a["isin"]) - set(b2["isin"]))
            new_ = (set(b["isin"]) | set(b2["isin"])) - set(a["isin"])
            b = pd.concat([b, b2]).drop_duplicates("isin")
            b = b[~b["isin"].isin(gone)]
        wa, wb = load("WSML", pre), load("WSML", post)
        inm = (lambda df: df.location == "Canada") if mkt == "CA" else (lambda df: df.location.isin(B.EU_LOC))
        sc_a, sc_b = set(wa.loc[inm(wa), "isin"]), set(wb.loc[inm(wb), "isin"])
        for r in a[~a["isin"].isin(b["isin"])].itertuples():
            rows.append(dict(market=mkt, iso=B.EU_LOC.get(r.location, "CA"), side="delete", name=r.name, isin=r.isin,
                             kind="MIG_STD_TO_SC" if r.isin in sc_b else "DELETE"))
        for r in b[~b["isin"].isin(a["isin"])].itertuples():
            rows.append(dict(market=mkt, iso=B.EU_LOC.get(r.location, "CA"), side="add", name=r.name, isin=r.isin,
                             kind="MIG_SC_TO_STD" if r.isin in sc_a else "ADD"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- matching / scores
def match(pred: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    """Attach each actual change to a predicted company row (ISIN first, then name), market by market."""
    act = act.copy()
    act["company"] = None
    for i, r in act.iterrows():
        P = pred[pred.market == r.market]
        hit = None
        if isinstance(r.get("isin"), str):
            h = P[P.isins.fillna("").str.contains(r["isin"], regex=False)]
            if len(h):
                hit = h.iloc[0]
        if hit is None:
            sc = P.name.map(lambda n: same_company(n, r["name"]))
            if len(sc) and sc.max() >= 0.6:
                hit = P.loc[sc.idxmax()]
        if hit is not None:
            act.loc[i, "company"] = hit.company
            act.loc[i, "pred_name"] = hit["name"]
    return act


def score(pred: pd.DataFrame, act: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precision / recall of calls (p >= 0.5), expected vs actual, Brier, per market x side."""
    act = match(pred, act)
    pred = pred.copy()
    pred["pred_side"] = np.where(pred.category.isin(DEL_CATS), "delete", np.where(pred.category.isin(ADD_CATS), "add", ""))
    rows, detail = [], []
    for mkt in ("CA", "EU"):
        for side in ("delete", "add"):
            P = pred[(pred.market == mkt) & (pred.pred_side == side)]
            A = act[(act.market == mkt) & (act.side == side)]
            a_set = set(A.company.dropna())
            calls = set(P.loc[P.p_change >= 0.5, "company"])
            tp = calls & a_set
            y = P.company.isin(a_set).astype(float)
            brier = float(((P.p_change - y) ** 2).mean()) if len(P) else np.nan
            rows.append(dict(market=mkt, side=side, calls=len(calls), actual=len(A), actual_matched=len(a_set),
                             hits=len(tp), precision=len(tp) / len(calls) if calls else np.nan,
                             recall=len(tp) / len(A) if len(A) else np.nan,
                             expected_sum_p=round(float(P.loc[P.p_change >= 0.05, "p_change"].sum()), 2),
                             brier=round(brier, 4), n_pred_rows=len(P)))
            for r in A.itertuples():
                pr = P[P.company == r.company]
                detail.append(dict(market=mkt, side=side, actual_name=r.name, kind=r.kind, matched=r.pred_name if
                                   isinstance(r.company, str) else None,
                                   p=float(pr.p_change.iloc[0]) if len(pr) else 0.0,
                                   why=pr.why.iloc[0] if len(pr) else "not a predicted-side row (no candidate)"))
            for r in P[(P.p_change >= 0.5) & ~P.company.isin(a_set)].itertuples():
                detail.append(dict(market=mkt, side=side, actual_name=None, kind="FALSE POSITIVE", matched=r.name,
                                   p=r.p_change, why=r.why))
    return pd.DataFrame(rows), pd.DataFrame(detail)


# ---------------------------------------------------------------------------------------------- paper P&L
def pnl(files: list[Path], to: pd.Timestamp, fetch_date: str) -> pd.DataFrame:
    import yfinance as yf
    out = []
    for f in files:
        ch = pd.read_csv(f)
        if ch.empty:
            continue
        asof = pd.Timestamp(ch.asof.iloc[0])
        syms = sorted(set(ch.yf_sym.dropna()) | {"EWC", "IEUR"})
        fxs = ["EURUSD=X", "GBPUSD=X", "CADUSD=X", "CHFUSD=X", "SEKUSD=X", "NOKUSD=X", "DKKUSD=X"]
        x = yf.download(syms + fxs, start=str((asof - pd.Timedelta(days=7)).date()),
                        end=str((to + pd.Timedelta(days=1)).date()), auto_adjust=False, progress=False)["Close"]
        x.index = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
        x = x.ffill()
        x.to_parquet(C.CACHE / f"pnl_{f.stem}_{to:%Y%m%d}.parquet")
        a, b = x.loc[:asof].iloc[-1], x.loc[:to].iloc[-1]

        def usd_ret(sym, ccy):
            if sym not in a or not np.isfinite(a[sym]) or not np.isfinite(b[sym]):
                return np.nan
            r = b[sym] / a[sym]
            fx = f"{ {'GBX': 'GBP'}.get(str(ccy), str(ccy)) }USD=X".replace(" ", "")
            if str(ccy) not in ("USD", "nan") and fx in a:
                r *= b[fx] / a[fx]
            return r - 1
        bench = {"CA": usd_ret("EWC", "USD"), "EU": usd_ret("IEUR", "USD")}
        for r in ch.itertuples():
            ret = usd_ret(r.yf_sym, r.mkt_ccy)
            out.append(dict(run=f.stem, asof=asof.date(), to=to.date(), market=r.market, category=r.category,
                            name=r.name, side=r.side, p=r.p_change, ret_usd=ret, bench=bench[r.market],
                            excess_signed=r.side * (ret - bench[r.market]) if np.isfinite(ret) else np.nan))
    return pd.DataFrame(out)


def pnl_table(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, mkt), g in P.groupby(["run", "market"]):
        calls = g[g.p >= 0.5].dropna(subset=["excess_signed"])
        w = g[g.p >= 0.10].dropna(subset=["excess_signed"])
        rows.append(dict(run=run, market=mkt, n_calls=len(calls),
                         calls_mean_excess=calls.excess_signed.mean() if len(calls) else np.nan,
                         n_pweighted=len(w), pweighted_excess=(w.excess_signed * w.p).sum() / w.p.sum() if len(w) else np.nan,
                         bench=g.bench.iloc[0]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- main
def md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(none)"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def log_append(lines: list[str]):
    with open(C.LOG, "a") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="score the replay of a past review, e.g. 2026-05")
    ap.add_argument("--announced", help="announcement date YYYY-MM-DD: download and parse the MSCI public lists")
    ap.add_argument("--st-pdf", help="Standard list PDF saved by hand (if the download is blocked)")
    ap.add_argument("--sc-pdf", help="Small Cap list PDF saved by hand")
    ap.add_argument("--holdings-diff", nargs=2, metavar=("PRE", "POST"), help="iShares UCITS holdings dates across T")
    ap.add_argument("--pnl-to", help="close date for the paper P&L of every logged Nov run")
    ap.add_argument("--pnl-label", default="", help="label for the P&L block, e.g. T or T+5")
    ap.add_argument("--fetch-date", default=dt.date.today().isoformat())
    ap.add_argument("--tag", default="", help="replay: tag of the replay prediction to score")
    args = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rawdir = F.rawdir(args.fetch_date)
    files, L = [], []
    if args.replay:
        rev = args.replay
        P = C.PREV[rev]
        pred_f = sorted(C.REPLAY.glob(f"pred_{P['cutoff']}{('_' + args.tag) if args.tag else ''}_all.csv"))[-1]
        pred = pd.read_csv(pred_f)
        act = actual_from_holdings(P["pre"], P["post"], rawdir, P.get("post2"))
        tab, det = score(pred, act)
        sfx = f"_{args.tag}" if args.tag else ""
        f1 = C.SCORES / f"replay_{rev}{sfx}_scores.csv"; tab.to_csv(f1, index=False)
        f2 = C.SCORES / f"replay_{rev}{sfx}_detail.csv"; det.to_csv(f2, index=False)
        f3 = C.SCORES / f"replay_{rev}{sfx}.md"
        f3.write_text(f"# Replay of the {rev} review (prediction `{pred_f.name}`, actual = iShares UCITS holdings "
                      f"{P['pre']} -> {P['post']})\n\n{md(tab)}\n\n## Detail\n\n{md(det)}\n")
        files = [f1, f2, f3]
        L = [f"\n## SCORE replay {rev} | {now}", "",
             f"- Prediction `replay/{pred_f.name}` (sha256 `{sha(pred_f)[:16]}…`) vs CSCA/IMEA/WSML holdings {P['pre']} -> "
             f"{P['post']} (and {P['post2']}).", "", md(tab), ""]
        L += [f"- {r.market} {r.side}: {r.actual_name} ({r.kind}) p={r.p:.2f}" for r in det[det.kind != "FALSE POSITIVE"].itertuples()]
        L += [f"- {r.market} {r.side}: FALSE POSITIVE {r.matched} p={r.p:.2f}" for r in det[det.kind == "FALSE POSITIVE"].itertuples()]
    if args.announced or args.st_pdf:
        d = C.RAW / f"msci_{args.announced or args.fetch_date}"
        d.mkdir(parents=True, exist_ok=True)
        st_p = Path(args.st_pdf) if args.st_pdf else F.fetch_msci_list("STD", C.REVIEW, d)[0]
        sc_p = Path(args.sc_pdf) if args.sc_pdf else F.fetch_msci_list("SC", C.REVIEW, d)[0]
        if st_p is None:
            sys.exit("Standard list not downloadable (HTTP error): save it by hand and pass --st-pdf")
        st = parse_list(st_p)
        sc = parse_list(sc_p) if sc_p else None
        act = actual_from_lists(st, sc)
        act.to_csv(C.SCORES / "review_changes_2026-11_publiclist.csv", index=False)
        L += [f"\n## SCORE announcement {C.REVIEW} | {now}", "",
              f"- Lists: `{st_p.name}` sha256 `{sha(st_p)}`" + (f"; `{sc_p.name}` sha256 `{sha(sc_p)}`" if sc_p else
                                                                  "; Small Cap list not obtained")]
        for f in sorted(C.PRED.glob("pred_*_all.csv")):
            pred = pd.read_csv(f)
            tab, det = score(pred, act)
            f1 = C.SCORES / f"score_{f.stem}_publiclist.csv"; tab.to_csv(f1, index=False)
            f2 = C.SCORES / f"score_{f.stem}_publiclist_detail.csv"; det.to_csv(f2, index=False)
            files += [f1, f2]
            L += [f"- run `{f.name}`:", "", md(tab), ""]
    if args.holdings_diff:
        act = actual_from_holdings(args.holdings_diff[0], args.holdings_diff[1], rawdir)
        act.to_csv(C.SCORES / f"review_changes_2026-11_holdings_{args.holdings_diff[1]}.csv", index=False)
        L += [f"\n## SCORE holdings diff {args.holdings_diff[0]} -> {args.holdings_diff[1]} | {now}", ""]
        for f in sorted(C.PRED.glob("pred_*_all.csv")):
            pred = pd.read_csv(f)
            tab, det = score(pred, act)
            f1 = C.SCORES / f"score_{f.stem}_holdings.csv"; tab.to_csv(f1, index=False)
            f2 = C.SCORES / f"score_{f.stem}_holdings_detail.csv"; det.to_csv(f2, index=False)
            files += [f1, f2]
            L += [f"- run `{f.name}`:", "", md(tab), ""]
    if args.pnl_to:
        to = pd.Timestamp(args.pnl_to)
        P = pnl(sorted(C.PRED.glob("pred_*_changes.csv")), to, args.fetch_date)
        f1 = C.SCORES / f"pnl_to_{to:%Y%m%d}{('_' + args.pnl_label) if args.pnl_label else ''}.csv"
        P.to_csv(f1, index=False)
        T = pnl_table(P)
        files.append(f1)
        L += [f"\n## SCORE paper P&L to {to.date()} {args.pnl_label} | {now}", "",
              "Signed excess USD return vs EWC (Canada) / IEUR (Europe) from each run's as-of close; + = the predicted "
              "side made money (long adds, short deletes); gross, no costs.", "", md(T), ""]
    if files:
        L += ["| File | SHA-256 |", "|---|---|"] + [f"| `scores/{f.name}` | `{sha(f)}` |" for f in files]
        log_append(L)
        print("\n".join(L))


if __name__ == "__main__":
    main()
