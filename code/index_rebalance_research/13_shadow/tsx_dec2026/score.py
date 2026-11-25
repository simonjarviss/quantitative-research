"""Score shadow predictions against announced S&P/TSX Composite changes, and paper-track the predicted portfolio.

Record the announced changes (one of):
  python3 score.py --review 2026-12 --release raw/releases/<file>.html     # saved Newswire.ca / S&P DJI release
  python3 score.py --review 2026-12 --release https://www.newswire.ca/news-releases/...html
  python3 score.py --review 2026-12 --add ABC,DEF --delete XYZ --announced 2026-12-04 --source "S&P DJI 4 Dec 17:15 ET"
  python3 score.py --review 2026-12 --xic-diff 2026-12-17 2026-12-18      # backup: XIC holdings diff at the effective date
Then (or later, any number of times) score every run of that review:
  python3 score.py --review 2026-12 [--run runs/2026-12/2026-10-09_H-10w] [--to 2026-12-18] [--from YYYY-MM-DD]

Accuracy: precision / recall per variant (rules, main) and side, against the recorded changes.
Paper P&L (no orders, no accounts): equal-weight long the predicted adds and short the predicted deletes, from the
close of the run's as-of date (or --from) to the close of --to (default: latest close). Total returns from yfinance
Adj Close, benchmark XIC.TO (iShares Core S&P/TSX Capped Composite) total return. Gross of costs and borrow.
Writes runs/<review>/<run>/score_<to>.md and .csv and appends a SCORE entry to shadow_log.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import live_common as lc


# ------------------------------------------------------------------------------ announced changes
def parse_release(text: str) -> tuple[list[str], list[str]]:
    """Adds and deletes from the S&P/TSX Composite table(s) of an S&P DJI release (HTML or text).

    Release tables (Newswire.ca, Sep 2024 - Jun 2026): a header 'S&P/TSX COMPOSITE INDEX - <date>' then rows
    'ADDED | Company (TSX:SYM) | sector | sub-industry' or 'DELETED | ...'. Other index tables (S&P/TSX 60,
    Completion, SmallCap ...) are skipped.
    """
    t = html.unescape(re.sub(r"<[^>]+>", " ", text))
    t = re.sub(r"\s+", " ", t)
    heads = list(re.finditer(r"S&P/TSX [A-Z0-9 /&\-]*?INDEX", t, flags=re.I))
    secs = []
    for i, h in enumerate(heads):
        endpos = heads[i + 1].start() if i + 1 < len(heads) else len(t)
        if "COMPOSITE" in h.group(0).upper() and h.group(0).isupper():
            secs.append(t[h.end():endpos])
    if not secs:                                  # fall back to the whole text
        secs = [t]
    adds, dels = [], []
    pat = re.compile(r"\b(ADDED|ADDITIONS?|DELETED|DELETIONS?|REMOVED)\b[^()]{0,200}?\((?:TSX|TSE)\s*:\s*([A-Z0-9][A-Z0-9.]*)\s*\)",
                     flags=re.I)
    for s in secs:
        for m in pat.finditer(s):
            sym = m.group(2).upper().rstrip(".")
            (adds if m.group(1).upper().startswith("ADD") else dels).append(sym)
    return sorted(set(adds)), sorted(set(dels))


def record_changes(review: str, adds: list[str], dels: list[str], announced: str, source: str):
    ch = lc.load_review_changes()
    ch = ch[ch.review != review]
    new = pd.DataFrame([dict(review=review, side="add", symbol=s, name="", announced=announced, source=source) for s in adds] +
                       [dict(review=review, side="delete", symbol=s, name="", announced=announced, source=source) for s in dels])
    out = pd.concat([ch, new], ignore_index=True).sort_values(["review", "side", "symbol"])
    out.to_csv(lc.REVIEW_CHANGES, index=False)
    print(f"recorded {review}: adds {adds} deletes {dels} ({source})")


def xic_members(d: pd.Timestamp) -> set:
    p = lc.RAW / "xic" / f"XIC_holdings_asof{d:%Y%m%d}.csv"
    if not p.exists():
        p.write_bytes(lc.http_get(lc.XIC_URL + f"&asOfDate={d:%Y%m%d}"))
    return set(lc.parse_xic(p)[1].symbol)


# ------------------------------------------------------------------------------ scoring
def accuracy(pred: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows, lines = [], []
    A = set(actual[actual.side == "add"].symbol)
    D = set(actual[actual.side == "delete"].symbol)
    for var in ["rules", "main"]:
        for side, col, act in [("add", f"pred_add_{var}", A), ("delete", f"pred_del_{var}", D)]:
            P = set(pred.loc[pred[col].astype(bool), "symbol"])
            tp = P & act
            rows.append(dict(variant=var, side=side, n_pred=len(P), n_actual=len(act), hits=len(tp),
                             precision=round(len(tp) / len(P), 3) if P else np.nan,
                             recall=round(len(tp) / len(act), 3) if act else np.nan))
            lines.append(f"{var} {side}: hits {sorted(tp)} | false positives {sorted(P - act)} | missed {sorted(act - P)}")
    return pd.DataFrame(rows), lines


def paper_pnl(pred: pd.DataFrame, d_from: pd.Timestamp, d_to: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    import yfinance as yf
    names = pred[pred.pred_add_rules.astype(bool) | pred.pred_del_rules.astype(bool)].copy()
    syms = sorted(set(names.yf)) + ["XIC.TO"]
    end = (d_to + pd.Timedelta(days=1)) if d_to is not None else pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    px = yf.download(syms, start=(d_from - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     auto_adjust=False, progress=False, group_by="column")
    ac, cl = px["Adj Close"], px["Close"]
    ac.index = pd.DatetimeIndex(ac.index).tz_localize(None).normalize()
    cl.index = ac.index
    ac = ac[ac.index <= end]
    i0 = ac.index[ac.index <= d_from]
    if len(i0) == 0:
        raise SystemExit(f"no prices on or before {d_from.date()}")
    t0 = i0[-1]
    t1 = ac.index[-1] if d_to is None else ac.index[ac.index <= d_to][-1]
    xr = ac.loc[t1, "XIC.TO"] / ac.loc[t0, "XIC.TO"] - 1
    rows = []
    for r in names.itertuples():
        s = ac[r.yf] if r.yf in ac else pd.Series(dtype=float)
        s0 = s.loc[:t0].dropna()
        s1 = s.loc[:t1].dropna()
        if s0.empty or s1.empty:
            rows.append(dict(symbol=r.symbol, note="no prices"))
            continue
        ret = s1.iloc[-1] / s0.iloc[-1] - 1
        note = "" if s1.index[-1] == t1 else f"last price {s1.index[-1].date()} (stopped trading?)"
        rows.append(dict(symbol=r.symbol, issuer=r.issuer, side="add" if r.pred_add_rules else "delete",
                         in_main=bool(r.pred_add_main if r.pred_add_rules else r.pred_del_main),
                         close_recorded=r.close_asof, close_from=cl[r.yf].loc[:t0].dropna().iloc[-1],
                         total_return=ret, xic_return=xr, excess=ret - xr,
                         position_pnl=(ret if r.pred_add_rules else -ret),
                         position_excess=((ret - xr) if r.pred_add_rules else -(ret - xr)), note=note))
    per = pd.DataFrame(rows)
    summ = []
    for var in ["rules", "main"]:
        x = per if var == "rules" else per[per.in_main.fillna(False).astype(bool)]
        a, d = x[x.side == "add"], x[x.side == "delete"]
        summ.append(dict(variant=var, n_adds=len(a), adds_mean_return=a.total_return.mean(), adds_mean_excess=a.excess.mean(),
                         n_deletes=len(d), deletes_mean_return=d.total_return.mean(),
                         short_deletes_excess=(-d.excess).mean() if len(d) else np.nan,
                         long_short_excess=(a.excess.mean() if len(a) else 0) - (d.excess.mean() if len(d) else 0)
                         if (len(a) and len(d)) else np.nan, xic_return=xr))
    return per, pd.DataFrame(summ), (t0, t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--release", help="path or URL of the S&P DJI release (HTML or text)")
    ap.add_argument("--add", default=None)
    ap.add_argument("--delete", default=None)
    ap.add_argument("--xic-diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--announced", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--run", default=None, help="one run folder; default: every run of the review")
    ap.add_argument("--from", dest="d_from", default=None)
    ap.add_argument("--to", dest="d_to", default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    rv = a.review
    rd = lc.review_dates(rv)
    ann = a.announced or str(rd["A"].date())

    if a.release:
        if a.release.startswith("http"):
            raw = lc.http_get(a.release)
            d = lc.RAW / "releases"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{rv}_{dt.date.today():%Y%m%d}_{abs(hash(a.release)) % 10**8}.html"
            p.write_bytes(raw)
            text = raw.decode("utf-8", "replace")
            src = a.source or f"release {a.release} (saved {p.name}, sha256 {lc.sha256(p)[:16]})"
        else:
            p = Path(a.release)
            text = p.read_text(errors="replace")
            src = a.source or f"release file {p.name} (sha256 {lc.sha256(p)[:16]})"
        adds, dels = parse_release(text)
        if not adds and not dels:
            raise SystemExit("parser found no Composite changes: check the file, or enter them with --add/--delete")
        record_changes(rv, adds, dels, ann, src)
    elif a.add is not None or a.delete is not None:
        adds = [s.strip().upper() for s in (a.add or "").split(",") if s.strip()]
        dels = [s.strip().upper() for s in (a.delete or "").split(",") if s.strip()]
        record_changes(rv, adds, dels, ann, a.source or "entered by hand")
    elif a.xic_diff:
        b, af = (pd.Timestamp(x) for x in a.xic_diff)
        m0, m1 = xic_members(b), xic_members(af)
        adds, dels = sorted(m1 - m0), sorted(m0 - m1)
        print("NOTE: an XIC diff also contains off-cycle changes (M&A deletions, spin-offs); edit review_changes.csv if needed")
        record_changes(rv, adds, dels, ann, a.source or f"XIC holdings diff {b.date()} -> {af.date()}")

    actual = lc.load_review_changes()
    actual = actual[actual.review == rv]
    runs = [Path(a.run)] if a.run else sorted((lc.RUNS / rv).glob("*_*"))
    runs = [r if r.is_absolute() else lc.HERE / r for r in runs]
    if not runs:
        raise SystemExit(f"no runs for {rv}")
    log = []
    for run in runs:
        pred = pd.read_csv(run / "predictions.csv", dtype={"symbol": str})
        meta = pd.read_json(run / "run_meta.json", typ="series")
        d_from = pd.Timestamp(a.d_from or meta["asof"])
        d_to = pd.Timestamp(a.d_to) if a.d_to else None
        out = [f"# Score: {rv} run {run.name}\n", f"Scored {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC.\n"]
        acc = None
        if not actual.empty:
            acc, lines = accuracy(pred, actual)
            out += ["## Accuracy vs announced changes\n", f"Source: {actual.source.iloc[0]}\n",
                    acc.to_markdown(index=False), "\n", *[f"- {x}" for x in lines], "\n"]
        else:
            out.append("No announced changes recorded yet for this review (accuracy not computed).\n")
        per, summ, (t0, t1) = paper_pnl(pred, d_from, d_to)
        out += [f"## Paper P&L, close {t0.date()} to close {t1.date()} (total return, gross, equal weight)\n",
                summ.to_markdown(index=False, floatfmt=".4f"), "\n", per.to_markdown(index=False, floatfmt=".4f"), "\n"]
        tag = f"{t1:%Y%m%d}"
        (run / f"score_{tag}.md").write_text("\n".join(out))
        per.to_csv(run / f"score_{tag}.csv", index=False)
        print("\n".join(out))
        s = summ.set_index("variant")

        def pc(x):
            return "n/a" if pd.isna(x) else f"{x:+.2%}"
        accs = ""
        if acc is not None:
            accs = "; ".join(f"{r.variant} {r.side} P {r.precision} R {r.recall} ({r.hits}/{r.n_pred} pred, {r.n_actual} actual)"
                             for r in acc.itertuples())
        log.append(f"- run `{run.relative_to(lc.HERE)}`{' (BACKFILL)' if meta.get('backfill') else ''}: P&L {t0.date()}→{t1.date()}: "
                   + "; ".join(f"{v}: adds {pc(s.loc[v,'adds_mean_excess'])} vs XIC (n {s.loc[v,'n_adds']}), "
                               f"short deletes {pc(s.loc[v,'short_deletes_excess'])} vs XIC (n {s.loc[v,'n_deletes']})" for v in s.index)
                   + f"; XIC {pc(s.iloc[0]['xic_return'])}. " + (f"Accuracy: {accs}." if accs else "Accuracy: no announcement yet.")
                   + f" Report `score_{tag}.md` sha256 `{lc.sha256(run / f'score_{tag}.md')[:16]}…`\n")
    if not a.no_log:
        with open(lc.LOG, "a") as f:
            f.write(f"\n## SCORE {rv} | {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC\n\n" + "".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
