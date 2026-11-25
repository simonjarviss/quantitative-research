"""Step 8: summary statistics and markdown tables (report_tables.md + table_*.csv).

t-stats: mean / cluster-robust SE with clusters = review date (SE = sqrt(sum_g (sum_i in g e_i)^2) / N).
Spread t-stats: across reviews. Robustness: median and 10% trimmed mean.
"""
import numpy as np, pandas as pd
from scipy.stats import trim_mean
from common import DATA, OUT, tsx_calendar, review_dates

cal = tsx_calendar()
rev = review_dates(cal)
ce = pd.read_csv(OUT / "car_events.csv")
pr = pd.read_csv(OUT / "predictions_final.csv")
md = []


def clt(x, g):
    x = np.asarray(x, float); g = np.asarray(g); ok = ~np.isnan(x); x, g = x[ok], g[ok]
    n = len(x)
    if n < 5 or pd.Series(g).nunique() < 3:
        return np.nan
    e = pd.Series(x - x.mean()).groupby(g).sum()
    return x.mean() / (np.sqrt((e ** 2).sum()) / n)


def cell(x, g):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "–"
    t = clt(x, pd.Series(g)[x.index])
    return f"{x.mean()*100:+.2f} ({t:+.1f})"


def tbl(df):
    return df.to_markdown()


SEG = [("H6", "H3", "H6→H3"), ("H3", "R", "H3→R"), ("R", "A", "R→A"), ("A", "Tm1", "A→T−1"), ("Tm1", "T", "T−1→T (trade day)"),
       ("T", "Tp1", "T→T+1"), ("T", "Tp20", "T→T+20"), ("T", "Tp60", "T→T+60")]
pref = "car"
for side, sname in [("add", "Adds (long)"), ("del", "Deletes (short leg; negative = profit for a short)")]:
    for pref, lab in [("car", "matched-control"), ("cari", "index-adjusted")]:
        cols = {}
        specs = [("Hindsight (actual), formed H6", "H6", f"hind_{side}"), ("Predicted at H6", "H6", f"pred_{side}"),
                 ("Predicted at H3", "H3", f"pred_{side}"), ("Predicted at R", "R", f"pred_{side}"),
                 ("Naive at R", "R", f"naive_{side}")]
        for name, form, port in specs:
            x = ce[(ce.form == form) & (ce.portfolio == port)]
            c = {"n names": f"{len(x)}", "n reviews": f"{x.review.nunique()}",
                 "horizon→T": cell(x[f"{pref}_{form}_T"], x.review)}
            for a, b, l in SEG:
                c[l] = cell(x[f"{pref}_{a}_{b}"], x.review)
            cols[name] = c
        t = pd.DataFrame(cols)
        t.to_csv(OUT / f"table_car_{side}_{pref}.csv")
        md.append(f"### CAR by segment, {sname}, {lab} (% mean, cluster t by review)\n\n" + tbl(t) + "\n")

# segment CARs by period (decay), capturable and hindsight
bp = []
for per, lo_, hi_ in [("2006–2015", 2006, 2015), ("2016–2026", 2016, 2026)]:
    for name, form, port in [("Hindsight adds (H6)", "H6", "hind_add"), ("Pred adds H6", "H6", "pred_add"), ("Pred adds H3", "H3", "pred_add"),
                             ("Pred adds R", "R", "pred_add"), ("Hindsight deletes (H6)", "H6", "hind_del"), ("Pred deletes H6", "H6", "pred_del"),
                             ("Pred deletes H3", "H3", "pred_del"), ("Pred deletes R", "R", "pred_del")]:
        x = ce[(ce.form == form) & (ce.portfolio == port) & ce.year.between(lo_, hi_)]
        r = {"period": per, "portfolio": name, "n": len(x), "horizon→T": cell(x[f"car_{form}_T"], x.review)}
        for a, b, l in SEG:
            r[l] = cell(x[f"car_{a}_{b}"], x.review)
        bp.append(r)
bp = pd.DataFrame(bp)
bp.to_csv(OUT / "table_car_by_period.csv", index=False)
md.append("### Segment CARs by period (matched-control, % mean, cluster t)\n\n" + bp.to_markdown(index=False) + "\n")

# robustness: median and trimmed mean for horizon->T and post windows
rob = []
for form in ["H6", "H3", "R"]:
    for port in ["pred_add", "pred_del", "hind_add", "hind_del"]:
        x = ce[(ce.form == form) & (ce.portfolio == port)]
        for w in [f"{form}_T", "T_Tp1", "T_Tp20", "T_Tp60"]:
            v = x[f"car_{w}"].dropna()
            rob.append(dict(form=form, portfolio=port, window=w.replace("_", "→").replace("Tp", "T+"), n=len(v),
                            mean=v.mean() * 100, t_cluster=clt(v, x.loc[v.index, "review"]), median=v.median() * 100,
                            trim10=trim_mean(v, 0.1) * 100, share_pos=(v > 0).mean()))
rob = pd.DataFrame(rob).round(2)
rob.to_csv(OUT / "table_robustness.csv", index=False)
md.append("### Robustness: mean vs median vs 10% trimmed mean (%, matched-control CAR)\n\n" + rob.to_markdown(index=False) + "\n")

# long/short spread per review
sp = []
for form in ["H6", "H3", "R"]:
    for kind, (pa, pdl) in {"pred": ("pred_add", "pred_del"), "hind": ("hind_add", "hind_del"), "naive": ("naive_add", "naive_del")}.items():
        for w in [f"{form}_T", "T_Tp20"]:
            a = ce[(ce.form == form) & (ce.portfolio == pa)].groupby("review")[f"car_{w}"].mean()
            d = ce[(ce.form == form) & (ce.portfolio == pdl)].groupby("review")[f"car_{w}"].mean()
            s = (a - d).dropna()
            yrs = pd.Series(s.index.str[:4].astype(int), index=s.index)
            for per, msk in [("2006–2026", yrs > 0), ("2006–2015", yrs <= 2015), ("2016–2026", yrs >= 2016)]:
                v = s[msk]
                sp.append(dict(portfolio=kind, form=form, window=w.replace("_", "→").replace("Tp", "T+"), period=per,
                               n_reviews=len(v), mean_spread=v.mean() * 100, t=v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 2 else np.nan,
                               median=v.median() * 100, share_pos=(v > 0).mean()))
sp = pd.DataFrame(sp).round(2)
sp.to_csv(OUT / "table_spread.csv", index=False)
md.append("### Long adds / short deletes, equal-weighted spread per review (adds CAR − deletes CAR, %, reviews with both legs)\n\n" + sp.to_markdown(index=False) + "\n")

# decay and size splits (matched-control)
sl = []
for form in ["H6", "H3", "R"]:
    for port in ["pred_add", "pred_del", "hind_add", "hind_del"]:
        x = ce[(ce.form == form) & (ce.portfolio == port)].copy()
        x["period"] = np.where(x.year <= 2015, "2006–2015", "2016–2026")
        med = x.size_rel.median()
        x["size"] = np.where(x.size_rel >= med, "large half", "small half")
        for by in ["period", "size", "tsx60"]:
            for k, g in x.groupby(by):
                r = dict(form=form, portfolio=port, split=by, group=str(k), n=len(g), n_reviews=g.review.nunique())
                for w in [f"{form}_T", "R_A", "A_T", "Tm1_T", "T_Tp1", "T_Tp20"]:
                    v = g[f"car_{w}"]
                    r[w.replace("_", "→").replace("Tp", "T+").replace(form + "→T", "h→T")] = cell(v, g.review)
                sl.append(r)
sl = pd.DataFrame(sl)
sl.to_csv(OUT / "table_splits.csv", index=False)
md.append("### Splits: decay (2006–2015 vs 2016–2026), size (halves of size relative to index), TSX 60 membership\n\n" + sl.to_markdown(index=False) + "\n")

# per-year table (capturable at R and H3, hindsight) h->T and T->T+20
py = []
for form in ["H3", "R"]:
    for port in ["pred_add", "pred_del", "hind_add", "hind_del"]:
        x = ce[(ce.form == form) & (ce.portfolio == port)]
        g = x.groupby("year")
        t = pd.DataFrame({"n": g.size(), f"{form}→T": g[f"car_{form}_T"].mean() * 100, "T→T+20": g.car_T_Tp20.mean() * 100})
        t["form"] = form; t["portfolio"] = port
        py.append(t.reset_index())
py = pd.concat(py)
py.to_csv(OUT / "table_per_year.csv", index=False)
wide = py.pivot_table(index="year", columns=["portfolio", "form"], values=[c for c in py.columns if "→" in c]).round(1)
pos = []
for (form, port), g in py.groupby(["form", "portfolio"]):
    col = f"{form}→T"
    v = g[col].dropna(); sign = 1 if port.endswith("add") else -1
    post = g["T→T+20"].dropna()
    pos.append(dict(form=form, portfolio=port, years=len(v),
                    years_pre_trade_profitable=int((sign * v > 0).sum()),          # long adds / short deletes, horizon->T
                    years_reversal_trade_profitable=int((-sign * post > 0).sum())))  # short adds / long deletes, T->T+20
pos = pd.DataFrame(pos)
md.append("### Per-year mean CAR (%), matched-control\n\n" + py.pivot_table(index="year", columns=["form", "portfolio"], values="n").fillna(0).astype(int).to_markdown() + "\n")
for form in ["H3", "R"]:
    t = py[py.form == form].pivot_table(index="year", columns="portfolio", values=[f"{form}→T", "T→T+20"]).round(1)
    t.columns = [f"{a} {b}" for a, b in t.columns]
    md.append(f"#### Formed at {form}\n\n" + t.to_markdown() + "\n")
md.append("Profitable years: pre-trade = long adds / short deletes from the horizon to T; reversal = short adds / long deletes from T to T+20.\n\n" + pos.to_markdown(index=False) + "\n")

# prediction accuracy
def acc(df, pc, ac):
    tp = (df[pc] & df[ac]).sum(); fp = (df[pc] & ~df[ac]).sum(); fn = (~df[pc] & df[ac]).sum()
    return pd.Series(dict(n_pred=tp + fp, n_actual=tp + fn, hits=tp, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1)))
ac = []
for h in ["H6", "H3", "R"]:
    x = pr[pr.horizon == h]
    for model, pa, pdl in [("main", "pred_add", "pred_del"), ("rules only", "pred_add_rules", "pred_del_rules"),
                           ("rules + M&A", "pred_add_ma", "pred_del_ma"), ("naive", "naive_add", "naive_del")]:
        for side, pc, a in [("add", pa, "act_add"), ("delete", pdl, "act_del")]:
            r = acc(x, pc, a); r["horizon"] = h; r["model"] = model; r["side"] = side
            ac.append(r)
ac = pd.DataFrame(ac)[["horizon", "model", "side", "n_pred", "n_actual", "hits", "precision", "recall"]]
ac["precision"] = ac.precision.round(3); ac["recall"] = ac.recall.round(3)
ac.to_csv(OUT / "table_accuracy.csv", index=False)
md.append("### Prediction accuracy, pooled Mar 2006 – Jun 2026 (82 reviews)\n\n" + ac.to_markdown(index=False) + "\n")
# H6 alternative target: added at q-1 or q
# per year
ay = []
for (h, y), x in pr.groupby(["horizon", "year"]):
    for side, pc, a in [("add", "pred_add", "act_add"), ("delete", "pred_del", "act_del")]:
        r = acc(x, pc, a); r["horizon"] = h; r["year"] = y; r["side"] = side; ay.append(r)
ay = pd.DataFrame(ay)
ay.to_csv(OUT / "table_accuracy_by_year.csv", index=False)
t = ay[ay.horizon.isin(["H3", "R"])].pivot_table(index="year", columns=["horizon", "side"], values=["precision", "recall"]).round(2)
t.columns = [f"{a[:4]} {b} {c}" for a, b, c in t.columns]
md.append("### Accuracy by year (main model)\n\n" + t.to_markdown() + "\n")
# by float source at R
fs = []
x = pr[pr.horizon == "R"]
for src, g in x.groupby("ff_src"):
    for side, pc, a in [("add", "pred_add", "act_add"), ("delete", "pred_del", "act_del")]:
        r = acc(g, pc, a); r["float_source"] = src; r["side"] = side; fs.append(r)
fs = pd.DataFrame(fs)
fs.to_csv(OUT / "table_accuracy_by_float_source.csv", index=False)
md.append("### Accuracy at R by float source of the predicted/actual name (main model)\n\n" + fs.round(3).to_markdown(index=False) + "\n")
try:
    b = pd.read_csv(OUT / "predictions_all100.csv"); a0 = pd.read_csv(OUT / "predictions.csv")
    fl = []
    for h in ["H6", "H3", "R"]:
        for nm, df in [("estimated float (DS -> WS -> 100%)", a0), ("every float = 100%", b)]:
            xx = df[df.horizon == h]
            for side, pc, a in [("add", "pred_add", "act_add"), ("delete", "pred_del", "act_del")]:
                r = acc(xx, pc, a); r["horizon"] = h; r["float"] = nm; r["side"] = side; fl.append(r)
    fl = pd.DataFrame(fl)[["horizon", "float", "side", "n_pred", "hits", "precision", "recall"]].round(3)
    fl.to_csv(OUT / "table_float_sensitivity.csv", index=False)
    md.append("### Float sensitivity (rules-only screen)\n\n" + fl.to_markdown(index=False) + "\n")
except FileNotFoundError:
    pass

open(OUT / "report_tables.md", "w").write("\n".join(md))
print("\n".join(md)[:20000])
