"""Step 12: markdown versions of the tables (report_tables.md) and the derived size table (table_sizes_headline.csv).
RESULTS.md quotes these numbers; every number in it can be traced to a CSV written by s03-s11.
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, clt


def md(df, floatfmt=2):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{floatfmt}f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


parts = []
# --- pi_visible by year (June review date or nearest)
pi = pd.read_parquet(DATA / "pi_daily.parquet")
pi["y"] = pi.date.dt.year
t = pi[pi.date.dt.month == 6].groupby(["family", "y"]).pi_visible.last().unstack(0) * 100
pm = pd.read_parquet(DATA / "pi_us_monthly.parquet")
pm["y"] = pm.date.dt.year
u = pm[pm.date.dt.month == 6].pivot_table(index="y", columns="family", values="pi_etf") * 100
u.columns = [c + " (ETF)" for c in u.columns]
v = pm[pm.date.dt.month == 6].pivot_table(index="y", columns="family", values="pi_allfunds") * 100
v.columns = [c + " (all US index funds)" for c in v.columns]
P = t.join(u, how="outer").join(v[["SP500 (all US index funds)", "R2000 (all US index funds)"]], how="outer")
P = P.loc[[y for y in P.index if y in (2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2025, 2026)]]
parts.append("## pi_visible (%), June of each year\n\n" + md(P.reset_index().rename(columns={"y": "year"}), 3))

# --- R
R = pd.read_csv(OUT / "table_R.csv")
parts.append("## R fits (table_R.csv)\n\n" + md(R[["sample", "family", "units", "R", "se", "t", "n", "dates"]], 2))
S = pd.read_csv(OUT / "table_R_selected.csv")
parts.append("## R selected for pi_calibration.csv (full sample; the file uses the windowed values by date)\n\n" + md(S, 3))

# --- pi_true snapshot
pc = pd.read_csv(OUT / "pi_calibration.csv")
last = pc.dropna(subset=["R"]).groupby("family").tail(1)[["family", "date", "pi_visible", "R", "pi_true", "R_window"]]
parts.append("## pi_calibration.csv, latest row per family\n\n" + md(last, 4))

# --- flow test
FT = pd.read_csv(OUT / "table_flow_test.csv")
for spec in FT.spec.unique():
    x = FT[(FT.spec == spec) & (FT.side == "all")]
    a = x.pivot_table(index=["market", "group", "period"], columns="leg", values="coef_flow")
    b = x.pivot_table(index=["market", "group", "period"], columns="leg", values="t_flow")
    cols = [c for c in ["pre5", "d0", "on1", "id1", "r1", "r1_5", "r1_20"] if c in a.columns]
    tab = a[cols].round(1).astype(str) + " (" + b[cols].round(1).astype(str) + ")"
    n = x[x.leg == "d0"].set_index(["market", "group", "period"])[["n", "dates", "corr_flow_turn"]]
    tab = tab.join(n)
    parts.append(f"## Flow test: {spec} — coefficient bp (t)\n\n" + md(tab.reset_index(), 2))
# by side
x = FT[(FT.spec == "flow + turnover + size + price, date FE") & (FT.leg == "d0")]
parts.append("## Flow test, trade day, by side\n\n" + md(x[["market", "group", "side", "period", "coef_flow", "t_flow", "coef_turn",
                                                           "t_turn", "n", "dates", "corr_flow_turn", "mean_y"]], 2))
# --- mean signed legs
F = pd.read_parquet(DATA / "flow_sample.parquet")
rows = []
for (m, g, p), d in F.groupby(["market", "grp", "period"]):
    r = dict(market=m, group=g, period=p, n=len(d))
    for leg in ["pre5", "d0", "on1", "r1", "r1_5"]:
        y = d[f"y_{leg}"].values
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        mm, se, tt = clt(y[ok], d["T"].astype(str).values[ok])
        r[leg] = f"{mm:+.0f} ({tt:.1f})"
    rows.append(r)
ML = pd.DataFrame(rows)
ML.to_csv(OUT / "table_mean_legs.csv", index=False)
parts.append("## Mean signed abnormal legs, bp (t) — + = moved with net holdings-implied demand\n\n" + md(ML))
TT = pd.read_csv(OUT / "table_flow_terciles.csv")
x = TT[(TT.leg == "d0") & (TT.period == "all") & (TT.split == "D/ADV bucket")]
parts.append("## Trade day by |D_true|/ADV60 bucket, bp (t)\n\n" + md(x[["market", "group", "bin", "n", "dates", "mean_bp", "t", "med_dadv"]], 2))

# --- sizes headline: per market x period, medians across reviews
T = pd.read_csv(OUT / "table_sizes_totals_by_review.csv")
Sz = pd.read_csv(OUT / "table_sizes_by_review.csv")
Sz = Sz[~((Sz.market == "CA") & Sz["mode"].isin(["month-end", "nearest"]))]
ev_comps = ["add", "delete", "R1000->R2000", "R2000->R1000", "tsx60 move"]
w_comps = [c for c in Sz.comp.unique() if str(c).startswith("weight")]
H = []
for (m, p), g in Sz.groupby(["market", "period"]):
    per = g.groupby("review")
    ev = per.apply(lambda x: x[x.comp.isin(ev_comps)])
    rows = {}
    for rv, x in g.groupby("review"):
        e = x[x.comp.isin(ev_comps)]; w = x[x.comp.isin(w_comps)]; u = x[x.comp == "unchanged (funding)"]
        pl = x[x.comp.str.startswith("placebo")]
        rows[rv] = dict(ev_n=e.n.sum(), ev_true=e.gross_true.sum(), w_n=w.n.sum(), w_true=w.gross_true.sum(),
                        w_over1=w.n_over_1adv.sum(), ev_over1=e.n_over_1adv.sum(), u_true=u.gross_true.sum(),
                        placebo=pl.gross_true.sum() if len(pl) else np.nan)
    X = pd.DataFrame(rows).T
    X["w_over_ev"] = X.w_true / X.ev_true.replace(0, np.nan)
    X["cont_over_ev"] = (X.w_true + X.u_true) / X.ev_true.replace(0, np.nan)
    med_adv_w = g[g.comp.isin(w_comps)].med_adv.median()
    med_adv_e = g[g.comp.isin(ev_comps)].med_adv.median()
    H.append(dict(market=m, period=p, reviews=len(X), changes_names=X.ev_n.median(), changes_notional_m=X.ev_true.median() / 1e6,
                  changes_med_adv=med_adv_e, changes_over_1adv=X.ev_over1.median(),
                  weight_names=X.w_n.median(), weight_notional_m=X.w_true.median() / 1e6, weight_med_adv=med_adv_w,
                  weight_over_1adv=X.w_over1.median(), other_members_notional_m=X.u_true.median() / 1e6,
                  weight_over_changes=X.w_over_ev.median(), all_members_over_changes=X.cont_over_ev.median(),
                  placebo_m=X.placebo.median() / 1e6))
H = pd.DataFrame(H)
H.to_csv(OUT / "table_sizes_headline.csv", index=False)
parts.append("## Size of weight-change flows vs index changes per review (medians across reviews; notional in millions of local currency, EUR for Europe; true = R x visible)\n\n" + md(H, 2))

# --- checks
C = pd.read_csv(OUT / "table_checks.csv")
parts.append("## Checks (table_checks.csv)\n\n" + md(C.fillna("")))
K = pd.read_csv(OUT / "table_known_events.csv")
parts.append("## Known events (table_known_events.csv)\n\n" + md(K.fillna(""), 3))

open(OUT / "report_tables.md", "w").write("# Net-flow study: report tables (generated by s12_report.py)\n\n" + "\n\n".join(parts) + "\n")
print(H.round(2).to_string())
print(ML.to_string())
