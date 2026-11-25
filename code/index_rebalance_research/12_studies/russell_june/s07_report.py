"""Step 7: build the markdown tables (output/results_tables.md) and CAR-path charts used in RESULTS.md."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, CAL, MAIN_YEARS, EXT_YEARS

G4 = ["BUY_add", "BUY_mig", "SELL_del", "SELL_mig"]
LAB = {"BUY_add": "BUY new add (out→R2000)", "BUY_mig": "BUY R1000→R2000", "SELL_del": "SELL R2000 delete",
       "SELL_mig": "SELL R2000→R1000", "BUY_all": "BUY pooled", "SELL_all": "SELL pooled"}
md = []


def pct(x, d=1):
    return "" if pd.isna(x) else f"{x*100:+.{d}f}%"


def tab(df):
    cols = list(df.columns)
    s = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    return s


S = pd.read_csv(OUT / "car_summary.csv")
S0 = pd.read_csv(OUT / "car_summary_2008_2015.csv")

# ---- Table 1: capturable portfolio horizon -> T
rows = []
for h in ["H6", "H3", "HR"]:
    for g in G4 + ["BUY_all", "SELL_all"]:
        r = S[(S.portfolio == f"PRED-{h}") & (S.group == g) & (S.window == f"{h}_T")].iloc[0]
        rows.append({"horizon": h, "group": LAB[g], "n": r.n, "mean CAR": pct(r.mean_car), "t (yr-clust)": f"{r.t_cluster_year:.2f}",
                     "median": pct(r.median_car), "trim 1% mean": pct(r.mean_car_trim1pct), "raw": pct(r.mean_raw),
                     "minus IWM": pct(r.mean_minus_iwm), "yrs in trade dir": f"{r.years_in_trade_dir}/10", "hit rate": f"{r.hit_rate:.0%}"})
md.append("### T1. Capturable (predicted) portfolio, abnormal return from horizon close to recon close T, 2016-2025\n\n" + tab(pd.DataFrame(rows)))

rows = []
for h in ["H6", "H3", "HR"]:
    for g in G4:
        r = S[(S.portfolio == f"ACT-{h}") & (S.group == g) & (S.window == f"{h}_T")].iloc[0]
        q = S[(S.portfolio == f"PRED-{h}") & (S.group == g) & (S.window == f"{h}_T")].iloc[0]
        rows.append({"horizon": h, "group": LAB[g], "n actual": r.n, "hindsight mean": pct(r.mean_car), "t": f"{r.t_cluster_year:.2f}",
                     "hindsight median": pct(r.median_car), "predicted mean (T1)": pct(q.mean_car), "predicted median (T1)": pct(q.median_car)})
md.append("### T1b. Hindsight reference: the ACTUAL changes held from each horizon to T (not capturable), 2016-2025\n\n" + tab(pd.DataFrame(rows)))

# listing-age split of predicted adds (diagnostic: are recent IPOs / de-SPACs the drag?)
import lib as _lib
SL = pd.read_csv(OUT / "car_stock_level.csv.gz")
first = _lib.con().execute("SELECT permno, min(dlycaldt) first_dt FROM dsf GROUP BY 1").df().set_index("permno").first_dt
SL = SL[(SL.year >= 2016) & (SL.group == "BUY_add") & SL.portfolio.isin(["PRED-H6", "PRED-H3", "PRED-HR"])]
SL = SL[SL.window == SL.portfolio.str[5:] + "_T"]
SL["age"] = (pd.to_datetime(SL.start) - pd.to_datetime(SL.permno.map(first))).dt.days / 365.25
SL["listing"] = np.where(SL.age >= 1.5, "listed >= 18m", "listed < 18m")
rows = []
for (p_, l_), d in SL.groupby(["portfolio", "listing"]):
    m, se, t = _lib.cluster_t(d.ar, d.year)
    rows.append({"portfolio": p_, "listing age at horizon": l_, "n": len(d), "mean CAR": pct(m), "t": f"{t:.2f}", "median": pct(d.ar.median())})
md.append("### T1c. Predicted adds split by listing age at the horizon (recent IPOs / de-SPACs vs seasoned), horizon -> T, 2016-2025\n\n" + tab(pd.DataFrame(rows)))

# ---- Table 2: segments of the H6 portfolio + reversal, two periods
segs = ["H6_H3", "H3_HR", "HR_P", "P_Tm1", "Tm1_T", "T_Tp1", "T_Tp20", "T_Tp60"]
rows = []
for lab, SS in [("2016-25", S), ("2008-15", S0)]:
    for g in G4:
        x = SS[(SS.portfolio == "PRED-H6") & (SS.group == g)].set_index("window")
        d = {"period": lab, "group": LAB[g], "n": int(x.loc["H6_H3", "n"])}
        for w in segs:
            d[w] = f"{x.loc[w, 'mean_car']*100:+.1f} ({x.loc[w, 't_cluster_year']:.1f})"
        rows.append(d)
md.append("### T2. Where it is earned: H6-predicted portfolio by segment, mean CAR % (t clustered by year)\n\n"
          "Segments: H6=end-Dec, H3=end-Mar, HR=rank day, P=preliminary-list day (lists published after its close), "
          "Tm1=T-1, T=recon trade day, Tp20/Tp60 = 20/60 trading days after T.\n\n" + tab(pd.DataFrame(rows)))

# ---- Table 3: spread per year
sp = pd.read_csv(OUT / "spread_by_year.csv")
rows = []
for p, w in [("PRED-H6", "H6_T"), ("PRED-H3", "H3_T"), ("PRED-HR", "HR_T"), ("PRED-HR", "Tm1_T"), ("PRED-HR", "T_Tp20"), ("PRED-HR", "T_Tp60"), ("ACT-P", "P_T"), ("ACT-H6", "H6_T")]:
    x = sp[(sp.portfolio == p) & (sp.window == w)].set_index("year").sort_index()
    for yy, lab in [(MAIN_YEARS, "2016-25"), (EXT_YEARS, "2008-15")]:
        z = x.loc[x.index.isin(yy), "spread_ar"]
        d = {"portfolio": p, "window": w, "period": lab}
        if lab == "2016-25":
            for y in MAIN_YEARS:
                d[str(y)] = f"{z.get(y, np.nan)*100:+.1f}"
        else:
            for y in MAIN_YEARS:
                d[str(y)] = ""
        d["mean"] = f"{z.mean()*100:+.2f}%"
        d["t"] = f"{z.mean()/(z.std(ddof=1)/np.sqrt(len(z))):.2f}"
        d["yrs>0"] = f"{int((z>0).sum())}/{len(z)}"
        rows.append(d)
md.append("### T3. Dollar-neutral long-BUY / short-SELL, equal-weighted, abnormal-return spread by year (%)\n\n" + tab(pd.DataFrame(rows)))

B = pd.read_csv(OUT / "car_by_year.csv")
rows = []
for y in MAIN_YEARS + EXT_YEARS:
    d = {"year": y}
    for h in ["H6", "H3", "HR"]:
        for g in G4:
            r = B[(B.portfolio == f"PRED-{h}") & (B.window == f"{h}_T") & (B.group == g) & (B.year == y)]
            d[f"{h} {g}"] = "" if r.empty else f"{r.mean_car.iloc[0]*100:+.1f} ({int(r.n.iloc[0])})"
    rows.append(d)
md.append("### T3b. Capturable portfolio by year: mean BHAR % horizon -> T (n)\n\n" + tab(pd.DataFrame(rows)))

# ---- Table 4: prediction accuracy
A = pd.read_csv(OUT / "prediction_accuracy.csv")
rows = []
for lab, yy in [("2016-25", MAIN_YEARS), ("2008-15", EXT_YEARS)]:
    a = A[A.year.isin(yy)].groupby(["horizon", "group"])[["n_pred", "n_actual", "n_actual_listed_at_h", "n_hit"]].sum()
    for (h, g), r in a.iterrows():
        rows.append({"period": lab, "horizon": h, "group": LAB[g], "predicted": r.n_pred, "actual": r.n_actual, "hits": r.n_hit,
                     "precision": f"{r.n_hit/r.n_pred:.0%}", "recall": f"{r.n_hit/r.n_actual:.0%}",
                     "recall (listed at h)": f"{r.n_hit/r.n_actual_listed_at_h:.0%}"})
t4 = pd.DataFrame(rows)
t4["o"] = t4.horizon.map({"H6": 0, "H3": 1, "HR": 2})
md.append("### T4. Prediction accuracy (company level), pooled\n\n" + tab(t4.sort_values(["period", "o", "group"], ascending=[False, True, True]).drop(columns="o")))
rows = []
for y in MAIN_YEARS + EXT_YEARS:
    d = {"year": y}
    for h in ["H6", "H3", "HR"]:
        for g in ["BUY_add", "SELL_del"]:
            r = A[(A.year == y) & (A.horizon == h) & (A.group == g)].iloc[0]
            d[f"{g} {h} P/R"] = f"{r.precision:.0%}/{r.recall:.0%}"
    rows.append(d)
md.append("### T4b. Precision / recall by year (adds and deletes)\n\n" + tab(pd.DataFrame(rows)))

# ---- Table 5: footprints
F = pd.read_csv(OUT / "footprint_13f.csv")
F = F[(F.years == "2016-2025") & (F.portfolio == "PRED-HR")]
rows = []
for g in G4 + ["control"]:
    d = {"group": LAB.get(g, "controls (all)")}
    for q in ["Sep-1", "Dec-1", "Mar", "Jun", "Sep"]:
        r = F[(F.group == g) & (F.q == q)].iloc[0]
        extra = "" if g == "control" else f" [{r.diff_mean:+.2f}]"
        d[q] = f"{r['median']:.2f} / {r['mean']:.2f}{extra}"
    d["held by any pod (Mar)"] = f"{F[(F.group == g) & (F.q == 'Mar')].pct_held_any.iloc[0]:.0%}"
    d["mean # pods (Mar→Jun)"] = f"{F[(F.group == g) & (F.q == 'Mar')].mean_n_pods.iloc[0]:.1f}→{F[(F.group == g) & (F.q == 'Jun')].mean_n_pods.iloc[0]:.1f}"
    rows.append(d)
md.append("### T5a. 14-pod combined 13F holdings, % of shares outstanding: median / mean [mean minus size-momentum-matched controls], "
          "rank-day-predicted groups, 2016-2025\n\n" + tab(pd.DataFrame(rows)))
PP = pd.read_csv(OUT / "footprint_13f_pods.csv")
w = PP.pivot_table(index="family", columns=["group", "q"], values="diff") * 100
cols = [("BUY_add", "Dec-1"), ("BUY_add", "Mar"), ("BUY_add", "Jun"), ("BUY_mig", "Mar"), ("BUY_mig", "Jun"),
        ("SELL_del", "Mar"), ("SELL_del", "Jun"), ("SELL_mig", "Mar"), ("SELL_mig", "Jun")]
w = w[cols].round(1)
w.columns = [f"{LAB[g].split(' ')[0]} {g.split('_')[1]} {q}" for g, q in cols]
w = w.reset_index()
md.append("### T5a2. Per pod family: 13F holding minus size-momentum-matched controls, basis points of shares outstanding, "
          "rank-day-predicted groups, 2016-2025\n\n" + tab(w))
SI = pd.read_csv(OUT / "footprint_shortint.csv")
SI = SI[(SI.years == "2016-2025") & (SI.portfolio == "PRED-HR")]
rows = []
for b in ["11b", "12b", "01b", "02b", "03b", "04b", "05a", "05b", "06a", "06b", "07b", "08b", "09b"]:
    d = {"settlement": b.replace("a", " mid").replace("b", " end")}
    for g in G4 + ["control"]:
        r = SI[(SI.group == g) & (SI.bucket == b)]
        d[LAB.get(g, "controls")] = "" if r.empty else (f"{r['median'].iloc[0]:.2f}" + ("" if g == "control" else f" [{r.diff_median.iloc[0]:+.2f}]"))
    rows.append(d)
md.append("### T5b. Short interest, % of shares: median [median minus matched control], rank-day-predicted groups, 2016-2025 (month = settlement month; Nov-Dec are y-1)\n\n" + tab(pd.DataFrame(rows)))
TO = pd.read_csv(OUT / "footprint_turnover.csv", dtype={"month": str})
TO = TO[(TO.years == "2016-2025") & (TO.portfolio.isin(["PRED-HR", "PRED-H6"]))]
rows = []
for p in ["PRED-H6", "PRED-HR"]:
    for g in G4:
        d = {"portfolio": p, "group": LAB[g]}
        for m in ["11", "12", "01", "02", "03", "04", "05", "06", "07", "08", "09"]:
            r = TO[(TO.portfolio == p) & (TO.group == g) & (TO.month == m)]
            d[m] = f"{r.median_abn_pct.iloc[0]:+.0f}"
        rows.append(d)
md.append("### T5c. Abnormal monthly turnover, median % vs own Jul(y-2)-May(y-1) baseline, net of the controls' median, 2016-2025\n\n" + tab(pd.DataFrame(rows)))
V = pd.read_csv(OUT / "footprint_reconday_volume.csv")
V = V[V.portfolio.isin(["PRED-HR", "ACT-P"])]
md.append("### T5d. Recon-day volume / median daily volume over T-25..T-6\n\n" + tab(V.round(1)))

# ---- Table 6: capacity
C = pd.read_csv(OUT / "capacity_summary.csv")
C = C[C.group.isin(G4 + ["LONG_SHORT"])].copy()
C["o"] = C.horizon.map({"H6": 0, "H3": 1, "HR": 2})
C = C.sort_values(["o", "group"]).drop(columns="o")
md.append("### T6. Capacity (per recon, averages over 2016-2025; $m)\n\n" + tab(C.round(2)))

# ---- Table 7: hindsight from prelim (public-study comparable) + extension
rows = []
for lab, SS in [("2016-25", S), ("2008-15", S0)]:
    for g in G4:
        x = SS[(SS.portfolio == "ACT-P") & (SS.group == g)].set_index("window")
        d = {"period": lab, "group": LAB[g], "n": int(x.loc["P_T", "n"])}
        for w in ["P_T", "Tm1_T", "T_Tp1", "T_Tp20", "T_Tp60"]:
            d[w] = f"{x.loc[w, 'mean_car']*100:+.1f} ({x.loc[w, 't_cluster_year']:.1f})"
        rows.append(d)
md.append("### T7. Hindsight: actual changes, measured from the preliminary-list close (as public studies do), mean CAR % (t)\n\n" + tab(pd.DataFrame(rows)))
rows = []
for h in ["H6", "H3", "HR"]:
    for g in G4:
        r = S0[(S0.portfolio == f"PRED-{h}") & (S0.group == g) & (S0.window == f"{h}_T")].iloc[0]
        rows.append({"horizon": h, "group": LAB[g], "n": r.n, "mean CAR": pct(r.mean_car), "t": f"{r.t_cluster_year:.2f}",
                     "median": pct(r.median_car), "trim 1%": pct(r.mean_car_trim1pct), "yrs in dir": f"{r.years_in_trade_dir}/8", "hit": f"{r.hit_rate:.0%}"})
md.append("### T8. Extension 2008-2015: capturable portfolio, horizon -> T\n\n" + tab(pd.DataFrame(rows)))
rc = pd.read_csv(OUT / "recon_day_check.csv")
md.append("### T9. Recon trade-day check (CRSP volume) and list sizes\n\n" + tab(rc))
(OUT / "results_tables.md").write_text("\n\n".join(md))

# ---- charts
P = pd.read_csv(OUT / "car_paths.csv")
for lab, code in [("2016_2025", 0), ("2008_2015", 1)]:
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, g in zip(axs.flat, G4):
        for p, st in [("PRED-H6", "-"), ("PRED-H3", "-"), ("PRED-HR", "-"), ("ACT-P", "--")]:
            x = P[(P.year == code) & (P.portfolio == p) & (P.group == g)]
            ax.plot(x.rel_day, x.car * 100, st, label=p)
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(LAB[g])
        ax.set_ylabel("CAR % vs matched controls")
    yrs = MAIN_YEARS if code == 0 else EXT_YEARS
    rd = np.mean([-(pd.Series(1, index=pd.bdate_range(CAL.loc[y, 'rank_day'], CAL.loc[y, 'T'])).size - 1) for y in yrs])
    pr = np.mean([-(pd.Series(1, index=pd.bdate_range(CAL.loc[y, 'prelim'], CAL.loc[y, 'T'])).size - 1) for y in yrs])
    for ax in axs.flat:
        ax.axvline(rd, color="tab:red", lw=0.6, ls=":")
        ax.axvline(pr, color="tab:purple", lw=0.6, ls=":")
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].set_xlabel("trading days relative to recon day T (dotted: avg rank day, avg prelim list)")
    axs[1, 1].set_xlabel("trading days relative to recon day T")
    fig.suptitle(f"Russell June recon, daily CAR T-130..T+60, {lab.replace('_', '-')} (membership fixed at each horizon: CAR before the horizon is selection, not P&L)")
    fig.tight_layout()
    fig.savefig(OUT / f"car_paths_{lab}.png", dpi=110)

# capturable view: each year's path rebased to 0 at that portfolio's own formation day, then averaged across years
import lib
FORM = {"PRED-H6": "H6", "PRED-H3": "H3", "PRED-HR": "HR", "ACT-P": "P"}
for lab, yrs in [("2016_2025", MAIN_YEARS), ("2008_2015", EXT_YEARS)]:
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, g in zip(axs.flat, G4):
        for p, st in [("PRED-H6", "-"), ("PRED-H3", "-"), ("PRED-HR", "-"), ("ACT-P", "--")]:
            curves = []
            for y in yrs:
                hd = lib.horizon_dates(y)
                td = lib.tdays()
                k = -(int(((td > hd[FORM[p]]) & (td <= hd["T"])).sum()))     # formation day relative to T
                x = P[(P.year == y) & (P.portfolio == p) & (P.group == g)].set_index("rel_day").mean_ar
                x = x[x.index > k].fillna(0).cumsum()
                curves.append(x)
            c = pd.concat(curves, axis=1).sort_index()
            first = min(cc.index.min() for cc in curves)
            c = c.reindex(range(first, 61)).fillna(0).where(lambda d: d.notna(), 0)
            c = c.ffill().mean(axis=1)
            ax.plot(c.index, c.values * 100, st, label=p)
        ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(LAB[g]); ax.set_ylabel("CAR % since formation")
    axs[0, 0].legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("trading days relative to recon day T")
    fig.suptitle(f"Capturable CAR: each portfolio starts at 0 on its own formation close (H6, H3, rank day, prelim list), {lab.replace('_', '-')}")
    fig.tight_layout()
    fig.savefig(OUT / f"car_paths_capturable_{lab}.png", dpi=110)
print(open(OUT / "results_tables.md").read()[:500])
