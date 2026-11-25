"""Step 12: assemble RESULTS.md from RESULTS_template.md (the written narrative) and the CSV outputs (tables),
as in the TSX study (tsx_composite/11_write_results.py). Reads outputs only; run after 10_report.py and 11_capacity.py.
"""
import numpy as np, pandas as pd
from common import OUT


def md(df, index=False):
    return df.to_markdown(index=index)


def car_table(fam, side):
    t = pd.read_csv(OUT / f"table_car_{fam}_{side}_car.csv", index_col=0)
    t.index.name = "segment"
    return md(t, index=True)


SEGCOLS = ["horizon→T", "H3→C", "A→T−1", "T−1→T (trade day)", "T→T+1", "T→T+20"]


def splits(fam, split, extra=()):
    t = pd.read_csv(OUT / f"table_car_splits_{fam}.csv")
    t = t[(t.split == split) & ~t.portfolio.str.startswith("Rules+memory")]
    return md(t[["group", "portfolio", "n"] + SEGCOLS + list(extra)])


def spread(fam):
    t = pd.read_csv(OUT / f"table_spread_{fam}.csv")
    t = t[t.model == "rules"].drop(columns="model")
    t.columns = ["formed", "window", "period", "n reviews", "mean spread %", "t (across reviews)", "median %",
                 "share of reviews > 0"]
    return md(t)


def per_year(fam):
    return md(pd.read_csv(OUT / f"table_per_year_{fam}.csv"))


def robust(fam):
    t = pd.read_csv(OUT / f"table_robustness_{fam}.csv")
    t = t[t.window.str.match(r"^(H6|H3|C)→T$")]
    return md(t)


def hits(fam):
    return md(pd.read_csv(OUT / f"table_hit_split_{fam}.csv"))


# ---- accuracy: compact "precision / recall (n predicted)" by horizon
ac = pd.read_csv(OUT / "table_accuracy.csv")
ac["cell"] = [f"{p:.0%} / {r:.0%} ({int(n)})" for p, r, n in zip(ac.precision, ac.recall, ac.n_pred)]
acc = ac.pivot_table(index=["family", "model", "side"], columns="horizon", values="cell", aggfunc="first")
acc = acc[["H6", "H3", "C"]].fillna("–")
acc["n actual"] = ac.groupby(["family", "model", "side"]).n_actual.max().astype(int)
order = {"rules": 0, "rules + memory": 1, "rules + M&A": 2, "v0 (no eligibility proxies)": 3, "naive size": 4}
acc = acc.reset_index()
acc["o1"] = acc.family.map({"STOXX 600": 0, "DAX family": 1})
acc["o2"] = acc.model.map(order).fillna(5)
acc = acc.sort_values(["o1", "o2", "model", "side"]).drop(columns=["o1", "o2"])
ap = pd.read_csv(OUT / "table_accuracy_by_period.csv")
ap["cell"] = [f"{p:.0%} / {r:.0%}" for p, r in zip(ap.precision, ap.recall)]
app = ap.pivot_table(index=["family", "side", "period"], columns="horizon", values="cell", aggfunc="first")[["H6", "H3", "C"]]
fs = pd.read_csv(OUT / "table_accuracy_stoxx_by_float_source.csv").round(3)
for c_ in [c for c in fs.columns if c.endswith(("n_pred", "n_actual", "hits"))]:
    fs[c_] = fs[c_].astype(int)
era = pd.read_csv(OUT / "table_accuracy_dax_by_era.csv").round(3)
for c_ in [c for c in era.columns if c.endswith(("n_pred", "n_actual", "hits"))]:
    era[c_] = era[c_].astype(int)
era["era"] = era.era.map({1: "1 (to Jun 2018)", 2: "2 (Sep 2018 – Jun 2021)", 3: "3 (Sep 2021 on)"})

# ---- DAX promotions by target, actual moves
tg = pd.read_csv(OUT / "table_dax_promotions_by_target.csv")
tg = tg[(tg.form != "C") & (tg.n >= 10)]
mv = pd.read_csv(OUT / "table_dax_by_move.csv")
mv = mv[mv.form == "C"].drop(columns=["form", "H3→T"], errors="ignore")

# ---- volume
vs = pd.read_csv(OUT / "volume_summary.csv")
keep = [("stoxx", "primary"), ("stoxx", "lit_cons"), ("dax", "de_consol")]
vs = vs[[(f, b) in keep for f, b in zip(vs.family, vs.basis)] & vs.group.isin(
    ["actual_add", "actual_del", "actual_up", "actual_down", "predC_add", "predC_del", "predC_up", "predC_down", "control"])]
vs = vs[["family", "basis", "group", "n_names", "n_reviews", "abto_m-6", "abto_m-3", "abto_m-1", "abto_T", "abto_m+1",
         "mult_T", "prim_share_normal", "prim_share_T"]]
vs = vs.astype(object).where(vs.notna(), "–")

# ---- pod short footprint: mean summed pod % of shares (difference vs matched controls)
sm = pd.read_csv(OUT / "shorts_summary.csv")
sm = sm[(sm["sample"] == "covered") & (sm.group != "control")]
sm["cell"] = [f"{m:.2f} ({d:+.2f})" for m, d in zip(sm.mean_pod_pct, sm.diff_pod_pct)]
sh = sm.pivot_table(index=["family", "group"], columns="k", values="cell", aggfunc="first")[[-6, -3, -1, 0, 1, 3]]
sh.columns = [f"month {k:+d}" if k else "T" for k in sh.columns]
k1 = sm[sm.k == -1].set_index(["family", "group"])
sh["names at −1"] = k1.n.astype(int)
sh["share with a pod short at −1 (controls)"] = [f"{a:.0%} ({b:.0%})" for a, b in zip(k1.share_any_pod, k1.ctrl_share_any_pod)]
sh = sh.reset_index()
# who: per pod, STOXX actual deletes, month -1
fp = pd.read_csv(OUT / "shorts_footprint.csv", usecols=["family", "group", "k", "covered", "gvkey", "review"])
bp = pd.read_csv(OUT / "shorts_by_pod.csv")
n_del = len(fp[(fp.family == "stoxx") & (fp.group == "actual_del") & (fp.k == -1) & fp.covered])
w = bp[(bp.family == "stoxx") & (bp.group == "actual_del") & (bp.k == -1)]
who = pd.DataFrame({"mean % of shares (all deletes)": w.groupby("pod").pct.sum() / n_del,
                    "share of deletes with a disclosed short": w.groupby("pod").apply(
                        lambda x: x[["review", "gvkey"]].drop_duplicates().shape[0], include_groups=False) / n_del})
who = who.sort_values("mean % of shares (all deletes)", ascending=False).head(6).round(3).reset_index()

# ---- capacity
cap = pd.read_csv(OUT / "table_capacity.csv")
cap = cap[cap.horizon != "C"]                                           # H6 and H3 columns
cap["col"] = cap.family.map({"stoxx": "STOXX 600", "dax": "DAX family"}) + " " + cap.horizon + " " + cap.period
capt = cap.set_index("col").drop(columns=["family", "horizon", "period"]).T
capt = capt.loc[[r for r in capt.index if not r.endswith("_pooled")]].dropna(how="all").round(2)
capt = capt.astype(object).where(capt.notna(), "–")
oc = pd.read_csv(OUT / "table_offcycle.csv")

text = open(OUT / "RESULTS_template.md").read()
repl = {
    "{{CAR_STOXX_ADD}}": car_table("stoxx", "add"), "{{CAR_STOXX_DEL}}": car_table("stoxx", "del"),
    "{{CAR_DAX_UP}}": car_table("dax", "up"), "{{CAR_DAX_DOWN}}": car_table("dax", "down"),
    "{{HIT_STOXX}}": hits("stoxx"), "{{HIT_DAX}}": hits("dax"),
    "{{DAX_TARGET}}": md(tg), "{{DAX_MOVES}}": md(mv),
    "{{DECAY_STOXX}}": splits("stoxx", "period"), "{{DECAY_DAX}}": splits("dax", "period"),
    "{{TAX}}": splits("stoxx", "taxgroup", ["mean buy tax %"]),
    "{{SPREAD_STOXX}}": spread("stoxx"), "{{SPREAD_DAX}}": spread("dax"),
    "{{PER_YEAR_STOXX}}": per_year("stoxx"), "{{PER_YEAR_DAX}}": per_year("dax"),
    "{{ROBUST_STOXX}}": robust("stoxx"), "{{ROBUST_DAX}}": robust("dax"),
    "{{ACC}}": md(acc), "{{ACC_PERIOD}}": md(app.reset_index()), "{{ACC_FLOAT}}": md(fs), "{{ACC_ERA}}": md(era),
    "{{VOLUME}}": md(vs), "{{SHORTS}}": md(sh), "{{PODS_WHO}}": md(who),
    "{{CAPACITY}}": md(capt, index=True), "{{OFFCYCLE}}": md(oc),
}
for k, v in repl.items():
    assert k in text, k
    text = text.replace(k, v)
open(OUT / "RESULTS.md", "w").write(text)
print("RESULTS.md written,", len(text.split()), "words")
