"""Step 11: assemble RESULTS.md from the CSV outputs (tables) and the written narrative."""
import numpy as np, pandas as pd
from common import OUT

def md(df, index=False):
    return df.to_markdown(index=index)

# --- main CAR tables
def car_table(side):
    t = pd.read_csv(OUT / f"table_car_{side}_car.csv", index_col=0)
    t.index.name = "segment"
    return md(t, index=True)

# --- decay table
bp = pd.read_csv(OUT / "table_car_by_period.csv")
bp = bp[["period", "portfolio", "n", "horizon→T", "H3→R", "R→A", "A→T−1", "T−1→T (trade day)", "T→T+1", "T→T+20", "T→T+60"]]

# --- spread
sp = pd.read_csv(OUT / "table_spread.csv")
sp = sp[sp.portfolio == "pred"].drop(columns="portfolio")
sp.columns = ["formed", "window", "period", "n reviews", "mean spread %", "t (across reviews)", "median %", "share of reviews > 0"]

# --- per-year compact
ce = pd.read_csv(OUT / "car_events.csv")
def g(form, port, col):
    x = ce[(ce.form == form) & (ce.portfolio == port)]
    return x.groupby("year")[col].agg(["mean", "size"])
spec = [("H6", "pred_add", "car_H6_T", "pred adds H6→T"), ("H6", "pred_del", "car_H6_T", "pred deletes H6→T"),
        ("H3", "pred_add", "car_H3_T", "pred adds H3→T"), ("H3", "pred_del", "car_H3_T", "pred deletes H3→T"),
        ("R", "pred_add", "car_R_T", "pred adds R→T"), ("R", "pred_del", "car_R_T", "pred deletes R→T"),
        ("R", "hind_add", "car_T_Tp20", "actual adds T→T+20"), ("R", "hind_del", "car_T_Tp20", "actual deletes T→T+20")]
py = {}
pos = {}
for f, p, c, lab in spec:
    t = g(f, p, c)
    py[lab] = (t["mean"].mul(100).round(1).map("{:+.1f}".format) + " (" + t["size"].astype(str) + ")").str.replace("+nan", "–", regex=False)
    sign = 1 if p.endswith("add") else -1
    if c.endswith("Tp20"):
        sign = -sign
    m = t["mean"].dropna()
    pos[lab] = f"{int((sign * m > 0).sum())} of {len(m)}"
py = pd.DataFrame(py)
py.loc["profitable years"] = pd.Series(pos)
py.index.name = "year"

# --- splits (size)
sl = pd.read_csv(OUT / "table_splits.csv")
sl = sl[(sl.split == "size") & sl.portfolio.str.startswith("pred")].drop(columns=["split"])

# --- accuracy
ac = pd.read_csv(OUT / "table_accuracy.csv")
ac["n_pred"] = ac.n_pred.astype(int); ac["n_actual"] = ac.n_actual.astype(int); ac["hits"] = ac.hits.astype(int)
ay = pd.read_csv(OUT / "table_accuracy_by_year.csv")
ay["period"] = np.where(ay.year <= 2015, "2006–2015", "2016–2026")
ap = ay.groupby(["horizon", "side", "period"])[["n_pred", "n_actual", "hits"]].sum().astype(int)
ap["precision"] = (ap.hits / ap.n_pred).round(3); ap["recall"] = (ap.hits / ap.n_actual).round(3)
ap = ap.reset_index()
ayw = ay[ay.horizon == "R"].pivot_table(index="year", columns="side", values=["precision", "recall", "n_actual"]).round(2)
ayw.columns = [f"R {b} {a}" for a, b in ayw.columns]
fs = pd.read_csv(OUT / "table_accuracy_by_float_source.csv").round(3)
fs = fs[["float_source", "side", "n_pred", "n_actual", "hits", "precision", "recall"]]
for c_ in ["n_pred", "n_actual", "hits"]:
    fs[c_] = fs[c_].astype(int)
fl = pd.read_csv(OUT / "table_float_sensitivity.csv")
for c_ in ["n_pred", "hits"]:
    fl[c_] = fl[c_].astype(int)
h6 = pd.read_csv(OUT / "table_h6_timing.csv").rename(columns={"Unnamed: 0": "review at which the H6-predicted name actually changed"})
# --- volume
vs = pd.read_csv(OUT / "volume_summary.csv")
# --- capacity
cap = pd.read_csv(OUT / "table_capacity.csv")
# --- offcycle, 13F
oc = pd.read_csv(OUT / "table_offcycle.csv")
pods = pd.read_csv(OUT / "table_pods_13f.csv").set_index("portfolio")
pods = pods[[c for c in pods.columns if c.startswith("mean qe") or c.startswith("median qe") or c.startswith("count qe-1") or c.startswith("mean n_pods")]]

text = open(OUT / "RESULTS_template.md").read()
repl = {
    "{{CAR_ADD}}": car_table("add"), "{{CAR_DEL}}": car_table("del"), "{{DECAY}}": md(bp), "{{SPREAD}}": md(sp),
    "{{PER_YEAR}}": md(py, index=True), "{{SIZE}}": md(sl), "{{ACC}}": md(ac), "{{ACC_PERIOD}}": md(ap),
    "{{ACC_YEAR}}": md(ayw, index=True), "{{ACC_FLOAT}}": md(fs), "{{FLOAT_SENS}}": md(fl), "{{H6_TIMING}}": md(h6),
    "{{VOLUME}}": md(vs), "{{CAPACITY}}": md(cap.assign(col=cap.horizon + " " + cap.period).set_index("col").drop(columns=["horizon", "period"]).T, index=True), "{{OFFCYCLE}}": md(oc),
    "{{PODS}}": md(pods.T, index=True),
}
for k, v in repl.items():
    text = text.replace(k, v)
open(OUT / "RESULTS.md", "w").write(text)
print("RESULTS.md written,", len(text.split()), "words")
