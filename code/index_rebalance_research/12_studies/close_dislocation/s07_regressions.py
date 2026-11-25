"""Step 7: regressions of each signed leg on predicted flow/ADV, crowding and flow x crowding.

y = side x matched-control abnormal return of the leg (bp; + = in the direction of index demand).
Regressors are z-scored within the estimation sample after winsorising at 1%/99%, so a coefficient is bp per 1 SD.
Every spec has a constant, z(log market cap) as a size control, a sell-side dummy (index selling) and a 2016+ dummy (pooled-period samples only).
Specs:
  flow       z(log flow/ADV)
  flow_fe    the same with event-date (review) fixed effects (within transformation): only within-review variation
             counts, so the passive-share level pi drops out
  flow_groupfe  fixed effects for family x move type x period (DAX: each move type has its own net pi)
  flow_baseADV  flow over the months -12..-6 baseline ADV instead of ADV60 (removes pre-event trading from the
             denominator)
  realised   z(log V_T / ADV60)  (ex-post flow proxy; mechanically related to the T move, read as descriptive)
  crowd:<c>  z(log flow/ADV) + z(c) + z(log flow/ADV) x z(c)
Crowding c (dated before T; pre-positioning measures before the announcement):
  abto      abnormal turnover months -6 .. announcement (vs months -12..-6, minus controls)
  runup3_s  side x CAR from H3 (3 months out) to the announcement; runup6_s from H6
  pod_mar   US: pods' 13F holdings at the March quarter-end (% of shares) - index buys only
  si_pre    US: short interest before the preliminary list (% of shares) - index sells only
  pod_q     CA: pods' 13F holdings at the last quarter-end before T (interlisted names only)
  all_short / pod_short  EU: disclosed net shorts (all holders / 14 pods) at T-21 sessions - index sells, to Mar 2025
SEs are clustered by event date (CR1). |t| < 2.5 is suggestive.
Output: table_regressions.csv
"""
import numpy as np
import pandas as pd
from common import OUT, ols_cluster
from s06_tables import load

LEGS = ["d0", "on1", "id1", "r1", "r1_5", "r1_20"]
CROWD = {"US Russell": [("abto", None), ("runup3_s", None), ("runup6_s", None), ("pod_mar", 1), ("si_pre", -1)],
         "CA S&P/TSX": [("abto", None), ("runup3_s", None), ("runup6_s", None), ("pod_q", None)],
         "EU STOXX 600": [("abto", None), ("runup3_s", None), ("runup6_s", None), ("all_short", -1), ("pod_short", -1)],
         "EU DAX family": [("abto", None), ("runup3_s", None), ("runup6_s", None), ("all_short", -1), ("pod_short", -1)]}


def wz(x):
    x = pd.Series(x, dtype=float)
    lo, hi = x.quantile([0.01, 0.99])
    x = x.clip(lo, hi)
    return (x - x.mean()) / x.std()


def fit(d, y, cols, fe=False, pooled=True, size=True, fe_key="review"):
    d = d.dropna(subset=[y] + cols + (["lmcap"] if size else [])).copy()
    if len(d) < 30:
        return []
    Z = pd.DataFrame({c: wz(d[c]).values for c in cols}, index=d.index)
    names = list(cols)
    if len(cols) == 2:
        Z["x"] = Z[cols[0]] * Z[cols[1]]
        names.append(f"{cols[0]} x {cols[1]}")
    if size:
        Z["size"] = wz(d.lmcap).values
        names.append("log market cap")
    Z["sell"] = (d.side < 0).astype(float).values
    names.append("sell side")
    if pooled and d.period.nunique() > 1:
        Z["p16"] = (d.period == "2016+").astype(float).values
        names.append("2016+")
    yy = d[y].values * 1e4
    X = Z.values
    if fe:
        g = d[fe_key].astype(str).values
        X = X - pd.DataFrame(X).groupby(g).transform("mean").values
        yy = yy - pd.Series(yy).groupby(g).transform("mean").values
        keep = X.std(0) > 1e-9
        X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    else:
        X = np.column_stack([np.ones(len(d)), X])
        names = ["const"] + names
    res, (n, G, r2) = ols_cluster(yy, X, d.review.values, names)
    if res is None:
        return []
    return [dict(term=r.term, coef_bp=r.coef, t=r.t, n=n, clusters=G, r2=r2) for r in res.itertuples()]


if __name__ == "__main__":
    df = load()
    df["fam2"] = df.fam
    df["grpkey"] = df.family + "|" + df.move.astype(str) + "|" + df.period
    eu = df[df.market == "EU"].copy()
    eu["fam2"] = "EU pooled"
    rows = []
    for fam, d0 in pd.concat([df, eu]).groupby("fam2"):
        for sample in ["all", "2016+"]:
            d = d0 if sample == "all" else d0[d0.period == "2016+"]
            for leg in LEGS:
                y = f"s_{leg}"
                for spec, cols, fe, sub in ([("flow", ["lflow"], False, None), ("flow_fe", ["lflow"], "review", None),
                                             ("flow_groupfe", ["lflow"], "grpkey", None),
                                             ("flow_baseADV", ["lflow_b"], False, None),
                                             ("realised", ["lvmult"], False, None)]
                                            + [(f"crowd:{c}", ["lflow", c], False, s) for c, s in CROWD.get(fam, CROWD["EU STOXX 600"])]):
                    dd = d if sub is None else d[d.side == sub]
                    for r in fit(dd, y, cols, fe=bool(fe), pooled=(sample == "all"), fe_key=fe or "review"):
                        rows.append(dict(family=fam, sample=sample, leg=leg, spec=spec,
                                         subset={None: "both sides", 1: "index buys", -1: "index sells"}[sub], **r))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_regressions.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    x = t[(t.sample == "all") & (t.term != "const") & ~t.term.isin(["sell side", "2016+"])]
    x = x.assign(cell=x.coef_bp.round(0).astype(int).astype(str) + " (" + x.t.round(1).astype(str) + ")")
    print(x.pivot_table(index=["family", "spec", "subset", "term"], columns="leg", values="cell", aggfunc="first").to_string())
