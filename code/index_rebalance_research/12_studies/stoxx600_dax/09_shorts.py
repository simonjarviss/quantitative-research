"""Step 9: pod footprint in EU net short disclosures (wrdsapps_eushort.eushort, positions >= 0.5% of shares, to
2025-03-18).

Position of holder h in ISIN i at date d = the latest disclosure on or before d (a disclosure below 0.5% closes the
position). Holders grouped into the 14 pod families of the Russell study (Millennium, Citadel (excl. Citadel
Securities), Balyasny, Schonfeld, ExodusPoint, Point72, Jane Street, Hudson Bay, Verition, Marshall Wace,
Squarepoint, Walleye, Two Sigma, Qube); 'all holders' = every disclosed holder.
Dates: T + 21k sessions, k = -6..+3 (months before/after the trade day), only dates <= 2025-03-18.
Coverage: Spanish disclosures carry no ISIN (Spain excluded); UK and Belgian rows thin out after 2019 (the
'ex-UK/BE after 2019' cut drops those country-years).
Groups: predicted deletes (rules model at C and H3), actual deletes, predicted/actual adds (STOXX); predicted and
actual demotions/promotions (DAX family); controls = the control universe of 07_returns.py at C. Matched control
value = mean over controls of the same review in the same country (STOXX) and size quintile; fallback size quintile.
Outputs: shorts_footprint.csv (name x date), shorts_summary.csv, shorts_by_pod.csv
"""
import re
import numpy as np, pandas as pd
from common import DATA, OUT, eu_calendar, xetra_calendar, review_dates

cal = eu_calendar()
L = pd.read_parquet(DATA / "panel_lines.parquet")
e = pd.read_parquet(DATA / "eushort.parquet")
e = e[e["isin"].notna() & e.net_short_position.notna()].copy()
e["position_date"] = pd.to_datetime(e.position_date)
h = e.position_holder.fillna("").str.upper()
PODS = [("Millennium", r"MILLENNIUM"), ("Citadel", r"CITADEL(?!.*SECURITIES)"), ("Balyasny", r"BALYASNY"),
        ("Schonfeld", r"SCHONFELD"), ("ExodusPoint", r"EXODUS ?POINT"), ("Point72", r"POINT ?72"),
        ("Jane Street", r"JANE STREET"), ("Hudson Bay", r"HUDSON BAY"), ("Verition", r"VERITION"),
        ("Marshall Wace", r"MARSHALL WACE"), ("Squarepoint", r"SQUAREPOINT"), ("Walleye", r"WALLEYE"),
        ("Two Sigma", r"TWO SIGMA"), ("Qube", r"QUBE")]
e["pod"] = None
for nm, pat in PODS:
    e.loc[h.str.contains(pat, regex=True) & e.pod.isna(), "pod"] = nm
e["holder"] = np.where(e.pod.notna(), e.pod, h.str.replace(r"[^A-Z0-9 ]", "", regex=True).str.strip())
# one record per holder family x ISIN x date (several regulators / entities may report the same position)
e = e.groupby(["holder", "pod", "isin", "position_date"], dropna=False).net_short_position.max().reset_index()
e["pos"] = np.where(e.net_short_position >= 0.5, e.net_short_position, 0.0)
e = e.sort_values(["isin", "holder", "position_date"])
END = pd.Timestamp("2025-03-18")
print("pod rows:", e.pod.notna().sum(), "| pods:", e.pod.value_counts().to_dict())

# ISINs of each panel line's company (all lines' ISINs + Datastream ISIN history of the line)
dn = pd.read_parquet(DATA / "ds_names.parquet")
ic_isins = dn.groupby("infocode")["isin"].apply(lambda x: set(x.dropna())).to_dict()
g_isins = {}
for g, x in L.groupby("gvkey"):
    s = set(x["isin"].dropna())
    for ic in x.ds_pq.dropna():
        s |= ic_isins.get(int(ic), set())
    g_isins[g] = s
# per company: one (holder, pod, dates, positions) record per holder x ISIN
HOLD = {}
for (isin, holder), x in e.groupby(["isin", "holder"], dropna=False):
    HOLD.setdefault(isin, []).append((holder, x.pod.iloc[0], x.position_date.values, x.pos.values))
G_HOLD = {g: [r for i in s for r in HOLD.get(i, [])] for g, s in g_isins.items()}


def snaps(g, dates):
    """Per date: n pods with >= 0.5%, sum of pod %, n all holders, sum of all %, and per-pod % dict."""
    dts = np.array(dates, dtype="datetime64[ns]")
    n_p = np.zeros(len(dts)); s_p = np.zeros(len(dts)); n_a = np.zeros(len(dts)); s_a = np.zeros(len(dts))
    per = [dict() for _ in dts]
    pods_seen = [set() for _ in dts]
    for holder, pod, hd, hp in G_HOLD.get(g, []):
        j = np.searchsorted(hd, dts, side="right") - 1
        v = np.where(j >= 0, hp[np.clip(j, 0, None)], 0.0)
        on = v > 0
        n_a += on; s_a += v
        if isinstance(pod, str):
            s_p += v
            for k in np.where(on)[0]:
                pods_seen[k].add(pod); per[k][pod] = per[k].get(pod, 0) + v[k]
    n_p = np.array([len(x) for x in pods_seen], float)
    return n_p, s_p, n_a, s_a, per


ces = pd.read_csv(OUT / "car_events_stoxx.csv", dtype={"gvkey": str},
                  usecols=["review", "form", "portfolio", "col", "gvkey", "cty", "mcap"])
ced = pd.read_csv(OUT / "car_events_dax.csv", dtype={"gvkey": str},
                  usecols=["review", "form", "portfolio", "col", "gvkey", "cty", "mcap"])
CT = {k: g.col.values for k, g in pd.read_parquet(DATA / "controls_C.parquet").groupby(["family", "review"])}
rows, podrows = [], []
for fam, ce, calf in [("stoxx", ces, cal), ("dax", ced, xetra_calendar())]:
    rev = review_dates(calf, fam)
    s1, s2 = ("add", "del") if fam == "stoxx" else ("up", "down")
    for q in rev.itertuples():
        if q.T - pd.Timedelta(days=200) > END:
            continue
        x = ce[(ce.review == q.review) & (ce.form != "P130")]
        grp = {}
        for (form, port), g in x.groupby(["form", "portfolio"]):
            if port.startswith("hind") and form == "C":
                grp["actual_" + port.split("_")[1]] = g
            if port.startswith("pred_") and form in ("C", "H3"):
                grp[f"pred{form}_" + port.split("_")[1]] = g
        if not grp:
            continue
        iT = cal.get_loc(q.T)
        dates = {k: cal[min(iT + 21 * k, len(cal) - 1)] for k in range(-6, 4)}
        dates = {k: d for k, d in dates.items() if d <= END}
        ctrl_cols = CT.get((fam, q.review), np.array([], int))
        grp["control"] = pd.DataFrame({"col": ctrl_cols, "gvkey": L.gvkey.values[ctrl_cols], "cty": L.home_cty.values[ctrl_cols]})
        ks = list(dates); dl_ = [dates[k] for k in ks]
        for gname, g in grp.items():
            for r in g.drop_duplicates("gvkey").itertuples():
                n_p, s_p, n_a, s_a, per = snaps(r.gvkey, dl_)
                for j, k in enumerate(ks):
                    rows.append(dict(family=fam, review=q.review, year=q.year, group=gname, gvkey=r.gvkey, col=r.col, cty=r.cty,
                                     k=k, date=dl_[j], n_pods=n_p[j], pod_pct=s_p[j], any_pod=float(n_p[j] > 0), n_all=n_a[j],
                                     all_pct=s_a[j]))
                    if gname != "control":
                        for p_, v in per[j].items():
                            podrows.append(dict(family=fam, review=q.review, group=gname, gvkey=r.gvkey, k=k, pod=p_, pct=v))
        print(fam, q.review, len(rows), flush=True)
S = pd.DataFrame(rows)
S["ukbe_late"] = S.cty.isin(["GBR", "BEL"]) & (S.year >= 2020)
S["covered"] = S.cty != "ESP"
S.to_csv(OUT / "shorts_footprint.csv", index=False)
pd.DataFrame(podrows).to_csv(OUT / "shorts_by_pod.csv", index=False)

# matched controls: same review, same country (STOXX) / all (DAX), same size quintile at C
mc = pd.concat([ces[ces.form == "C"][["review", "col", "mcap"]], ced[ced.form == "C"][["review", "col", "mcap"]]]).drop_duplicates()
z = np.load(DATA / "panel.npz")
P_, SH_, AJ_, FX_ = z["P"], z["SH"], z["AJ"], z["FX"]
AJf = pd.DataFrame(AJ_).ffill().values
out = []
for sample, msk in [("covered", S.covered), ("covered ex UK/BE after 2019", S.covered & ~S.ukbe_late)]:
    d = S[msk].copy()
    res = []
    for (fam, rv), g in d.groupby(["family", "review"]):
        rev = review_dates(cal if fam == "stoxx" else xetra_calendar(), fam).set_index("review")
        f = cal.get_loc(rev.loc[rv, "C"])
        m = P_[f] * SH_[f] / AJf[f] * FX_[f]
        g = g.assign(mc=m[g.col.values])
        ctrl = g[g.group == "control"]
        if ctrl.empty:
            continue
        qb = np.nanquantile(ctrl.drop_duplicates("col").mc, [0.2, 0.4, 0.6, 0.8])
        g = g.assign(sq=np.searchsorted(qb, g.mc.fillna(0)))
        ctrl = g[g.group == "control"]
        key = ["cty", "sq", "k"] if fam == "stoxx" else ["sq", "k"]
        m1 = ctrl.groupby(key)[["n_pods", "pod_pct", "any_pod"]].mean().add_prefix("ctrl_")
        m2 = ctrl.groupby(["sq", "k"])[["n_pods", "pod_pct", "any_pod"]].mean().add_prefix("ctrl2_")
        ev_ = g[g.group != "control"].join(m1, on=key).join(m2, on=["sq", "k"])
        for c in ["n_pods", "pod_pct", "any_pod"]:
            ev_[f"ctrl_{c}"] = ev_[f"ctrl_{c}"].fillna(ev_[f"ctrl2_{c}"])
        res.append(pd.concat([ev_, ctrl]))
    d = pd.concat(res)
    for (fam, grp_, k), g in d.groupby(["family", "group", "k"]):
        r = dict(sample=sample, family=fam, group=grp_, k=k, n=len(g), n_reviews=g.review.nunique(),
                 mean_n_pods=g.n_pods.mean(), share_any_pod=g.any_pod.mean(), mean_pod_pct=g.pod_pct.mean(),
                 median_pod_pct=g.pod_pct.median(), mean_all_pct=g.all_pct.mean())
        if grp_ != "control":
            r.update(ctrl_mean_n_pods=g.ctrl_n_pods.mean(), ctrl_share_any_pod=g.ctrl_any_pod.mean(),
                     ctrl_mean_pod_pct=g.ctrl_pod_pct.mean(), diff_pod_pct=(g.pod_pct - g.ctrl_pod_pct).mean(),
                     diff_n_pods=(g.n_pods - g.ctrl_n_pods).mean())
        out.append(r)
sm = pd.DataFrame(out)
sm.to_csv(OUT / "shorts_summary.csv", index=False)
print(sm[(sm["sample"] == "covered")].pivot_table(index=["family", "group"], columns="k", values="mean_pod_pct").round(2).to_string())
