"""Step 5: descriptive footprints of the predicted groups vs matched controls.

  (a) 14 pods (17 13F filer numbers) combined 13F holdings as % of the security's shares outstanding, at the
      13F quarter-ends Sep(y-1), Dec(y-1), Mar(y), Jun(y) (just after T), Sep(y).  tr_13f.s34 by fdate;
      CUSIP8 -> permno via CRSP msf_v2 cusip/hdrcusip that month; shares out = CRSP shrout that month.
      13F is long-only (no shorts, no swaps), filed 45 days after quarter-end.
  (b) Short interest (comp_na_daily_all.sec_shortint, settlement dates) / CRSP shares, semi-monthly Nov(y-1)..Sep(y).
  (c) Abnormal monthly turnover: log(turnover_m / own baseline) minus the controls' median of the same, baseline =
      median monthly turnover Jul(y-2)..May(y-1) (skips the June y-1 recon month).  Plus recon-day volume ratio.
Groups: portfolio PRED-HR (main), PRED-H6, ACT-P; controls = that portfolio's controls; "matched" = mean of the
controls in the event stock's size x momentum cell (from s04).
Outputs: output/footprint_13f.csv, footprint_13f_pods.csv, footprint_shortint.csv, footprint_turnover.csv,
         footprint_reconday_volume.csv
"""
import numpy as np
import pandas as pd
from common import CACHE, OUT, CAL, PODS, MAIN_YEARS, EXT_YEARS
import lib

YEARS = MAIN_YEARS + EXT_YEARS
PORTS = ["PRED-HR", "PRED-H6", "ACT-P"]
CELLS = pd.concat([pd.read_parquet(CACHE / f"cells_{y}.parquet") for y in YEARS if (CACHE / f"cells_{y}.parquet").exists()])
CELLS = CELLS[CELLS.portfolio.isin(PORTS)][["portfolio", "year", "role", "group", "permco", "permno", "sz", "mq"]]
CELLS["permno"] = CELLS.permno.astype(int)
MSF = pd.read_parquet(CACHE / "msf.parquet", columns=["permno", "mthcaldt", "cusip", "hdrcusip", "shrout"])
MSF["ym"] = MSF.mthcaldt.dt.to_period("M")


def matched(df, val, keys=("portfolio", "year")):
    """Attach the mean of `val` over the controls in each event's (sz, mq) cell (fallback: size decile)."""
    keys = list(keys)
    c = df[df.role == "control"]
    full = c.groupby(keys + ["sz", "mq"])[val].agg(["mean", "count"]).reset_index()
    full = full[full["count"] >= 3].rename(columns={"mean": "m_full"}).drop(columns="count")
    sz = c.groupby(keys + ["sz"])[val].mean().rename("m_sz").reset_index()
    e = df[df.role == "event"].merge(full, on=keys + ["sz", "mq"], how="left").merge(sz, on=keys + ["sz"], how="left")
    e["matched_ctrl"] = e.m_full.fillna(e.m_sz)
    return e.drop(columns=["m_full", "m_sz"])


def describe(e, c, val, by):
    out = []
    for k, g in e.groupby(by):
        k = k if isinstance(k, tuple) else (k,)
        d = dict(zip(by, k))
        d.update(n=len(g), mean=g[val].mean(), median=g[val].median(), matched_ctrl_mean=g.matched_ctrl.mean(),
                 diff_mean=(g[val] - g.matched_ctrl).mean(), diff_median=(g[val] - g.matched_ctrl).median())
        out.append(d)
    for k, g in c.groupby([b for b in by if b != "group"]):
        k = k if isinstance(k, tuple) else (k,)
        d = dict(zip([b for b in by if b != "group"], k))
        d.update(group="control", n=len(g), mean=g[val].mean(), median=g[val].median())
        out.append(d)
    return pd.DataFrame(out)


# ---------------- (a) 13F
h = pd.read_parquet(CACHE / "pods13f.parquet")
h["family"] = h.mgrno.map(PODS)
h["ym"] = h.fdate.dt.to_period("M")
cm = pd.concat([MSF[["permno", "ym", "cusip"]], MSF[["permno", "ym", "hdrcusip"]].rename(columns={"hdrcusip": "cusip"})]).drop_duplicates(["ym", "cusip"])
h = h.merge(cm, on=["ym", "cusip"], how="inner")
hp = h.groupby(["ym", "permno", "family"]).shares.sum().reset_index()
so = MSF.set_index(["ym", "permno"]).shrout * 1000
QS = [("Sep-1", -1, 9), ("Dec-1", -1, 12), ("Mar", 0, 3), ("Jun", 0, 6), ("Sep", 0, 9)]
rows, podrows = [], []
for y in YEARS:
    base = CELLS[CELLS.year == y]
    if base.empty:
        continue
    for lab, dy, m in QS:
        ym = pd.Period(year=y + dy, month=m, freq="M")
        hh = hp[hp.ym == ym]
        tot = hh.groupby("permno").shares.sum()
        npods = hh[hh.shares > 0].groupby("permno").family.nunique()
        x = base.copy()
        x["q"] = lab
        x["shr"] = [so.get((ym, p), np.nan) for p in x.permno]
        x["pod_pct"] = x.permno.map(tot).fillna(0) / x.shr * 100
        x["n_pods"] = x.permno.map(npods).fillna(0)
        x["held"] = (x.n_pods > 0).astype(float)
        rows.append(x.dropna(subset=["shr"]))
        fam = hh.pivot_table(index="permno", columns="family", values="shares", aggfunc="sum").fillna(0)
        z = base[["portfolio", "year", "role", "group", "permno", "sz", "mq"]].copy()
        z["q"] = lab
        z["shr"] = [so.get((ym, p), np.nan) for p in z.permno]
        for f in sorted(set(PODS.values())):
            z[f] = z.permno.map(fam[f]).fillna(0) / z.shr * 100 if f in fam.columns else 0.0
        podrows.append(z.dropna(subset=["shr"]))
F = pd.concat(rows, ignore_index=True)
F["pod_pct"] = F.pod_pct.clip(upper=100)
e = matched(F, "pod_pct", keys=("portfolio", "year", "q"))
c = F[F.role == "control"]
res = []
for lab, yy in [("2016-2025", MAIN_YEARS), ("2008-2015", EXT_YEARS)]:
    ee, cc = e[e.year.isin(yy)], c[c.year.isin(yy)]
    d = describe(ee, cc, "pod_pct", ["portfolio", "group", "q"]).assign(years=lab)
    held = pd.concat([ee, cc.assign(group="control")]).groupby(["portfolio", "group", "q"]).agg(pct_held_any=("held", "mean"), mean_n_pods=("n_pods", "mean")).reset_index()
    res.append(d.merge(held, on=["portfolio", "group", "q"], how="left"))
by_year = describe(e, c, "pod_pct", ["portfolio", "group", "q", "year"]).assign(years="by_year")
pd.concat(res + [by_year]).to_csv(OUT / "footprint_13f.csv", index=False)
# per-pod family: event minus matched control, PRED-HR, 2016-2025
P = pd.concat(podrows, ignore_index=True)
P = P[(P.portfolio == "PRED-HR") & P.year.isin(MAIN_YEARS)]
fams = sorted(set(PODS.values()))
pr = []
for f in fams:
    ef = matched(P.rename(columns={f: "v"})[["portfolio", "year", "role", "group", "permno", "sz", "mq", "q", "v"]], "v",
                 keys=("portfolio", "year", "q"))
    g = ef.groupby(["group", "q"]).apply(lambda d: pd.Series({"n": len(d), "mean_pct": d.v.mean(), "matched_ctrl": d.matched_ctrl.mean(),
                                                               "diff": (d.v - d.matched_ctrl).mean()}), include_groups=False).reset_index()
    pr.append(g.assign(family=f))
pd.concat(pr).to_csv(OUT / "footprint_13f_pods.csv", index=False)
print("13F done", flush=True)

# ---------------- (b) short interest
si = pd.read_parquet(CACHE / "shortint.parquet")
ccm = pd.read_parquet(CACHE / "ccm.parquet")
si = si.merge(ccm, left_on=["gvkey", "iid"], right_on=["gvkey", "liid"])
si = si[(si.datadate >= si.linkdt) & (si.datadate <= si.linkenddt.fillna(pd.Timestamp("2099-12-31")))]
si = si.drop_duplicates(["permno", "datadate"])
sh = lib.con().execute("SELECT permno, dlycaldt, shrout FROM dsf WHERE shrout IS NOT NULL").df()
sh = sh.sort_values("dlycaldt")
si = si.sort_values("datadate")
si["permno"] = si.permno.astype("int64"); sh["permno"] = sh.permno.astype("int64")
si = pd.merge_asof(si, sh.rename(columns={"dlycaldt": "datadate"}), on="datadate", by="permno", direction="backward",
                   tolerance=pd.Timedelta(days=10))
si["si_pct"] = (si.shortint / (si.shrout * 1000) * 100).clip(upper=100)
srows = []
for y in YEARS:
    base = CELLS[CELLS.year == y]
    if base.empty:
        continue
    w = si[(si.datadate >= pd.Timestamp(y - 1, 11, 1)) & (si.datadate <= pd.Timestamp(y, 9, 30))].copy()
    w["bucket"] = w.datadate.dt.strftime("%m") + np.where(w.datadate.dt.day <= 15, "a", "b")
    w = w.groupby(["permno", "bucket"]).si_pct.mean().reset_index()
    srows.append(base.merge(w, on="permno"))
S = pd.concat(srows, ignore_index=True).dropna(subset=["si_pct"])
e = matched(S, "si_pct", keys=("portfolio", "year", "bucket"))
c = S[S.role == "control"]
order = {f"{m:02d}{h}": i for i, (m, h) in enumerate([(m, h) for m in [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9] for h in "ab"])}
out = []
for lab, yy in [("2016-2025", MAIN_YEARS), ("2008-2015", EXT_YEARS)]:
    out.append(describe(e[e.year.isin(yy)], c[c.year.isin(yy)], "si_pct", ["portfolio", "group", "bucket"]).assign(years=lab))
S2 = pd.concat(out)
S2["order"] = S2.bucket.map(order)
S2.sort_values(["years", "portfolio", "group", "order"]).to_csv(OUT / "footprint_shortint.csv", index=False)
print("SI done", flush=True)

# ---------------- (c) turnover
mt = lib.con().execute("""SELECT permno, date_trunc('month', dlycaldt) m, sum(vol) vol, avg(shrout) sh, count(vol) nd
                          FROM dsf WHERE dlydelflg <> 'Y' GROUP BY 1, 2""").df()
mt["to"] = mt.vol / (mt.sh * 1000)
mt["ym"] = pd.to_datetime(mt.m).dt.to_period("M")
trows, vrows = [], []
for y in YEARS:
    base = CELLS[CELLS.year == y]
    if base.empty:
        continue
    b = mt[(mt.ym >= pd.Period(f"{y-2}-07", "M")) & (mt.ym <= pd.Period(f"{y-1}-05", "M")) & (mt.nd >= 10)]
    bl = b.groupby("permno").to.agg(["median", "count"])
    bl = bl[bl["count"] >= 6]["median"]
    w = mt[(mt.ym >= pd.Period(f"{y-1}-11", "M")) & (mt.ym <= pd.Period(f"{y}-09", "M"))].copy()
    w = w[w.permno.isin(bl.index)]
    w["lr"] = np.log(w.to.clip(lower=1e-6) / w.permno.map(bl))
    x = base.merge(w[["permno", "ym", "lr"]], on="permno")
    for p_, g in x.groupby("portfolio"):
        cmed = g[g.role == "control"].groupby("ym").lr.median()
        g = g.assign(abn=g.lr - g.ym.map(cmed), month=g.ym.dt.strftime("%m"))
        trows.append(g)
    # recon-day volume ratio
    T = CAL.loc[y, "T"]
    dv = lib.daily(lib.tshift(T, -25), T, "permno, dlycaldt, vol")
    pre = dv[(dv.dlycaldt >= lib.tshift(T, -25)) & (dv.dlycaldt <= lib.tshift(T, -6))].groupby("permno").vol.median()
    onT = dv[dv.dlycaldt == T].set_index("permno").vol
    vr = (onT / pre).replace([np.inf], np.nan)
    vrows.append(base.assign(vol_ratio_T=base.permno.map(vr)))
Tn = pd.concat(trows, ignore_index=True)
out = []
for lab, yy in [("2016-2025", MAIN_YEARS), ("2008-2015", EXT_YEARS)]:
    z = Tn[Tn.year.isin(yy)]
    g = z.assign(group=np.where(z.role == "control", "control", z.group)).groupby(["portfolio", "group", "month"]).agg(
        n=("abn", "count"), median_abn_log=("abn", "median"), mean_abn_log=("abn", "mean")).reset_index()
    g["median_abn_pct"] = (np.exp(g.median_abn_log) - 1) * 100
    g["years"] = lab
    out.append(g)
mo = {f"{m:02d}": i for i, m in enumerate([11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
T2 = pd.concat(out)
T2["order"] = T2.month.map(mo)
T2.sort_values(["years", "portfolio", "group", "order"]).to_csv(OUT / "footprint_turnover.csv", index=False)
V = pd.concat(vrows, ignore_index=True)
V = V.assign(group=np.where(V.role == "control", "control", V.group), yrs=np.where(V.year >= 2016, "2016-2025", "2008-2015"))
V.groupby(["yrs", "portfolio", "group"]).vol_ratio_T.agg(["count", "median", "mean"]).reset_index().to_csv(OUT / "footprint_reconday_volume.csv", index=False)
print("turnover done")
