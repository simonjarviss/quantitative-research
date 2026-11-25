"""Step 11: rough capacity and off-cycle events.

Capacity (per predicted name): ADV = median daily EUR value traded over the 63 sessions to the formation date on
the PRIMARY venue (Compustat cshtrd x close x FX). Primary-only volume understates tradable volume: the lit-MTF
subsample of 08_volume.py gives the primary share of primary + lit MTF on normal days (median), and
ADV_lit = ADV_primary / that share is shown next to it. Deployable per name = 10% x ADV x sessions from formation to T.
Per review = sum over the predicted names of a leg (median across reviews). Gross P&L per review = median deployable per
review x mean matched-control CAR (horizon -> T) of that leg in the matching sub-period (2021-26 rows: 2016-26 CAR;
2008-20 rows: 2008-20 CAR), sign flipped for a short leg. CARs enter the P&L arithmetic floored at -100%: CAR sums
daily ARs, so a collapse can sum below -100% (Wirecard, STOXX H6 delete for Dec 2020: -233%), which no long can lose
and no short can earn. The deploy-weighted CAR is concentrated in a few liquid names; the share of the leg's deployable
capital in its 5 largest names is reported next to it.
Off-cycle: index-adjusted CARs around the effective date for STOXX 600 changes outside the quarterly review.
Outputs: capacity_names.csv, table_capacity.csv, offcycle_events.csv, table_offcycle.csv
"""
import numpy as np, pandas as pd
from common import DATA, OUT, eu_calendar, xetra_calendar, review_dates

cal = eu_calendar()
z = np.load(DATA / "panel.npz")
VAL, RET = z["VAL"], z["RET"]
mkt = pd.read_parquet(DATA / "index_proxy.parquet").stoxx_proxy_tr_local.fillna(0).values
vs = pd.read_csv(OUT / "volume_summary.csv")
share = vs[(vs.basis == "lit_cons") & (vs.group == "control")].set_index("family").prim_share_normal.to_dict()
rows = []
for fam, calf, sides in [("stoxx", cal, [("add", 1), ("del", -1)]), ("dax", xetra_calendar(), [("up", 1), ("down", -1)])]:
    rev = review_dates(calf, fam)
    ce = pd.read_csv(OUT / f"car_events_{fam}.csv", dtype={"gvkey": str})
    for h in ["H6", "H3", "C"]:
        x = ce[(ce.form == h) & ce.portfolio.isin([f"pred_{s}" for s, _ in sides])]
        mean_car = x.groupby("portfolio")[f"car_{h}_T"].mean().to_dict()
        for r in x.itertuples():
            q = rev[rev.review == r.review].iloc[0]
            f = cal.get_loc(q[h]); iT = cal.get_loc(q["T"])
            adv = np.nanmedian(VAL[max(0, f - 62):f + 1, r.col])
            rows.append(dict(family=fam, review=r.review, year=r.year, horizon=h, side=r.portfolio.split("_")[1], conm=r.conm,
                             cty=r.cty, adv_eur_primary=adv, sessions_to_T=iT - f, deploy_eur=0.10 * adv * (iT - f),
                             car=getattr(r, f"car_{h}_T"), mean_car_leg=mean_car[r.portfolio]))
cap = pd.DataFrame(rows)
cap["car_floor"] = cap.car.clip(lower=-1.0)
cap.to_csv(OUT / "capacity_names.csv", index=False)
summ = []
for (fam, h, per), g in cap.assign(period=np.where(cap.year >= 2021, "2021–2026", "2008–2020")).groupby(["family", "horizon", "period"]):
    sh = share.get(fam, np.nan)
    r = dict(family=fam, horizon=h, period=per, median_adv_eur_m_primary=g.adv_eur_primary.median() / 1e6,
             primary_share_of_lit=sh, median_adv_eur_m_lit_est=g.adv_eur_primary.median() / 1e6 / sh if sh else np.nan,
             median_sessions=g.sessions_to_T.median())
    for s in g.side.unique():
        gs = g[g.side == s]
        per_rev = gs.groupby("review").agg(n=("conm", "size"), deploy=("deploy_eur", "sum"))
        sign = 1 if s in ("add", "up") else -1
        r[f"{s}_median_names_per_review"] = per_rev.n.median()
        r[f"{s}_median_deploy_per_review_eur_m"] = per_rev.deploy.median() / 1e6
        r[f"{s}_mean_car_pct_pooled"] = gs.mean_car_leg.iloc[0] * 100
        # CAR of the matching sub-period (2021-26 rows use the 2016-26 mean, 2008-20 rows the 2008-20 mean)
        allc = cap[(cap.family == fam) & (cap.horizon == h) & (cap.side == s)]
        ref = allc[allc.year >= 2016] if per == "2021–2026" else allc[allc.year <= 2020]
        r[f"{s}_mean_car_pct_subperiod"] = ref.car_floor.mean() * 100
        r[f"{s}_gross_pnl_per_review_eur_m"] = per_rev.deploy.median() / 1e6 * sign * ref.car_floor.mean()
        r[f"{s}_median_deploy_per_name_eur_m"] = gs.deploy_eur.median() / 1e6
        ok = ref.dropna(subset=["car_floor", "deploy_eur"])
        wcar = (ok.car_floor * ok.deploy_eur).sum() / ok.deploy_eur.sum() if len(ok) else np.nan
        r[f"{s}_deploy_weighted_car_pct_subperiod"] = wcar * 100
        r[f"{s}_gross_pnl_per_review_eur_m_weighted"] = per_rev.deploy.median() / 1e6 * sign * wcar
        r[f"{s}_top5_names_share_of_deploy"] = ok.deploy_eur.nlargest(5).sum() / ok.deploy_eur.sum() if len(ok) else np.nan
    summ.append(r)
summ = pd.DataFrame(summ).round(3)
summ.to_csv(OUT / "table_capacity.csv", index=False)
print(summ.T.to_string())

# ---- off-cycle STOXX 600 changes
ev = pd.read_csv(OUT / "events_stoxx.csv", parse_dates=["eff_date"], dtype={"gvkey": str})
oc = ev[ev.etype.isin(["offcycle", "offcycle_at_review"]) & (ev.col >= 0)]
rows = []
for r in oc.itertuples():
    iT = cal.searchsorted(r.eff_date) - 1
    ar = RET[:, r.col] - mkt

    def c(a, b):
        if iT + b >= len(cal) or iT + a < 0:
            return np.nan
        seg = ar[iT + a + 1:iT + b + 1]
        return np.nan if np.isnan(seg).all() else np.nansum(seg)
    rows.append(dict(side=r.side, conm=r.conm, eff=r.eff_date.date(), etype=r.etype, trades_after=r.n_post30 >= 5,
                     car_m20_m1=c(-20, -1), car_Tm1_T=c(-1, 0), car_T_T1=c(0, 1), car_T_T20=c(0, 20), mult_T_prim=r.mult_T_prim))
oc = pd.DataFrame(rows)
oc.to_csv(OUT / "offcycle_events.csv", index=False)
g = oc.groupby(["side", "trades_after"])
s = pd.DataFrame({"n": g.size(), "median_CAR_T-20→T-1 %": g.car_m20_m1.median() * 100,
                  "median_CAR_T-1→T %": g.car_Tm1_T.median() * 100, "median_CAR_T→T+1 %": g.car_T_T1.median() * 100,
                  "median_CAR_T→T+20 %": g.car_T_T20.median() * 100, "median_primary_volume_multiple_T": g.mult_T_prim.median()}).round(2)
s.to_csv(OUT / "table_offcycle.csv")
print(s.to_string())
