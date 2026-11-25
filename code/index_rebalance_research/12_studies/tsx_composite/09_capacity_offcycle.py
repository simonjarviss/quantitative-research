"""Step 9: rough capacity, off-cycle events, H6 eventual-hit rates.

Capacity: ADV = median daily C$ value traded over the 63 sessions to the formation date, on Canadian
consolidated volume where Datastream has it (2014+), else TSX-only (understates; flagged). Deployable per
name = 10% x ADV x sessions from formation to T. Per review = sum over predicted adds + deletes.
Gross P&L = deployable x pooled mean matched-control CAR (horizon -> T) of that leg.
Off-cycle: index-adjusted CARs around the effective date for changes outside the quarterly review.
"""
import numpy as np, pandas as pd
from common import DATA, OUT, tsx_calendar, review_dates

cal = tsx_calendar()
rev = review_dates(cal).reset_index(drop=True); rev["qn"] = np.arange(len(rev))
z = np.load(DATA / "panel.npz")
P, VC, VT, AJ, RET = z["P"], z["VC"], z["VT"], z["AJ"], z["RET"]
AJf = pd.DataFrame(AJ).ffill().values
VAL = np.where(np.isnan(VC), VT, np.fmax(VC, VT)) * P / AJf      # C$ value traded per day
VALsrc = ~np.isnan(VC)
pr = pd.read_csv(OUT / "predictions_final.csv", parse_dates=["date"])
ce = pd.read_csv(OUT / "car_events.csv")
mkt = pd.read_parquet(DATA / "index_proxy.parquet").proxy_tr.fillna(0).values
out = []
cap_rows = []
for q in rev.itertuples():
    iT = cal.get_loc(q.T)
    for h in ["H6", "H3", "R"]:
        f = cal.get_loc(getattr(q, h)); n_sess = iT - f
        x = pr[(pr.review == q.review) & (pr.horizon == h) & (pr.pred_add | pr.pred_del)]
        for r in x.itertuples():
            adv = np.nanmedian(VAL[max(0, f - 62):f + 1, r.col])
            cons = VALsrc[max(0, f - 62):f + 1, r.col].mean() > 0.5
            cap_rows.append(dict(review=q.review, year=q.year, horizon=h, side="add" if r.pred_add else "delete", tic=r.tic,
                                 adv_cad=adv, adv_consolidated=cons, sessions_to_T=n_sess,
                                 deploy_cad=0.10 * adv * n_sess, mcap=r.mcap))
cap = pd.DataFrame(cap_rows)
cap.to_csv(OUT / "capacity_names.csv", index=False)
meanCAR = {(h, s): ce[(ce.form == h) & (ce.portfolio == f"pred_{'add' if s == 'add' else 'del'}")][f"car_{h}_T"].mean()
           for h in ["H6", "H3", "R"] for s in ["add", "delete"]}
summ = []
for (h, per), g in cap.assign(period=np.where(cap.year >= 2021, "2021–2026", "2006–2020")).groupby(["horizon", "period"]):
    per_rev = g.groupby(["review", "side"]).agg(n=("tic", "size"), deploy=("deploy_cad", "sum")).reset_index()
    rr = dict(horizon=h, period=per, median_adv_cad_m=g.adv_cad.median() / 1e6,
              share_consolidated=g.adv_consolidated.mean(), median_sessions=g.sessions_to_T.median())
    for s in ["add", "delete"]:
        ps = per_rev[per_rev.side == s]
        rr[f"median_names_per_review_{s}"] = ps.n.median()
        rr[f"median_deploy_per_review_{s}_cad_m"] = ps.deploy.median() / 1e6
        rr[f"mean_car_{s}_pct"] = meanCAR[(h, s)] * 100
        sign = 1 if s == "add" else -1
        rr[f"gross_pnl_per_review_{s}_cad_m"] = ps.deploy.median() / 1e6 * sign * meanCAR[(h, s)]
    summ.append(rr)
summ = pd.DataFrame(summ).round(3)
summ.to_csv(OUT / "table_capacity.csv", index=False)
print(summ.T.to_string())

# ---- H6 eventual hit: predicted at H6 for q, added (deleted) at q-1, q or q+1 ----
ev = pd.read_csv(OUT / "events.csv")
ev = ev[ev.etype == "review"].merge(rev[["review", "qn"]], on="review")
ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
pr = pr.merge(rev[["review", "qn"]], on="review", how="left") if "qn" not in pr else pr
hits = []
for side, pc, es in [("add", "pred_add", "add"), ("delete", "pred_del", "delete")]:
    x = pr[(pr.horizon == "H6") & pr[pc]]
    e = ev[ev.side == es].groupby("lid").qn.apply(set).to_dict()
    def which(l, q):
        s = e.get(l, set())
        return "q" if q in s else "q-1" if q - 1 in s else "q+1" if q + 1 in s else "q+2..q+4" if s & {q + 2, q + 3, q + 4} else "none"
    w = pd.Series([which(l, q) for l, q in zip(x.lid, x.qn)]).value_counts(normalize=True).round(3)
    hits.append(w.rename(side))
h6 = pd.concat(hits, axis=1)
h6.to_csv(OUT / "table_h6_timing.csv")
print("H6 predictions: review at which the name actually changed\n", h6)

# ---- off-cycle events: index-adjusted CAR around the effective date ----
ev2 = pd.read_csv(OUT / "events.csv", parse_dates=["eff_date", "T"])
L = pd.read_parquet(DATA / "panel_lines.parquet"); lid_ix = pd.Index(L.lid)
ev2["col"] = lid_ix.get_indexer(ev2.gvkey.astype(str).str.zfill(6) + "_" + ev2.iid)
oc = ev2[ev2.etype.isin(["offcycle", "offcycle_at_review"]) & (ev2.col >= 0)]
rows = []
for r in oc.itertuples():
    iT = cal.searchsorted(r.eff_date) - 1
    ar = RET[:, r.col] - mkt
    def c(a, b):
        if iT + b >= len(cal) or iT + a < 0:
            return np.nan
        seg = ar[iT + a + 1:iT + b + 1]
        return np.nan if np.isnan(seg).all() else np.nansum(seg)
    rows.append(dict(side=r.side, tic=r.tic, eff=r.eff_date.date(), etype=r.etype, new_listing=r.new_listing,
                     trades_after=r.n_post30 >= 5, car_m20_m1=c(-20, -1), car_Tm1_T=c(-1, 0), car_T_T1=c(0, 1),
                     car_T_T20=c(0, 20), mult_T_tsx=r.mult_T_tsx))
oc = pd.DataFrame(rows)
oc.to_csv(OUT / "offcycle_events.csv", index=False)
g = oc.groupby(["side", "trades_after"])
s = pd.DataFrame({"n": g.size(), "new_listing": g.new_listing.sum(),
                  "median_CAR_T-20→T-1 %": g.car_m20_m1.median() * 100, "median_CAR_T-1→T %": g.car_Tm1_T.median() * 100,
                  "median_CAR_T→T+1 %": g.car_T_T1.median() * 100, "median_CAR_T→T+20 %": g.car_T_T20.median() * 100,
                  "median_TSX_volume_multiple_T": g.mult_T_tsx.median()}).round(2)
s.to_csv(OUT / "table_offcycle.csv")
print(s.to_string())
