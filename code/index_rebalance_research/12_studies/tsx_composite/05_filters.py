"""Step 5: point-in-time overlays on the rules-only screen (04_predict.py).

Model variants (all point-in-time):
  rules  = 04_predict.py screen only
  rules+MA = also drop confirmed M&A targets pending at the horizon date (SDC: control deal announced,
           not completed or withdrawn by d; SPDJI-PP p.7: pending targets are generally neither added nor
           deleted at a rebalancing). Link: SDC master_cusip = first 6 characters of the Compustat CUSIP.
  main   = rules+MA plus "committee memory" learned from announcements already public at d:
           - add side: drop a non-member that our R-screen passed with >= 1.25x the size threshold at one of
             the last two announced reviews but that was not added (structural exclusion we cannot see:
             security type, domicile, "inappropriate");
           - delete side: drop a member that our R-screen failed at one of the last two announced reviews
             but that was not deleted.
Output: predictions_final.csv (adds columns pred_add_rules / pred_add_ma / pred_add ...).
"""
import numpy as np, pandas as pd
from common import DATA, OUT, tsx_calendar, review_dates

cal = tsx_calendar()
rev = review_dates(cal).reset_index(drop=True)
rev["qn"] = np.arange(len(rev))
p = pd.read_csv(OUT / "predictions.csv", parse_dates=["date"])
p = p.merge(rev[["review", "qn"]], on="review")
L = pd.read_parquet(DATA / "panel_lines.parquet")
cus6 = dict(zip(L.lid, L.cusip.str[:6]))
p["cus6"] = p.lid.map(cus6)

sdc = pd.read_parquet(DATA / "sdc_ma_canada.parquet")
sdc = sdc[(sdc.psought.fillna(sdc.pctacq) >= 50) & sdc.status.isin(["Completed", "Pending", "Unconditional"])].copy()
for c in ["dateann", "dateeff", "datewith"]:
    sdc[c] = pd.to_datetime(sdc[c])
sdc_g = dict(tuple(sdc.groupby("master_cusip")))


def pending(c6, d):
    x = sdc_g.get(c6)
    if x is None:
        return False
    live = (x.dateann <= d) & (x.datewith.isna() | (x.datewith > d)) & \
           ((x.dateeff > d) | (x.dateeff.isna() & (x.status == "Pending") & (x.dateann >= d - pd.Timedelta(days=365))))
    return bool(live.any())


p["ma_pending"] = [pending(c, d) for c, d in zip(p.cus6, p.date)]
p["pred_add_rules"] = p.pred_add; p["pred_del_rules"] = p.pred_del
p["pred_add_ma"] = p.pred_add & ~p.ma_pending
p["pred_del_ma"] = p.pred_del & ~p.ma_pending

# memory from R-horizon screens of reviews already announced at d
thr = np.where(p.review < "2019-03", 0.0005, 0.0004)
pR = p[p.horizon == "R"].assign(thr=thr[p.horizon == "R"])
rej = pR[pR.pred_add_ma & ~pR.act_add & (pR.w >= 1.25 * pR.thr)][["lid", "qn"]]
prot = pR[pR.pred_del_ma & ~pR.act_del][["lid", "qn"]]
rej_g = rej.groupby("lid").qn.apply(list).to_dict(); prot_g = prot.groupby("lid").qn.apply(list).to_dict()
A_by_qn = dict(zip(rev.qn, rev.A))


def recent(lst, d):
    """True if the line was flagged at one of the last two reviews already announced at d."""
    if not lst:
        return False
    known = sorted(q for q, a in A_by_qn.items() if a <= d)[-2:]
    return any(q in known for q in lst)


p["mem_reject"] = [recent(rej_g.get(l), d) for l, d in zip(p.lid, p.date)]
p["mem_protect"] = [recent(prot_g.get(l), d) for l, d in zip(p.lid, p.date)]
p["pred_add"] = p.pred_add_ma & ~p.mem_reject
p["pred_del"] = p.pred_del_ma & ~p.mem_protect
p.drop(columns=["cus6"]).to_csv(OUT / "predictions_final.csv", index=False)


def acc(df, pc, ac):
    tp = (df[pc] & df[ac]).sum(); fp = (df[pc] & ~df[ac]).sum(); fn = (~df[pc] & df[ac]).sum()
    return tp + fp, tp + fn, tp, round(tp / max(tp + fp, 1), 3), round(tp / max(tp + fn, 1), 3)


for h in ["H6", "H3", "R"]:
    x = p[p.horizon == h]
    for v in ["rules", "ma", ""]:
        a = acc(x, f"pred_add_{v}" if v else "pred_add", "act_add"); dd = acc(x, f"pred_del_{v}" if v else "pred_del", "act_del")
        print(h, v or "main", "ADD npred/nact/tp/P/R", a, "| DEL", dd)
