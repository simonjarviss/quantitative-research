"""Step 7: volume footprint on Canadian consolidated volume (Datastream ds2scdqtprc, exch 295), reviews
with T >= 2015-06 (one year of consolidated history before T).

turnover(t) = consolidated volume(t) / shares outstanding(t), both split-adjusted.
Abnormal turnover in a window = mean daily turnover in the window / mean daily turnover in the baseline
window T-252..T-127 (months -12 to -6). Blocks of 21 sessions: m-6 = T-126..T-106 ... m-1 = T-21..T-1,
then T, m+1 = T+1..T+21, m+2 = T+22..T+42. Trade-day multiple = volume(T) / median volume(T-20..T-1),
consolidated and TSX-only. Groups: predicted adds/deletes (main model, R and H3), actual review changes,
controls (control universe of 06_returns.py: not predicted, not changed).
Output: volume_footprint.csv (per name), volume_summary.csv (medians by group).
"""
import numpy as np, pandas as pd
from common import DATA, OUT, tsx_calendar, review_dates

cal = tsx_calendar()
rev = review_dates(cal).reset_index(drop=True)
z = np.load(DATA / "panel.npz")
P, VC, VT, SH, AJ = z["P"], z["VC"], z["VT"], z["SH"], z["AJ"]
AJf = pd.DataFrame(AJ).ffill().values
L = pd.read_parquet(DATA / "panel_lines.parquet")
lid_ix = pd.Index(L.lid)
nd, nl = P.shape
dom = ((L["loc"] == "CAN") | (L.fic == "CAN")).values & L.gsubind.notna().values & (L.exchg != 9).values
MC = P * SH / AJf
TO = VC / SH
ev = pd.read_csv(OUT / "events.csv"); ev["col"] = lid_ix.get_indexer(ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid)
pr = pd.read_csv(OUT / "predictions_final.csv")
diag = pd.read_csv(OUT / "prediction_diagnostics.csv").set_index(["review", "horizon"])
blocks = {f"m-{k}": (-21 * k, -21 * k + 20) for k in range(6, 0, -1)}
blocks.update({"T": (0, 0), "m+1": (1, 21), "m+2": (22, 42), "last20": (-20, -1)})
rows = []
for q in rev[rev["T"] >= "2015-06-01"].itertuples():
    iT = cal.get_loc(q.T)
    e = ev[(ev.review == q.review) & (ev.etype == "review") & (ev.col >= 0)]
    x = pr[pr.review == q.review]
    grp = {"actual_add": e.loc[e.side == "add", "col"].values, "actual_del": e.loc[e.side == "delete", "col"].values}
    for h in ["H3", "R"]:
        grp[f"pred{h}_add"] = x.loc[(x.horizon == h) & x.pred_add, "col"].values
        grp[f"pred{h}_del"] = x.loc[(x.horizon == h) & x.pred_del, "col"].values
    excl = np.zeros(nl, bool)
    for v in grp.values():
        excl[v] = True
    excl[x.loc[x.pred_add_rules | x.pred_del_rules, "col"].values] = True
    iR = cal.get_loc(q.R)
    Mref = diag.loc[(q.review, "R"), "M_bn"] * 1e9
    live = (~np.isnan(P[iR - 9:iR + 1])).any(0)
    ctrl = live & dom & ~excl & (P[iR] >= 1) & (MC[iR] >= 0.00005 * Mref)
    grp["control"] = np.where(ctrl)[0]
    for g, cols in grp.items():
        for c in cols:
            base = TO[iT - 252:iT - 126, c]
            if np.sum(~np.isnan(base)) < 60 or np.nanmean(base) <= 0:
                continue
            b = np.nanmean(base)
            r = dict(review=q.review, year=q.year, group=g, col=c, tic=L.tic[c])
            for k, (a0, a1) in blocks.items():
                if iT + a1 < nd:
                    r[f"abto_{k}"] = np.nanmean(TO[iT + a0:iT + a1 + 1, c]) / b
            med = np.nanmedian(VC[iT - 20:iT, c]); medt = np.nanmedian(VT[iT - 20:iT, c])
            r["mult_T_cons"] = VC[iT, c] / med if med > 0 else np.nan
            r["mult_T_tsx"] = VT[iT, c] / medt if medt > 0 else np.nan
            r["tsx_share_T"] = VT[iT, c] / VC[iT, c] if VC[iT, c] > 0 else np.nan
            r["tsx_share_normal"] = np.nanmedian(VT[iT - 20:iT, c] / VC[iT - 20:iT, c])
            rows.append(r)
fp = pd.DataFrame(rows)
fp.to_csv(OUT / "volume_footprint.csv", index=False)
cols = [c for c in fp.columns if c.startswith("abto_") or c.startswith("mult_") or c.startswith("tsx_share")]
order = ["abto_m-6", "abto_m-5", "abto_m-4", "abto_m-3", "abto_m-2", "abto_m-1", "abto_last20", "abto_T", "abto_m+1",
         "abto_m+2", "mult_T_cons", "mult_T_tsx", "tsx_share_normal", "tsx_share_T"]
sm = fp.groupby("group")[order].median().round(2)
sm.insert(0, "n_names", fp.groupby("group").size())
sm.insert(1, "n_reviews", fp.groupby("group").review.nunique())
sm.to_csv(OUT / "volume_summary.csv")
print(sm.to_string())
