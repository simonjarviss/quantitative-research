"""Step 10 (optional): combined 13F holdings of multi-manager / quant pods around each event.

Interlisted (US-exchange-listed) names only: 13F covers US 13(f) securities. Holdings at quarter-ends
qe-2, qe-1 (before T), qe0 (first quarter-end after T, ~10 days after T) and qe+1, as % of shares
outstanding (13F shrout1, else Compustat). Long positions only; swaps and Canadian-line holdings invisible.
Pods (mgrno): Millennium 57986, Citadel Advisors 16210, Balyasny 7808, Schonfeld 14335, ExodusPoint 26244,
Point72 75106, Jane Street 10736, Hudson Bay 11912, Verition 12274, Marshall Wace 10643, Squarepoint 8853,
Walleye 22891/23308/17779, Two Sigma 10751/12759, Qube 28088.
Output: pods_13f.csv (per name x quarter), table_pods_13f.csv.
"""
import numpy as np, pandas as pd
import r2
from common import DATA, OUT, tsx_calendar, review_dates

PODS = [57986, 16210, 7808, 14335, 26244, 75106, 10736, 11912, 12274, 10643, 8853, 22891, 23308, 17779, 10751, 12759, 28088]
f = DATA / "s34_pods.parquet"
if not f.exists():
    with r2.session() as con:
        x = r2.sql(con, f"""SELECT fdate, rdate, mgrno::BIGINT AS mgrno, cusip, shares, shrout1, prc
                           FROM tr_13f.s34 WHERE mgrno IN ({",".join(map(str, PODS))}) AND fdate >= DATE '2005-06-30' AND fdate = rdate""")
    x.to_parquet(f, index=False)
s = pd.read_parquet(f)
s["rdate"] = pd.to_datetime(s.rdate)
L = pd.read_parquet(DATA / "panel_lines.parquet")
c8 = dict(zip(L.lid, L.cusip.str[:8]))
cal = tsx_calendar(); rev = review_dates(cal)
ce = pd.read_csv(OUT / "car_events.csv")
pr = pd.read_csv(OUT / "predictions_final.csv")
dual = pr[pr.horizon == "R"].set_index(["review", "lid"]).dual.to_dict()
grp = ce[(ce.form == "R") & ce.portfolio.isin(["hind_add", "hind_del", "pred_add", "pred_del"])][["review", "portfolio", "lid", "tic", "hit"]]
agg = s.groupby(["rdate", "cusip"]).agg(pod_shares=("shares", "sum"), n_pods=("mgrno", "nunique"), shrout1=("shrout1", "max")).reset_index()
qes = pd.DatetimeIndex(sorted(s.rdate.unique()))
z = np.load(DATA / "panel.npz"); SH, AJ = z["SH"], z["AJ"]
AJf = pd.DataFrame(AJ).ffill().values
lid_ix = pd.Index(L.lid)
rows = []
for r in grp.itertuples():
    T = rev.loc[rev.review == r.review, "T"].iloc[0]
    i0 = qes.searchsorted(T)                     # first quarter-end after T
    cu = c8.get(r.lid)
    for k in [-2, -1, 0, 1]:
        if not 0 <= i0 + k < len(qes):
            continue
        qe = qes[i0 + k]
        a = agg[(agg.rdate == qe) & (agg.cusip == cu)]
        di = cal.searchsorted(qe, side="right") - 1
        c = lid_ix.get_loc(r.lid)
        sh = SH[di, c] / AJf[di, c]
        so = a.shrout1.iloc[0] * 1e6 if len(a) and a.shrout1.iloc[0] > 0 else sh
        rows.append(dict(review=r.review, portfolio=r.portfolio, tic=r.tic, hit=r.hit, interlisted=dual.get((r.review, r.lid), False),
                         qe=k, rdate=qe, pod_pct=(a.pod_shares.iloc[0] / so * 100) if len(a) else 0.0,
                         n_pods=int(a.n_pods.iloc[0]) if len(a) else 0))
d = pd.DataFrame(rows)
d = d[d.interlisted & (d.rdate <= "2025-12-31")]
d.to_csv(OUT / "pods_13f.csv", index=False)
t = d.pivot_table(index="portfolio", columns="qe", values="pod_pct", aggfunc=["mean", "median", "count"]).round(3)
t.columns = [f"{a} qe{b:+d}" for a, b in t.columns]
n = d.pivot_table(index="portfolio", columns="qe", values="n_pods", aggfunc="mean").round(2)
n.columns = [f"mean n_pods qe{b:+d}" for b in n.columns]
t = t.join(n)
t.to_csv(OUT / "table_pods_13f.csv")
print(t.T.to_string())
