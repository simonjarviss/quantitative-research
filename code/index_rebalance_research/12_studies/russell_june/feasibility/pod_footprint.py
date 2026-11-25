"""Pod 13F footprint in June-2025 Russell 2000 adds (new to R3000) vs R2000 stayers, by 13F quarter."""
import pandas as pd
import r2

PODS = [57986, 16210, 7808, 14335, 26244, 75106, 10736, 11912, 12274, 10643, 8853, 22891, 10751, 12759, 28088]
with r2.session() as con:
    m = r2.sql(con, """
        WITH s AS (SELECT DISTINCT indexlistintcode::INT code, date_, infocode FROM tr_ds_equities.wrds_ds_indexconstmerged_mth
                   WHERE indexlistintcode IN (6967, 6968) AND date_ IN (DATE '2025-05-30', DATE '2025-06-30')),
        n AS (SELECT DISTINCT infocode, substr(dslocalcode, 2, 8) cusip8 FROM tr_ds_equities.wrds_ds_names
              WHERE infocode IN (SELECT infocode FROM s) AND dslocalcode LIKE 'U%' AND enddate >= DATE '2025-05-01')
        SELECT s.code, s.date_, s.infocode, n.cusip8 FROM s JOIN n USING (infocode)""")
    pre = set(m.loc[m.date_.astype(str) == "2025-05-30", "cusip8"])
    post2 = set(m.loc[(m.code == 6968) & (m.date_.astype(str) == "2025-06-30"), "cusip8"])
    pre2 = set(m.loc[(m.code == 6968) & (m.date_.astype(str) == "2025-05-30"), "cusip8"])
    adds, stay = post2 - pre, post2 & pre2
    ids = pd.DataFrame({"cusip8": list(adds | stay)})
    ids["grp"] = ids.cusip8.map(lambda c: "add" if c in adds else "stayer")
    con.register("ids", ids)
    h = r2.sql(con, f"""
        SELECT s.fdate, s.cusip, sum(s.shares) pod_sh, count(DISTINCT s.mgrno) n_pods, max(s.shrout2) shrout2_k
        FROM tr_13f.s34 s
        WHERE s.fdate IN (DATE '2024-09-30', DATE '2024-12-31', DATE '2025-03-31', DATE '2025-06-30', DATE '2025-09-30')
          AND s.mgrno IN ({",".join(map(str, PODS))}) AND s.cusip IN (SELECT cusip8 FROM ids)
        GROUP BY ALL""")
    so = r2.sql(con, """SELECT substr(cusip, 1, 8) cusip8, sum(shrout::DOUBLE) * 1000 shr FROM crsp_a_stock.msf_v2
                        WHERE mthcaldt = DATE '2025-04-30' AND cusip IS NOT NULL GROUP BY 1""")
rows = []
for fd in sorted(h.fdate.unique()):
    x = ids.merge(h[h.fdate == fd], left_on="cusip8", right_on="cusip", how="left").merge(so, on="cusip8", how="left")
    x["own"] = (x.pod_sh.fillna(0) / x.shr) * 100
    for g, y in x[x.shr.notna()].groupby("grp"):
        rows.append(dict(fdate=str(fd)[:10], grp=g, n=len(y), held_by_any_pod=round((y.n_pods.fillna(0) > 0).mean(), 3),
                         mean_n_pods=round(y.n_pods.fillna(0).mean(), 2), median_pod_own_pct=round(y.own.median(), 2),
                         mean_pod_own_pct=round(y.own.mean(), 2)))
print(pd.DataFrame(rows).to_string(index=False))
