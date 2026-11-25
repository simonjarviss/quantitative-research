"""Step 2: visible passive share pi per index family (tracker ETF AUM / official index MV, Datastream).

Method as in raw/audits/passive_share_estimates.md: AUM = wrds_ds2dsf.mktcap (close x numshrs) of the physical
headline trackers; MV = ds2indexdata.mv (level x divisor = float-adjusted index cap, millions).
Several tracker share counts are frozen in Datastream before 2014-17 (e.g. Xtrackers DAX 300,000 shares until 2017,
Amundi Core STOXX 600 50,000 until 2017), so the early series understate pi. The study therefore uses ONE pi per family:
the median daily value over 2023-01-01 .. 2026-06-23, when every tracker updates. For the Russell 2000, S&P/TSX
Composite and STOXX 600 the flow_sizing.md §4.4 values (2026-06-23) are used, and the R2 median is shown beside them.
Output: table_pi.csv (family, pi used, R2 median 2023-26, yearly medians).
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, PI_CONST

a = pd.read_parquet(DATA / "etf_aum.parquet")
a["marketdate"] = pd.to_datetime(a.marketdate)
m = pd.read_parquet(DATA / "index_mv.parquet")
m["valuedate"] = pd.to_datetime(m.valuedate)
# CHF line of an Amundi DAX ETF: convert with its own mktcap_usd and the EUR lines' USD rate on the day
eur_usd = a[a.currency == "EUR"].assign(r=lambda d: d.mktcap / d.mktcap_usd).groupby("marketdate").r.median()
a["aum_local"] = np.where(a.currency == "CHF", a.mktcap_usd * a.marketdate.map(eur_usd), a.mktcap)
aum = a.groupby(["family", "marketdate"]).aum_local.sum().reset_index()
x = aum.merge(m.rename(columns={"valuedate": "marketdate"})[["family", "marketdate", "mv"]], on=["family", "marketdate"])
x["pi"] = x.aum_local / (x.mv * 1e6)
x["y"] = x.marketdate.dt.year
yr = x.groupby(["family", "y"]).pi.median().unstack(0)
rec = x[(x.marketdate >= "2023-01-01") & (x.marketdate <= "2026-06-23")].groupby("family").pi.median()
last = x.sort_values("marketdate").groupby("family").tail(1).set_index("family").pi
rows = []
for fam in ["TSX", "STOXX", "DAX", "MDAX", "SDAX", "TecDAX"]:
    used = {"TSX": PI_CONST["TSX"], "STOXX": PI_CONST["STOXX"]}.get(fam, rec[fam])
    rows.append(dict(family=fam, pi_used=used, pi_r2_median_2023_26=rec[fam], pi_r2_last=last[fam],
                     source="flow_sizing.md §4.4 (2026-06-23)" if fam in ("TSX", "STOXX") else "R2 median 2023-01..2026-06",
                     **{f"pi_{y}": yr.loc[y, fam] if y in yr.index else np.nan for y in range(2010, 2027, 2)}))
rows.append(dict(family="R2000", pi_used=PI_CONST["R2000"], pi_r2_median_2023_26=np.nan, pi_r2_last=np.nan,
                 source="flow_sizing.md §4.4 (IWM+VTWO / holdings-implied cap, 2026-06-23)"))
t = pd.DataFrame(rows)
t.to_csv(OUT / "table_pi.csv", index=False)
pd.set_option("display.width", 250)
print(t[["family", "pi_used", "pi_r2_median_2023_26", "pi_r2_last", "source"]].to_string(index=False))
