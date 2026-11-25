"""Step 15 (added 2026-10-14): liquidity of the post-redesign S&P/TSX adds (s14), to judge the T close -> T+1 close
rebound net of costs. Datastream primary-quote (TSX) closing bid/ask, close, volume on T and T+1, and 60-day median
volume before T. Lines mapped gvkey/iid -> ISIN (tsx_composite/data/lines.parquet) -> Datastream infocode
(ds_names, isprimqt = 1). Output: table_tsx_costs.csv, tsx_adds_post2021_liquidity.csv
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # code/, where the r2 package lives
import r2
from common import OUT, HERE

TSX = HERE.parent / "tsx_composite" / "data"
E = pd.read_csv(OUT / "events_legs.csv", low_memory=False)
c = E[(E.market == "CA")].copy()
c["event_date"] = pd.to_datetime(c.event_date)
c = c[c.event_date >= "2021-10-18"].copy()
c[["gvkey", "iid"]] = c["id"].str.split("_", expand=True)
L = pd.read_parquet(TSX / "lines.parquet", columns=["gvkey", "iid", "isin"])
N = pd.read_parquet(TSX / "ds_names.parquet", columns=["infocode", "isin", "isprimqt"])
N = N[N.isprimqt == 1].drop_duplicates("isin")
c = c.merge(L, on=["gvkey", "iid"], how="left").merge(N[["isin", "infocode"]], on="isin", how="left")
codes = ",".join(str(int(x)) for x in c.infocode.dropna().unique())
with r2.session() as con:
    q = r2.sql(con, f"SELECT infocode, marketdate, close_, bid, ask, volume FROM tr_ds_equities.ds2primqtprc "
                    f"WHERE infocode IN ({codes}) AND marketdate >= DATE '2021-06-01'")
q["marketdate"] = pd.to_datetime(q.marketdate)
q = q.sort_values(["infocode", "marketdate"])
rows = []
for _, e in c.dropna(subset=["infocode"]).iterrows():
    s = q[q.infocode == e.infocode].set_index("marketdate")
    if e.event_date not in s.index:
        continue
    i = s.index.get_loc(e.event_date)
    pre = s.iloc[max(0, i - 60):i]
    def spr(r):
        return (r.ask - r.bid) / ((r.ask + r.bid) / 2) * 1e4 if r.bid > 0 and r.ask > r.bid else np.nan
    t0, t1 = s.iloc[i], s.iloc[i + 1] if i + 1 < len(s) else None
    rows.append(dict(id=e["id"], name=e["name"], group=e.group, event_date=e.event_date.date(), price=t0.close_,
                     spread_T_bp=spr(t0), spread_T1_bp=spr(t1) if t1 is not None else np.nan,
                     adv60_med_sh=pre.volume.median(), adv60_med_cad=(pre.volume * pre.close_).median(),
                     vol_T_x=t0.volume / pre.volume.median() if pre.volume.median() else np.nan,
                     s_r1_bp=e.s_r1 * 1e4))
x = pd.DataFrame(rows)
x.to_csv(OUT / "tsx_adds_post2021_liquidity.csv", index=False)
t = x.groupby("group").agg(n=("id", "size"), price_med=("price", "median"), spread_T_med_bp=("spread_T_bp", "median"),
                            spread_T1_med_bp=("spread_T1_bp", "median"), adv_med_cad_m=("adv60_med_cad", lambda v: v.median() / 1e6),
                            volT_x_med=("vol_T_x", "median"), s_r1_mean_bp=("s_r1_bp", "mean"), s_r1_med_bp=("s_r1_bp", "median"))
# does the rebound concentrate in the least liquid names?
a = x[x.group == "add"].copy()
a["spread_tercile"] = pd.qcut(a.spread_T_bp.rank(method="first"), 3, labels=["tight", "mid", "wide"])
t2 = a.groupby("spread_tercile", observed=True).agg(n=("id", "size"), spread_med=("spread_T_bp", "median"),
                                                   adv_med_cad_m=("adv60_med_cad", lambda v: v.median() / 1e6),
                                                   s_r1_mean_bp=("s_r1_bp", "mean"), s_r1_med_bp=("s_r1_bp", "median"))
pd.concat({"by group": t, "adds by closing-spread tercile": t2}).to_csv(OUT / "table_tsx_costs.csv")
pd.set_option("display.width", 250)
print("matched", len(x), "of", len(c)); print(t.round(1).to_string()); print(t2.round(1).to_string())
