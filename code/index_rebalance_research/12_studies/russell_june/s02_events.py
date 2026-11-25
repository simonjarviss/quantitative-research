"""Step 2: map Datastream Russell lists to CRSP companies (permco) and define the actual June events.

Outputs
  _cache/membership.parquet     ym, code (6967/6968), infocode, permno, permco  (every month-end 2007-06..2025-12)
  output/map_quality.csv        mapped / unmapped list members per snapshot used
  output/events_actual.csv      one row per (year, permco) actual change: group, pre/post status, trade permno
  output/recon_day_check.csv    closing-volume spike check of each year's T
"""
import numpy as np
import pandas as pd
from common import CACHE, OUT, CAL, R1000, R2000, snapshots
import lib

L = pd.read_parquet(CACHE / "ds_lists.parquet")
N = pd.concat([pd.read_parquet(CACHE / "ds_names.parquet"), pd.read_parquet(CACHE / "ds_names_full.parquet")])
S = pd.read_parquet(CACHE / "crsp_secinfo.parquet")
M = pd.read_parquet(CACHE / "msf.parquet", columns=["permno", "permco", "mthcaldt", "cusip", "hdrcusip", "cap"])

# ---- infocode -> cusip8 (all historical ISINs / local codes of the quote)
ic = pd.concat([
    N[["infocode"]].assign(c8=N["isin"].str[2:10]),
    N[["infocode"]].assign(c8=N["isin2"].str[2:10]),
    N[["infocode"]].assign(c8=N["dslocalcode"].where(N["dslocalcode"].str.startswith("U", na=False)).str[1:9]),
]).dropna().drop_duplicates()
ic = ic[ic.c8.str.len() == 8]
# ---- permno -> cusip8 (CRSP history)
pc = pd.concat([S[["permno"]].assign(c8=S.cusip.str[:8]), S[["permno"]].assign(c8=S.hdrcusip.str[:8]),
                M[["permno"]].assign(c8=M.cusip), M[["permno"]].assign(c8=M.hdrcusip)]).dropna().drop_duplicates()
cand = ic.merge(pc, on="c8").drop_duplicates(["infocode", "permno"])

L["ym"] = L.date_.dt.to_period("M")
M["ym"] = M.mthcaldt.dt.to_period("M")
alive = M[["permno", "permco", "ym", "cusip", "cap"]]
x = L.merge(cand[["infocode", "permno"]], on="infocode", how="left").merge(alive, on=["permno", "ym"], how="left")
x = x[x.permco.notna() | x.permno.isna()]
# prefer the permno whose CUSIP that month is one of the quote's CUSIPs, then the larger cap
x = x.merge(ic.rename(columns={"c8": "cusip"}).assign(cus_match=1), on=["infocode", "cusip"], how="left")
x["cus_match"] = x.cus_match.fillna(0)
x = x.sort_values(["code", "ym", "infocode", "cus_match", "cap"], ascending=[True, True, True, False, False])
x = x.drop_duplicates(["code", "ym", "infocode"])
mem = L[["code", "ym", "infocode"]].drop_duplicates().merge(x[["code", "ym", "infocode", "permno", "permco"]],
                                                              on=["code", "ym", "infocode"], how="left")
mem.assign(ym=mem.ym.astype(str)).to_parquet(CACHE / "membership.parquet", index=False)


def status(ym: pd.Period) -> pd.Series:
    """Company status at a month-end: R1000 beats R2000 if a permco is on both lists."""
    m = mem[(mem.ym == ym) & mem.permco.notna()]
    s = pd.Series("R2000", index=pd.Index(m.loc[m.code == R2000, "permco"].astype(int).unique(), name="permco"))
    s1 = m.loc[m.code == R1000, "permco"].astype(int).unique()
    s = pd.concat([s[~s.index.isin(s1)], pd.Series("R1000", index=pd.Index(s1, name="permco"))])
    return s


q = []
for ym in sorted(mem.ym.unique()):
    m = mem[mem.ym == ym]
    for code in (R1000, R2000):
        mm = m[m.code == code]
        q.append(dict(ym=str(ym), code=code, n=len(mm), unmapped=int(mm.permco.isna().sum())))
q = pd.DataFrame(q)
q.to_csv(OUT / "map_quality.csv", index=False)

# ---- actual events by year
rows, chk = [], []
for y in CAL.index:
    pre_d, post_d = snapshots(y)
    pre, post = status(pre_d.to_period("M")), status(post_d.to_period("M"))
    T = CAL.loc[y, "T"]
    tp1 = lib.tshift(T, 1)
    px = lib.daily(lib.tshift(T, -1), tp1, "permno, permco, dlycaldt, prc, cap")
    onT = px[px.dlycaldt == T].dropna(subset=["prc"])
    onT1 = px[px.dlycaldt == tp1].dropna(subset=["prc"])
    allp = pd.Index(pre.index.union(post.index))
    df = pd.DataFrame({"permco": allp})
    df["pre"] = df.permco.map(pre).fillna("out")
    df["post"] = df.permco.map(post).fillna("out")
    df = df[df.pre != df.post]
    df["group"] = np.select(
        [(df.pre == "out") & (df.post == "R2000"), (df.pre == "R1000") & (df.post == "R2000"),
         (df.pre == "R2000") & (df.post == "out"), (df.pre == "R2000") & (df.post == "R1000")],
        ["BUY_add", "BUY_mig", "SELL_del", "SELL_mig"], "OTHER")
    # trade permno = largest-cap class of the company on T
    big = onT.sort_values("cap", ascending=False).drop_duplicates("permco").set_index("permco").permno
    df["permno"] = df.permco.map(big)
    df["trades_T"] = df.permco.isin(onT.permco)
    df["trades_Tp1"] = df.permco.isin(onT1.permco)
    # recon deletes must still trade after T (else it is an M&A / delisting exit, not a recon sale)
    df["valid"] = np.where(df.group.str.startswith("SELL"), df.trades_Tp1 & df.trades_T, df.trades_T)
    df["year"] = y
    rows.append(df)
    # closing-volume spike: (a) R3000 members' total share volume; (b) median volume ratio of this year's
    # add/delete stocks (volume / own June median) -- (b) separates the recon from quad-witching days
    v = lib.daily(pd.Timestamp(y, 6, 1), pd.Timestamp(y, 7, 10), "permno, permco, dlycaldt, vol")
    tot = v[v.permco.isin(pre.index)].groupby("dlycaldt").vol.sum()
    evp = df.loc[df.group.isin(["BUY_add", "SELL_del"]) & df.valid, "permco"]
    ve = v[v.permco.isin(evp)].groupby(["permco", "dlycaldt"]).vol.sum().unstack()
    rel = ve.div(ve.median(axis=1), axis=0).median()
    chk.append(dict(year=y, T=T.date(), spike_day_total=tot.idxmax().date(), total_vol_T_over_june_median=round(tot.loc[T] / tot.median(), 2),
                    spike_day_events=rel.idxmax().date(), event_vol_T_over_own_median=round(rel.loc[T], 1),
                    second_highest_event_day_ratio=round(rel.drop(T).max(), 1),
                    pre_snap=pre_d.date(), post_snap=post_d.date(),
                    n_pre_R1000=int((pre == "R1000").sum()), n_pre_R2000=int((pre == "R2000").sum()),
                    n_post_R1000=int((post == "R1000").sum()), n_post_R2000=int((post == "R2000").sum())))
ev = pd.concat(rows, ignore_index=True)
ev.to_csv(OUT / "events_actual.csv", index=False)
chk = pd.DataFrame(chk)
chk.to_csv(OUT / "recon_day_check.csv", index=False)
print(chk.to_string(index=False))
print(ev[ev.valid].pivot_table(index="year", columns="group", values="permco", aggfunc="count").to_string())
print("invalid (no trade at T / T+1):", ev[~ev.valid].groupby("group").size().to_dict())
snaps = sorted({str(d.to_period("M")) for y in CAL.index for d in snapshots(y)})
print(q[q.ym.isin(snaps)].groupby("code")[["n", "unmapped"]].agg(["mean", "max"]).to_string())
