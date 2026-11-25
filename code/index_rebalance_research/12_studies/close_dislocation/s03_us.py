"""Step 3: US Russell June reconstitution events 2008-2025 -> one row per actual change with every leg, flow and
crowding variable (data/legs_us.parquet).

Events: russell_june/output/events_actual.csv (valid rows; BUY_add, BUY_mig, SELL_del, SELL_mig), trade line = the
company's largest-cap permno on T. Side +1 = index buying (BUY_*), -1 = index selling (SELL_*).
Controls: the sister study's ACT-P cells (russell_june/_cache/cells_<y>.parquet, portfolio ACT-P): companies ranked
600-5000 at rank day, not predicted at rank day and not changed that year; cell = size decile x 6-month momentum
quintile formed at the preliminary-list close; a cell with < 3 controls falls back to the size decile.
Index: IWM (CRSP permno 88222). Returns: CRSP dsf_v2 dlyret (delisting-aware), clipped to [-50%, +100%].
Opens: data/opens_us.parquet (dsf_v2 dlyopen / dlyclose on T-1, T, T+1).
Flow: D = pi_R2000 x CRSP float shares (msf mthfloatshrqty, end-May; shares outstanding if missing);
      ADV60 = median daily volume over T-65..T-6; flow/ADV = D / ADV60. Realised: V_T / ADV60 and
      the sister-study multiple V_T / median(V over T-20..T-1).
Crowding (all dated before T; the pre-positioning ones before the preliminary list P):
  abto      log(mean turnover T-126..P / mean turnover T-252..T-127) minus the same for the year's ACT-P controls (median)
  runup_h3  abnormal return end-Mar -> P (ACT-H3 windows H3_HR + HR_P of car_stock_level.csv.gz; BHAR, summed)
  runup_h6  abnormal return end-Dec -> P (ACT-H6 H6_H3 + H3_HR + HR_P)
  pod_mar   14 pods' combined 13F holdings at the March quarter-end, % of shares; pod_chg = Mar minus Dec
  si_pre    short interest at the last settlement date on or before P, % of shares; si_chg = minus ~3 months earlier
"""
import importlib.util
import numpy as np
import pandas as pd
import duckdb
from common import DATA, RUS, IWM_PERMNO, CLIP, PI_CONST
from legs import KS, K0, daily_legs, split_session, subday_legs, flatten

spec = importlib.util.spec_from_file_location("russell_common", RUS / "common.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
CAL, PODS = rc.CAL, rc.PODS
SIDE = {"BUY_add": 1, "BUY_mig": 1, "SELL_del": -1, "SELL_mig": -1}
PI = PI_CONST["R2000"]

con = duckdb.connect()
con.execute(f"CREATE VIEW dsf AS SELECT * FROM read_parquet('{RUS}/_cache/dsf/*.parquet')")
td = con.execute("SELECT dlycaldt, count(*) n FROM dsf GROUP BY 1 HAVING count(*) > 1000 ORDER BY 1").df()
TD = pd.DatetimeIndex(pd.to_datetime(td.dlycaldt))


def tday(d):
    return TD[TD.searchsorted(pd.Timestamp(d), side="right") - 1]


EV = pd.read_csv(RUS / "output" / "events_actual.csv")
EV = EV[EV.valid & EV.group.isin(SIDE) & EV.permno.notna()].copy()
EV["permno"] = EV.permno.astype(int)
OP = pd.read_parquet(DATA / "opens_us.parquet")
OP["dlycaldt"] = pd.to_datetime(OP.dlycaldt)
MSF = pd.read_parquet(RUS / "_cache" / "msf.parquet", columns=["permno", "mthcaldt", "floatsh", "shrout", "cusip", "hdrcusip"])
MSF["ym"] = MSF.mthcaldt.dt.to_period("M")
CSL = pd.read_csv(RUS / "output" / "car_stock_level.csv.gz",
                  usecols=["portfolio", "year", "window", "permco", "permno", "ar"])
CSL = CSL[CSL.portfolio.isin(["ACT-H3", "ACT-H6"])]
run = CSL.pivot_table(index=["year", "permco", "portfolio"], columns="window", values="ar", aggfunc="first").reset_index()
r3 = run[run.portfolio == "ACT-H3"].assign(runup_h3=lambda d: d.H3_HR + d.HR_P)[["year", "permco", "runup_h3"]]
r6 = run[run.portfolio == "ACT-H6"].assign(runup_h6=lambda d: d.H6_H3 + d.H3_HR + d.HR_P)[["year", "permco", "runup_h6"]]

# ---- 13F pods: combined holdings by permno and quarter-end
h = pd.read_parquet(RUS / "_cache" / "pods13f.parquet")
h["ym"] = h.fdate.dt.to_period("M")
cm = pd.concat([MSF[["permno", "ym", "cusip"]], MSF[["permno", "ym", "hdrcusip"]].rename(columns={"hdrcusip": "cusip"})])
cm = cm.dropna().drop_duplicates(["ym", "cusip"])
h = h.merge(cm, on=["ym", "cusip"], how="inner")
POD = h.groupby(["ym", "permno"]).shares.sum()
SO = MSF.set_index(["ym", "permno"]).shrout * 1000

# ---- short interest by permno
si = pd.read_parquet(RUS / "_cache" / "shortint.parquet")
ccm = pd.read_parquet(RUS / "_cache" / "ccm.parquet")
si = si.merge(ccm, left_on=["gvkey", "iid"], right_on=["gvkey", "liid"])
si = si[(si.datadate >= si.linkdt) & (si.datadate <= si.linkenddt.fillna(pd.Timestamp("2099-12-31")))]
si["permno"] = si.permno.astype("int64")
si = si.drop_duplicates(["permno", "datadate"]).sort_values("datadate")


def valid_open(o):
    """CRSP open usable: a positive open and a positive trade close on the same day."""
    return (o.dlyopen > 0) & (o.dlyclose > 0)


rows = []
for y in CAL.index:
    T = CAL.loc[y, "T"]; iT = TD.get_loc(T)
    P = tday(CAL.loc[y, "prelim"]); iP = TD.get_loc(P)
    lo, hi = iT - 260, min(iT + 21, len(TD) - 1)
    days = TD[lo:hi + 1]
    d = con.execute(f"""SELECT permno, dlycaldt, ret, prc, vol, shrout FROM dsf
                        WHERE dlycaldt BETWEEN DATE '{days[0].date()}' AND DATE '{days[-1].date()}'""").df()
    d["dlycaldt"] = pd.to_datetime(d.dlycaldt)
    R = d.pivot_table(index="dlycaldt", columns="permno", values="ret", aggfunc="first").reindex(days)
    V = d.pivot_table(index="dlycaldt", columns="permno", values="vol", aggfunc="first").reindex(days)
    SH = d.pivot_table(index="dlycaldt", columns="permno", values="shrout", aggfunc="first").reindex(days).ffill()
    PR = d.pivot_table(index="dlycaldt", columns="permno", values="prc", aggfunc="first").reindex(days).abs()
    Rc = R.clip(*CLIP)
    k0 = iT - lo                                     # row of T in the panel
    win = Rc.iloc[k0 - 5:k0 + 21]                   # k = -5..20 (may be short at the sample end)
    win = win.reindex(TD[iT - 5:iT + 21])
    # sub-day pieces for every permno
    o = OP[OP.dlycaldt.isin([T, TD[iT + 1]])].copy()
    oT = o[o.dlycaldt == T].set_index("permno").dlyclose
    oT1 = o[o.dlycaldt == TD[iT + 1]].set_index("permno").dlyopen
    eqprev = (oT1.reindex(oT.index) == oT)
    o = o[valid_open(o)]
    sub = {}
    for key, day in [("0", T), ("1", TD[iT + 1])]:
        oo = o[o.dlycaldt == day].set_index("permno")
        ret_day = R.loc[day].reindex(oo.index)
        on, idr = split_session(ret_day.values, oo.dlyopen.values, oo.dlyclose.values)
        sub["on" + key] = pd.Series(on, index=oo.index).clip(*CLIP)
        sub["id" + key] = pd.Series(idr, index=oo.index).clip(*CLIP)
    # cells (ACT-P)
    cells = pd.read_parquet(RUS / "_cache" / f"cells_{y}.parquet")
    cells = cells[cells.portfolio == "ACT-P"]
    ctrl = cells[cells.role == "control"].copy(); ctrl["permno"] = ctrl.permno.astype(int)
    evc = cells[cells.role == "event"].copy(); evc["permno"] = evc.permno.astype(int)
    full = ctrl.groupby(["sz", "mq"]).permno.apply(list).to_dict()
    szonly = ctrl.groupby("sz").permno.apply(list).to_dict()
    # crowding: abnormal turnover vs this year's controls
    TO = V / (SH * 1000)
    base = TO.iloc[k0 - 252:k0 - 126]
    pre = TO.iloc[k0 - 126:(iP - lo) + 1]
    nb, npre = base.notna().sum(), pre.notna().sum()
    lr = np.log(pre.mean() / base.mean()).where((nb >= 60) & (npre >= 40))
    lr = lr.replace([np.inf, -np.inf], np.nan)
    ctrl_med = lr.reindex(ctrl.permno).median()
    # volume
    adv60 = V.iloc[k0 - 65:k0 - 5].median().where(V.iloc[k0 - 65:k0 - 5].notna().sum() >= 20)
    med20 = V.iloc[k0 - 20:k0].median()
    advb = V.iloc[k0 - 252:k0 - 126].median().where(V.iloc[k0 - 252:k0 - 126].notna().sum() >= 60)
    vT = V.iloc[k0]
    # index
    xi = win[IWM_PERMNO].values if IWM_PERMNO in win.columns else np.full(len(KS), np.nan)
    ixs = {k: sub[k].get(IWM_PERMNO, np.nan) for k in sub}
    # float, 13F, SI
    ym_may = pd.Period(f"{y}-05", "M")
    fl = MSF[MSF.ym == ym_may].set_index("permno")
    qm, qd = pd.Period(f"{y}-03", "M"), pd.Period(f"{y-1}-12", "M")
    sib = si[si.datadate <= P].groupby("permno").tail(1).set_index("permno")
    sia = si[si.datadate <= P - pd.Timedelta(days=90)].groupby("permno").tail(1).set_index("permno")
    e = EV[EV.year == y].merge(evc[["permno", "sz", "mq", "ccap", "mom"]], on="permno", how="left")
    for r in e.itertuples():
        p = r.permno
        if p not in win.columns:
            continue
        if pd.notna(r.sz):
            cl = full.get((r.sz, r.mq), [])
            if len(cl) < 3:
                cl = szonly.get(r.sz, [])
            cell_lvl = "sz_mom" if len(full.get((r.sz, r.mq), [])) >= 3 else "sz"
        else:
            cl, cell_lvl = [], "none"
        cl = [c for c in cl if c in win.columns and c != p]
        cmean = win[cl].mean(axis=1).values if cl else np.full(len(KS), np.nan)
        rr = win[p].values
        legs = daily_legs(rr, cmean, xi)
        s = {k: sub[k].get(p, np.nan) for k in sub}
        cms = {k: sub[k].reindex(cl).mean() if cl else np.nan for k in sub}
        legs.update(subday_legs(s, cms, ixs))
        shr = SH[p].iloc[k0 - 1] * 1000 if p in SH.columns else np.nan
        flt = fl.floatsh.get(p, np.nan) * 1000 if p in fl.index else np.nan
        F = flt if np.isfinite(flt) and flt > 0 else shr
        px = PR[p].iloc[k0 - 1] if p in PR.columns else np.nan
        row = dict(market="US", family="R2000", event_date=T.date(), review=str(y), year=y, id=str(p), permco=r.permco,
                   name="", cty="USA", group=r.group, move=r.group, side=SIDE[r.group], cell=cell_lvl, n_ctrl=len(cl),
                   price=px, shares=shr, float_sh=F, float_src="crsp" if np.isfinite(flt) and flt > 0 else "shrout",
                   pi=PI, demand_sh=PI * F if np.isfinite(F) else np.nan,
                   adv60=adv60.get(p, np.nan), adv_base=advb.get(p, np.nan), vol_T=vT.get(p, np.nan), med20=med20.get(p, np.nan), vol_basis="CRSP consolidated",
                   abto=lr.get(p, np.nan) - ctrl_med if np.isfinite(ctrl_med) else np.nan, abto_raw=lr.get(p, np.nan),
                   ann_date=P.date(), d0_ret_is_zero=bool(np.isfinite(rr[K0]) and rr[K0] == 0.0),
                   open_T1_ok=bool(np.isfinite(s["on1"])), open_T_ok=bool(np.isfinite(s["on0"])),
                   open_T1_eqprev=bool(eqprev.get(p, False)))
        so_m, so_d = SO.get((qm, p), np.nan), SO.get((qd, p), np.nan)
        row["pod_mar"] = POD.get((qm, p), 0.0) / so_m * 100 if so_m > 0 else np.nan
        pod_dec = POD.get((qd, p), 0.0) / so_d * 100 if so_d > 0 else np.nan
        row["pod_chg"] = row["pod_mar"] - pod_dec
        row["si_pre"] = sib.shortint.get(p, np.nan) / shr * 100 if shr > 0 else np.nan
        si_old = sia.shortint.get(p, np.nan) / shr * 100 if shr > 0 else np.nan
        row["si_chg"] = row["si_pre"] - si_old
        row.update(flatten(legs))
        rows.append(row)
    print(y, len(rows), flush=True)

L = pd.DataFrame(rows)
L = L.merge(r3.rename(columns={"permco": "permco"}), on=["year", "permco"], how="left")
L = L.merge(r6, on=["year", "permco"], how="left")
L.to_parquet(DATA / "legs_us.parquet", index=False)
print(L.groupby(["group"]).size())
