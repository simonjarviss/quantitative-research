"""Step 4: S&P/TSX Composite quarterly-review events 2006 - Jun 2026 -> data/legs_ca.parquet (one row per change).

Events: tsx_composite/events.csv, etype == 'review' (389 adds, 275 deletes). Side +1 add, -1 delete.
Controls: the sister study's cells (06_returns.py cells_at) re-formed at the reference date R: Canadian, GICS-classified,
non-Venture TSX lines priced in the last 10 sessions, close >= C$1, cap >= 0.005% of index float cap, >= 60 sessions of
history, excluding every name predicted (any horizon, any variant) or changed at that review; cell = size decile x
6-month momentum tercile (>= 5 controls), else size decile.
Index: the sister cap-weighted Composite-member total-return proxy for close-to-close days; for the open-based pieces a
cap-weighted mean of members (T-1 caps) with a usable open.
Returns: panel RET (Compustat prccd x trfd / ajexdi), clipped to [-50%, +100%]. Opens: data/opens_ca.parquet
(Compustat secd prcod); an open is usable if prcod > 0 and the session traded (prcstd != 4).
Flow: D = pi_TSX (1.00%) x float shares (shares x Datastream/Worldscope float, as the sister study);
ADV60 = median consolidated daily volume (Datastream Canadian Composite) over T-65..T-6, TSX-only where consolidated
is missing (before ~2014). Realised: V_T / ADV60 and the sister multiple V_T / median(V T-20..T-1), same basis.
Crowding (before T; pre-positioning ones before the announcement A):
  abto      log(mean TSX turnover T-126..A-1 / mean T-252..T-127) minus the review's control median
  runup_h3  CAR H3 -> A (sister car_events.csv, hindsight portfolio formed at H3: car_H3_R + car_R_A)
  runup_h6  CAR H6 -> A (formed at H6: car_H6_H3 + car_H3_R + car_R_A)
  pod_q     14 pods' combined 13F holdings at the last quarter-end before T, % of shares (interlisted names only)
"""
import numpy as np
import pandas as pd
from common import DATA, TSX, CLIP, PI_CONST
from legs import KS, K0, daily_legs, split_session, subday_legs, flatten

PI = PI_CONST["TSX"]
D_ = TSX / "data"
cal = pd.DatetimeIndex(pd.read_parquet(D_ / "tsx_calendar.parquet").date)
rev = pd.read_csv(TSX / "reviews.csv", parse_dates=["H6", "H3", "R", "A", "T", "eff"])
z = np.load(D_ / "panel.npz")
P, RET, VT, VC, SH, AJ = z["P"], z["RET"], z["VT"], z["VC"], z["SH"], z["AJ"]
AJf = pd.DataFrame(AJ).ffill().values
nd, nl = P.shape
L = pd.read_parquet(D_ / "panel_lines.parquet")
lid_ix = pd.Index(L.lid)
MCAP = P * SH / AJf
R_cl = np.clip(RET, *CLIP)
TRI = np.nancumprod(np.nan_to_num(RET) + 1, axis=0)
dom = ((L["loc"] == "CAN") | (L.fic == "CAN")).values & L.gsubind.notna().values & (L.exchg != 9).values
MEM = np.load(D_ / "member_matrix.npy")
mkt = pd.read_parquet(D_ / "index_proxy.parquet").proxy_tr.fillna(0).values
TO = VT / SH

ev = pd.read_csv(TSX / "events.csv", parse_dates=["eff_date", "T"])
ev = ev[ev.etype == "review"].copy()
ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
ev["col"] = lid_ix.get_indexer(ev.lid)
pr = pd.read_csv(TSX / "predictions_final.csv")
diag = pd.read_csv(TSX / "prediction_diagnostics.csv").set_index(["review", "horizon"])
fm = pd.read_parquet(D_ / "float_monthly.parquet")
fm["ff"] = fm.ff_ds.fillna(fm.ff_ws).fillna(1.0)
FF = fm.pivot_table(index="month", columns="line", values="ff")

# opens
o = pd.read_parquet(DATA / "opens_ca.parquet")
o["datadate"] = pd.to_datetime(o.datadate)
o["lid"] = o.gvkey + "_" + o.iid
o["col"] = lid_ix.get_indexer(o.lid)
o = o[(o.col >= 0) & o.datadate.isin(cal)]
o["pos"] = cal.get_indexer(o.datadate)
o = o.sort_values(["col", "datadate"])
OPEN = np.full((nd, nl), np.nan, dtype=np.float64)
CLOSE = np.full((nd, nl), np.nan, dtype=np.float64)
ok = (o.prcod > 0) & (o.prccd > 0) & (o.prcstd.astype(float) != 4)
OPEN[o.pos[ok].values, o.col[ok].values] = o.prcod[ok].values
CLOSE[o.pos[ok].values, o.col[ok].values] = o.prccd[ok].values
FLAG_EQPREV = np.zeros((nd, nl), bool)        # open exactly equal to the previous session's close
prev = P[np.clip(o.pos.values - 1, 0, None), o.col.values]
FLAG_EQPREV[o.pos.values, o.col.values] = (o.prcod.values == prev)

# runups from the sister file (hindsight portfolios formed at H3 / H6)
ce = pd.read_csv(TSX / "car_events.csv", usecols=["review", "form", "portfolio", "lid", "tic", "car_H6_H3", "car_H3_R", "car_R_A"])
ce = ce[ce.portfolio.isin(["hind_add", "hind_del"])]
ru3 = ce[ce.form == "H3"].assign(runup_h3=lambda d: d.car_H3_R + d.car_R_A).drop_duplicates(["review", "lid"]).set_index(["review", "lid"]).runup_h3
ru6 = ce[ce.form == "H6"].assign(runup_h6=lambda d: d.car_H6_H3 + d.car_H3_R + d.car_R_A).drop_duplicates(["review", "lid"]).set_index(["review", "lid"]).runup_h6
tic2lid = ce.drop_duplicates(["review", "tic"]).set_index(["review", "tic"]).lid
pods = pd.read_csv(TSX / "pods_13f.csv")
pods = pods[(pods.qe == -1) & pods.portfolio.isin(["hind_add", "hind_del"])]
pods["lid"] = [tic2lid.get((r, t)) for r, t in zip(pods.review, pods.tic)]
POD = pods.dropna(subset=["lid"]).drop_duplicates(["review", "lid"]).set_index(["review", "lid"]).pod_pct


def cells_at(f, Mref, exclude):
    """06_returns.py cells_at, unchanged."""
    live = (~np.isnan(P[max(0, f - 9):f + 1])).any(0)
    mc = MCAP[f]
    mc = np.where(np.isnan(mc), pd.DataFrame(MCAP[max(0, f - 9):f + 1]).ffill().values[-1], mc)
    mom = TRI[f] / TRI[max(0, f - 126)] - 1
    ok_hist = (~np.isnan(P[max(0, f - 126):f + 1])).sum(0) >= 60
    univ = live & dom & (P[f] >= 1) & (mc >= 0.00005 * Mref) & ok_hist
    ctrl = univ & ~exclude
    sb = np.nanquantile(mc[univ], np.linspace(0.1, 0.9, 9))
    mb = np.nanquantile(mom[univ], [1 / 3, 2 / 3])
    sdec = np.searchsorted(sb, np.nan_to_num(mc, nan=0.0))
    mter = np.searchsorted(mb, np.nan_to_num(mom, nan=0.0))
    return sdec * 3 + mter, sdec, ctrl


def subday(iT):
    """per-line overnight / intraday pieces on T (key 0) and T+1 (key 1); NaN where the open is unusable."""
    out = {}
    for key, i in [("0", iT), ("1", iT + 1)]:
        on, idr = split_session(RET[i], OPEN[i], CLOSE[i])
        out["on" + key] = np.clip(on, *CLIP)
        out["id" + key] = np.clip(idr, *CLIP)
    return out


rows = []
for q in rev.itertuples():
    e = ev[(ev.review == q.review) & (ev.col >= 0)]
    if e.empty:
        continue
    iT = cal.get_loc(q.T); iR = cal.get_loc(q.R); iA = cal.get_loc(q.A)
    if iT + 1 >= nd:
        continue
    x = pr[pr.review == q.review]
    excl = np.zeros(nl, bool)
    excl[e.col.values] = True
    excl[x.loc[x.pred_add_rules | x.pred_del_rules | x.pred_add | x.pred_del, "col"].values] = True
    Mref = diag.loc[(q.review, "R"), "M_bn"] * 1e9
    code, sdec, ctrl = cells_at(iR, Mref, excl)
    ks = np.arange(iT - 5, min(iT + 21, nd))
    Rw = np.full((len(KS), nl), np.nan); Rw[:len(ks)] = R_cl[ks]
    xw = np.full(len(KS), np.nan); xw[:len(ks)] = mkt[ks]
    sub = subday(iT)
    # index sub-day pieces: cap-weighted members (T-1 caps) with usable opens
    w = np.where(MEM[iT - 1], np.nan_to_num(MCAP[iT - 1]), 0.0)
    ixs = {}
    for k, v in sub.items():
        m_ = np.isfinite(v) & (w > 0)
        ixs[k] = np.sum(w[m_] * v[m_]) / np.sum(w[m_]) if m_.any() else np.nan
    # abnormal turnover (TSX-only volume), baseline months -12..-6, window T-126..A-1
    base = np.nanmean(TO[iT - 252:iT - 126], 0); nb = (~np.isnan(TO[iT - 252:iT - 126])).sum(0)
    pre = np.nanmean(TO[iT - 126:iA], 0); npre = (~np.isnan(TO[iT - 126:iA])).sum(0)
    lr = np.where((nb >= 60) & (npre >= 40) & (base > 0) & (pre > 0), np.log(pre / np.where(base > 0, base, 1)), np.nan)
    ctrl_med = np.nanmedian(lr[ctrl])
    month = pd.Timestamp(q.T.year, q.T.month, 1)
    ffrow = FF.loc[month] if month in FF.index else pd.Series(dtype=float)
    for r in e.itertuples():
        c = r.col
        mbr = ctrl & (code == code[c]); mbr[c] = False
        lvl = "sz_mom"
        if mbr.sum() < 5:
            mbr = ctrl & (sdec == sdec[c]); mbr[c] = False; lvl = "sz"
        cm = np.nanmean(Rw[:, mbr], 1)
        legs = daily_legs(Rw[:, c], cm, xw)
        s = {k: v[c] for k, v in sub.items()}
        cms = {k: np.nanmean(v[mbr]) for k, v in sub.items()}
        legs.update(subday_legs(s, cms, ixs))
        vc = VC[iT - 65:iT - 5, c]; vt = VT[iT - 65:iT - 5, c]
        cons = (~np.isnan(vc)).sum() >= 20
        Vb = VC if cons else VT
        adv60 = np.nanmedian(Vb[iT - 65:iT - 5, c]) if (~np.isnan(Vb[iT - 65:iT - 5, c])).sum() >= 20 else np.nan
        vb = Vb[iT - 252:iT - 126, c]
        advb = np.nanmedian(vb) if (~np.isnan(vb)).sum() >= 60 else np.nan
        ff = ffrow.get(c, 1.0) if len(ffrow) else 1.0
        ff = 1.0 if not np.isfinite(ff) else ff
        shr = SH[iT - 1, c]                     # split-adjusted shares (same units as the volume matrices)
        F = shr * ff
        rows.append(dict(
            market="CA", family="TSX", event_date=q.T.date(), review=q.review, year=q.year, id=r.lid, permco=np.nan,
            name=r.conm, cty="CAN", group="add" if r.side == "add" else "delete", move=r.side,
            side=1 if r.side == "add" else -1, cell=lvl, n_ctrl=int(mbr.sum()),
            price=P[iT - 1, c], shares=shr / AJf[iT - 1, c], float_sh=F / AJf[iT - 1, c], float_src="ds/ws",
            pi=PI, demand_sh=PI * F / AJf[iT - 1, c],
            adv60=adv60 / AJf[iT - 1, c], adv_base=advb / AJf[iT - 1, c], vol_T=Vb[iT, c] / AJf[iT - 1, c],
            med20=np.nanmedian(Vb[iT - 20:iT, c]) / AJf[iT - 1, c],
            vol_basis="CA consolidated" if cons else "TSX only",
            abto=lr[c] - ctrl_med, abto_raw=lr[c], ann_date=q.A.date(),
            d0_ret_is_zero=bool(np.isfinite(RET[iT, c]) and RET[iT, c] == 0.0),
            open_T1_ok=bool(np.isfinite(s["on1"])), open_T_ok=bool(np.isfinite(s["on0"])),
            open_T1_eqprev=bool(FLAG_EQPREV[iT + 1, c]),
            runup_h3=ru3.get((q.review, r.lid), np.nan), runup_h6=ru6.get((q.review, r.lid), np.nan),
            pod_q=POD.get((q.review, r.lid), np.nan), new_listing=bool(r.new_listing), **flatten(legs)))
    print(q.review, len(rows), flush=True)

Lg = pd.DataFrame(rows)
Lg.to_parquet(DATA / "legs_ca.parquet", index=False)
print(Lg.groupby(["group"]).size(), Lg.vol_basis.value_counts())
