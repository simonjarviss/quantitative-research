"""Step 5: STOXX Europe 600 (Mar 2008 - Jun 2026) and DAX-family (Mar 2008 - Dec 2025) review events ->
data/legs_eu.parquet (one row per family x change).

Events: stoxx600_dax/events_stoxx.csv and events_dax.csv, etype == 'review'.
Side: STOXX add +1 / delete -1. DAX family: the sign of NET predicted passive demand,
  pi(index after) - pi(index before) + pi_TecDAX x (TecDAX after - TecDAX before), with the family pis of table_pi.csv
  (level 3 DAX, 2 MDAX, 1 SDAX, 0 none). The sister study's direction (up = promotion) is kept in `move`/`direction`.
Controls: the sister cells (07_returns.py) re-formed at the cut-off C:
  STOXX: one home line per company in a STOXX Europe country, operating company, priced in the last 10 sessions,
         cap within [0.15x, 6x] of the review's rank-550 float cap, >= 60 sessions of history, excluding every
         company predicted (any variant/horizon, naive included) or changed at the review; fallback chain
         country x size decile x momentum -> country x size quintile x momentum -> bloc x decile x momentum ->
         bloc x decile -> all x decile x momentum -> all x decile (>= 5 controls each).
  DAX:   German operating companies >= EUR 100m, not predicted/changed in the DAX family; decile x momentum -> decile.
Index: close-to-close = the sister benchmark (STOXX 600 member proxy, local currency; DAX/MDAX/TecDAX TR for DAX
moves). Open-based pieces: cap-weighted STOXX 600 members (German members for DAX moves) with a usable open.
Opens: Datastream ds2primqtprc open_/close_ of the primary quote (all years; data/opens_ds.parquet); Compustat g_secd
prcod is empty before mid-2018 and is used only as a cross-check / fallback.
Flow: D = pi x float shares (shares x Datastream/Worldscope float); ADV60 = median PRIMARY-venue volume T-65..T-6.
Crowding: abto (primary turnover T-126..A-1 vs T-252..T-127, minus the review's control median), runup_h3 / runup_h6
(sister car_events: car_H3_C + car_C_A, + car_H6_H3), EU net-short disclosures at month -1 (T-21 sessions, before A):
pod_short (14 pod families) and all_short (all disclosed holders), from shorts_footprint.csv (to 2025-03-18; Spain and
UK/BE after 2019 set to missing).
Tax fields: listing country regime, Compustat incorporation (fic), ISIN prefix, FTSE 100/250 status change in the same
review month (Datastream month-end lists 3670/3671), FTSE AIM All-Share membership (3673).
"""
import importlib.util
import numpy as np
import pandas as pd
from common import DATA, EU, CLIP, regime
from legs import KS, K0, daily_legs, split_session, subday_legs, flatten

spec = importlib.util.spec_from_file_location("eu_common", EU / "common.py")
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)
D_ = EU / "data"
cal = ec.eu_calendar()
z = np.load(D_ / "panel.npz")
P, RET, SH, AJ, FX, VAL, VP = z["P"], z["RET"], z["SH"], z["AJ"], z["FX"], z["VAL"], z["VP"]
AJf = pd.DataFrame(AJ).ffill().values
nd, nl = P.shape
L = pd.read_parquet(D_ / "panel_lines.parquet")
lid_ix = pd.Index(L.lid)
MCAP = P * SH / AJf * FX
R_cl = np.clip(RET, *CLIP)
TRI = np.nancumprod(np.nan_to_num(RET) + 1, axis=0)
ip = pd.read_parquet(D_ / "index_proxy.parquet")
IDX = {k: ip[k].fillna(0).values for k in ["stoxx_proxy_tr_local", "dax_tr", "mdax_tr", "tecdax_tr"]}
MS = np.load(D_ / "member_stoxx.npy")
operating = L.gsector.notna().values & (L.tpci == "0").values
cty = L.home_cty.values
gv = L.gvkey.values
TO = VP / SH
pit = pd.read_csv(DATA.parent / "table_pi.csv").set_index("family").pi_used
PI_LVL = {0: 0.0, 1: pit["SDAX"], 2: pit["MDAX"], 3: pit["DAX"]}
PI_TEC, PI_STOXX = pit["TecDAX"], pit["STOXX"]
LVL_NAME = {0: "none", 1: "SDAX", 2: "MDAX", 3: "DAX"}

# ---------------- opens: Datastream primary quote (all years), Compustat g_secd as fallback
dso = pd.read_parquet(DATA / "opens_ds.parquet")
dso["marketdate"] = pd.to_datetime(dso.marketdate)
gso = pd.read_parquet(DATA / "opens_eu.parquet")
gso["datadate"] = pd.to_datetime(gso.datadate)
gso["lid"] = gso.gvkey + "_" + gso.iid
OPEN = np.full((nd, nl), np.nan); CLOSE = np.full((nd, nl), np.nan)
SRC = np.zeros((nd, nl), dtype=np.int8)          # 1 = Datastream, 2 = Compustat
dpos = cal.get_indexer(dso.marketdate)
for colname in ["ds_pq", "ds_ic"]:
    m = L[colname].dropna().astype(int)
    ic2col = dict(zip(m.values, m.index))
    cols = dso.infocode.map(ic2col)
    okr = cols.notna() & (dpos >= 0) & (dso.open_ > 0) & (dso.close_ > 0)
    rr, cc = dpos[okr.values], cols[okr].astype(int).values
    empty = np.isnan(OPEN[rr, cc])
    OPEN[rr[empty], cc[empty]] = dso.open_[okr].values[empty]
    CLOSE[rr[empty], cc[empty]] = dso.close_[okr].values[empty]
    SRC[rr[empty], cc[empty]] = 1
gpos = cal.get_indexer(gso.datadate); gcol = lid_ix.get_indexer(gso.lid)
okg = (gpos >= 0) & (gcol >= 0) & (gso.prcod > 0).values & (gso.prccd > 0).values
rr, cc = gpos[okg], gcol[okg]
empty = np.isnan(OPEN[rr, cc])
OPEN[rr[empty], cc[empty]] = gso.prcod.values[okg][empty]
CLOSE[rr[empty], cc[empty]] = gso.prccd.values[okg][empty]
SRC[rr[empty], cc[empty]] = 2
# Compustat cross-check value of the open (2018+) for the checks file
GOPEN = np.full((nd, nl), np.nan); GCLOSE = np.full((nd, nl), np.nan)
GOPEN[rr, cc] = gso.prcod.values[okg]; GCLOSE[rr, cc] = gso.prccd.values[okg]


def subday(iT):
    out = {}
    for key, i in [("0", iT), ("1", iT + 1)]:
        on, idr = split_session(RET[i], OPEN[i], CLOSE[i])
        out["on" + key] = np.clip(on, *CLIP); out["id" + key] = np.clip(idr, *CLIP)
    return out


# ---------------- float, FTSE lists
fm = pd.read_parquet(D_ / "float_monthly.parquet")
fm["ff"] = fm.ff_ds.fillna(fm.ff_ws).fillna(1.0)
FF = fm.pivot_table(index="month", columns="line", values="ff")
fl = pd.read_parquet(DATA / "ftse_lists.parquet")
fl["startdate"] = pd.to_datetime(fl.startdate); fl["enddate"] = pd.to_datetime(fl.enddate)
ic2line = {}
for colname in ["ds_ic", "ds_pq"]:
    m = L[colname].dropna().astype(int)
    for ic, c in zip(m.values, m.index):
        ic2line.setdefault(ic, c)
fl["col"] = fl.infocode.map(ic2line)
fl = fl.dropna(subset=["col"])


def ftse_status(d):
    """col -> 'FTSE100' / 'FTSE250' / 'AIM' membership on month-end d (Datastream month-end lists)."""
    x = fl[(fl.startdate <= d) & (fl.enddate >= d)]
    st = {}
    for lst, nm in [(3673, "AIM"), (3671, "FTSE250"), (3670, "FTSE100")]:
        for c in x.loc[x.list == lst, "col"].astype(int):
            st[c] = nm
    return st


# ---------------- crowding inputs
sh = pd.read_csv(EU / "shorts_footprint.csv", usecols=["family", "review", "group", "gvkey", "k", "pod_pct", "all_pct", "covered",
                                                     "ukbe_late"], dtype={"gvkey": str})
# month -1 (T-21 sessions, dates <= 2025-03-18 in the sister file); Spain has no ISINs and UK/BE rows thin out after 2019,
# so those are set to missing rather than zero
sh = sh[(sh.k == -1) & sh.group.str.startswith("actual") & sh.covered & ~sh.ukbe_late]
SHORT = sh.drop_duplicates(["family", "review", "gvkey"]).set_index(["family", "review", "gvkey"])


def runups(fam):
    ce = pd.read_csv(EU / f"car_events_{fam}.csv", usecols=["review", "form", "portfolio", "col", "car_H6_H3", "car_H3_C", "car_C_A"])
    ce = ce[ce.portfolio.str.startswith("hind_")]
    r3 = ce[ce.form == "H3"].assign(v=lambda d: d.car_H3_C + d.car_C_A).drop_duplicates(["review", "col"]).set_index(["review", "col"]).v
    r6 = ce[ce.form == "H6"].assign(v=lambda d: d.car_H6_H3 + d.car_H3_C + d.car_C_A).drop_duplicates(["review", "col"]).set_index(["review", "col"]).v
    return r3, r6


def one_line_per_company(mask, f):
    adtv = np.nan_to_num(np.nanmean(VAL[max(0, f - 62):f + 1], 0), nan=-1)
    idx = np.where(mask)[0]
    df = pd.DataFrame({"g": gv[idx], "c": idx, "a": adtv[idx]}).sort_values(["g", "a"], ascending=[True, False])
    m = np.zeros(nl, bool); m[df.drop_duplicates("g").c.values] = True
    return m


def chain_keys(f, univ, family):
    mc = MCAP[f]
    mc = np.where(np.isnan(mc), pd.DataFrame(MCAP[max(0, f - 9):f + 1]).ffill().values[-1], mc)
    mom = TRI[f] / TRI[max(0, f - 126)] - 1
    sb = np.nanquantile(mc[univ], np.linspace(0.1, 0.9, 9))
    mb = np.nanquantile(mom[univ], [1 / 3, 2 / 3])
    sdec = np.searchsorted(sb, np.nan_to_num(mc, nan=0.0)); squi = sdec // 2
    mter = np.searchsorted(mb, np.nan_to_num(mom, nan=0.0))
    if family == "stoxx":
        keyset = cty.astype(str)
        blocs = np.array([ec.bloc(c) for c in keyset])
        return [("cty_dec_mom", np.char.add(np.char.add(keyset, "|"), (sdec * 3 + mter).astype(str))),
                ("cty_qui_mom", np.char.add(np.char.add(keyset, "|q"), (squi * 3 + mter).astype(str))),
                ("bloc_dec_mom", np.char.add(np.char.add(blocs.astype(str), "|"), (sdec * 3 + mter).astype(str))),
                ("bloc_dec", np.char.add(np.char.add(blocs.astype(str), "|d"), sdec.astype(str))),
                ("all_dec_mom", (sdec * 3 + mter).astype(str)),
                ("all_dec", np.char.add("d", sdec.astype(str)))]
    return [("dec_mom", (sdec * 3 + mter).astype(str)), ("dec", np.char.add("d", sdec.astype(str)))]


def run(family):
    rev = pd.read_csv(EU / f"reviews_{family}.csv", parse_dates=["H6", "H3", "C", "A", "T", "eff"])
    ev = pd.read_csv(EU / f"events_{family}.csv", dtype={"gvkey": str})
    ev = ev[(ev.etype == "review") & (ev.col >= 0)]
    pr = pd.read_csv(EU / f"predictions_{family}.csv", dtype={"gvkey": str})
    if family == "stoxx":
        diag = pd.read_csv(EU / "prediction_diagnostics_stoxx.csv").set_index(["review", "horizon"])
    r3, r6 = runups(family)
    rows = []
    for q in rev.itertuples():
        e = ev[ev.review == q.review]
        if e.empty:
            continue
        iT = cal.get_loc(q.T); iC = cal.get_loc(q.C); iA = cal.get_loc(q.A)
        if iT + 1 >= nd:
            continue
        x = pr[pr.review == q.review]
        if family == "stoxx":
            flagcols = [c for c in x.columns if c.startswith(("pred_", "naive_"))]
            excl_g = set(x.loc[x[flagcols].any(axis=1), "gvkey"]) | set(e.gvkey)
            L550 = diag.loc[(q.review, "C"), "L550_bn"] * 1e9
            ctry_ok = np.array([ec.stoxx_country_ok(c, q.review) for c in cty])
        else:
            excl_g = set(x.loc[x.pred_up | x.pred_down | x.naive_up | x.naive_down | x.pred_up_mem | x.pred_down_mem, "gvkey"]) | set(e.gvkey)
        excl = np.isin(gv, list(excl_g))
        f = iC
        live = (~np.isnan(P[max(0, f - 9):f + 1])).any(0)
        mc0 = MCAP[f]
        if family == "stoxx":
            base = one_line_per_company(live & operating & L.home_line.values & ctry_ok, f)
            univ = base & (mc0 >= 0.15 * L550) & (mc0 <= 6 * L550)
        else:
            base = one_line_per_company(live & operating & L.home_line.values & (cty == "DEU"), f)
            univ = base & (mc0 >= 1e8)
        ok_hist = (~np.isnan(P[max(0, f - 126):f + 1])).sum(0) >= 60
        ctrl = univ & ~excl & ok_hist
        chain = chain_keys(f, univ, family)
        ks = np.arange(iT - 5, min(iT + 21, nd))
        Rw = np.full((len(KS), nl), np.nan); Rw[:len(ks)] = R_cl[ks]
        sub = subday(iT)
        # index sub-day pieces
        wst = np.where(MS[iT - 1], np.nan_to_num(MCAP[iT - 1]), 0.0)
        wde = np.where(MS[iT - 1] & (cty == "DEU"), np.nan_to_num(MCAP[iT - 1]), 0.0)
        ixs = {}
        for nm_, w in [("stoxx", wst), ("de", wde)]:
            ixs[nm_] = {}
            for k, v in sub.items():
                m_ = np.isfinite(v) & (w > 0)
                ixs[nm_][k] = np.sum(w[m_] * v[m_]) / np.sum(w[m_]) if m_.any() else np.nan
        # abnormal turnover
        base_to = np.nanmean(TO[iT - 252:iT - 126], 0); nb = (~np.isnan(TO[iT - 252:iT - 126])).sum(0)
        pre_to = np.nanmean(TO[iT - 126:iA], 0); npre = (~np.isnan(TO[iT - 126:iA])).sum(0)
        lr = np.where((nb >= 40) & (npre >= 30) & (base_to > 0) & (pre_to > 0),
                      np.log(np.where(pre_to > 0, pre_to, 1) / np.where(base_to > 0, base_to, 1)), np.nan)
        ctrl_med = np.nanmedian(lr[ctrl])
        month = pd.Timestamp(q.T.year, q.T.month, 1)
        ffrow = FF.loc[month] if month in FF.index else pd.Series(dtype=float)
        me_pre = pd.Timestamp(q.T.year, q.T.month, 1) - pd.Timedelta(days=1)
        me_post = pd.Timestamp(q.T.year, q.T.month, 1) + pd.offsets.MonthEnd(0)
        fpre, fpost = ftse_status(me_pre), ftse_status(me_post)
        for r in e.itertuples():
            c = int(r.col)
            cm = None
            lvl = "none"
            for nm_, keys in chain:
                mbr = ctrl & (keys == keys[c]); mbr[c] = False
                if mbr.sum() >= 5:
                    cm, lvl = mbr, nm_
                    break
            if cm is None:
                continue
            cmean = np.nanmean(Rw[:, cm], 1)
            if family == "stoxx":
                xw_full = IDX["stoxx_proxy_tr_local"]
                ixk = "stoxx"
                side = 1 if r.side == "add" else -1
                group = "add" if r.side == "add" else "delete"
                move = group
                pi_net = PI_STOXX * side
            else:
                bk = "dax_tr" if 3 in (r.lvl_pre, r.lvl_post) else ("mdax_tr" if max(r.lvl_pre, r.lvl_post) > 0 else "tecdax_tr")
                xw_full = IDX[bk]
                ixk = "de"
                pi_net = PI_LVL[int(r.lvl_post)] - PI_LVL[int(r.lvl_pre)] + PI_TEC * (int(r.tec_post) - int(r.tec_pre))
                side = 1 if pi_net > 0 else -1
                move = (str(r.move) if isinstance(r.move, str) else "") + (" " + r.tec if isinstance(r.tec, str) else "")
                move = move.strip()
                group = r.direction          # up / down (sister definition)
            xw = np.full(len(KS), np.nan); xw[:len(ks)] = xw_full[ks]
            legs = daily_legs(Rw[:, c], cmean, xw)
            s = {k: v[c] for k, v in sub.items()}
            cms = {k: np.nanmean(v[cm]) for k, v in sub.items()}
            legs.update(subday_legs(s, cms, ixs[ixk]))
            vp = VP[iT - 65:iT - 5, c]
            adv60 = np.nanmedian(vp) if (~np.isnan(vp)).sum() >= 20 else np.nan
            vb = VP[iT - 252:iT - 126, c]
            advb = np.nanmedian(vb) if (~np.isnan(vb)).sum() >= 40 else np.nan
            ff = ffrow.get(c, 1.0) if len(ffrow) else 1.0
            ff = 1.0 if not np.isfinite(ff) else ff
            aj = AJf[iT - 1, c]
            F = SH[iT - 1, c] * ff / aj
            key = (family, q.review, r.gvkey)
            shrow = SHORT.loc[key] if key in SHORT.index else None
            rows.append(dict(
                market="EU", family="STOXX" if family == "stoxx" else "DAX", event_date=q.T.date(), review=q.review,
                year=q.year, id=L.lid[c], gvkey=r.gvkey, permco=np.nan, name=L.conm[c], cty=cty[c],
                group=group, move=move, side=side, cell=lvl, n_ctrl=int(cm.sum()),
                price=P[iT - 1, c], fx_eur=FX[iT - 1, c], shares=SH[iT - 1, c] / aj, float_sh=F, float_src="ds/ws",
                pi=abs(pi_net), pi_signed=pi_net, demand_sh=abs(pi_net) * F,
                adv60=adv60 / aj, adv_base=advb / aj, vol_T=VP[iT, c] / aj, med20=np.nanmedian(VP[iT - 20:iT, c]) / aj, vol_basis="primary",
                abto=lr[c] - ctrl_med, abto_raw=lr[c], ann_date=q.A.date(),
                d0_ret_is_zero=bool(np.isfinite(RET[iT, c]) and RET[iT, c] == 0.0),
                open_T1_ok=bool(np.isfinite(s["on1"])), open_T_ok=bool(np.isfinite(s["on0"])),
                open_src_T1=int(SRC[iT + 1, c]),
                open_T1_eqprev=bool(np.isfinite(OPEN[iT + 1, c]) and abs(OPEN[iT + 1, c] / CLOSE[iT, c] - 1) < 1e-9) if np.isfinite(CLOSE[iT, c]) else False,
                g_open_T1=GOPEN[iT + 1, c], ds_open_T1=OPEN[iT + 1, c] if SRC[iT + 1, c] == 1 else np.nan,
                g_close_T1=GCLOSE[iT + 1, c], ds_close_T1=CLOSE[iT + 1, c] if SRC[iT + 1, c] == 1 else np.nan,
                runup_h3=r3.get((q.review, c), np.nan), runup_h6=r6.get((q.review, c), np.nan),
                pod_short=shrow.pod_pct if shrow is not None else np.nan,
                all_short=shrow.all_pct if shrow is not None else np.nan,
                short_covered=bool(shrow.covered) if shrow is not None else False,
                regime=regime(cty[c]), fic=L.fic[c], isin_cc=str(L["isin"].iat[c])[:2] if isinstance(L["isin"].iat[c], str) else "",
                ftse_pre=fpre.get(c, "none"), ftse_post=fpost.get(c, "none"),
                lvl_pre=getattr(r, "lvl_pre", np.nan), lvl_post=getattr(r, "lvl_post", np.nan),
                **flatten(legs)))
        print(family, q.review, len(rows), flush=True)
    return pd.DataFrame(rows)


out = pd.concat([run("stoxx"), run("dax")], ignore_index=True)
# names that change in both families at the same review: total signed demand
k = out.groupby(["review", "gvkey"]).pi_signed.transform("sum")
n = out.groupby(["review", "gvkey"]).family.transform("nunique")
out["both_families"] = n > 1
out["pi_signed_total"] = k
out.to_parquet(DATA / "legs_eu.parquet", index=False)
print(out.groupby(["family", "group"]).size(), out.both_families.sum())
