"""Step 5 (US): holdings-implied passive demand per stock at the Russell June reconstitution (2011-2025) and at the
S&P 500 quarterly rebalances and additions (2010-2025), from CRSP MF month-end holdings of the iShares rulers.

iShares US daily history is not available (ishares.com ignores asOfDate; the product page offers only the latest,
last quarter-end and last year-end dates; probe in RESULTS.md), so the US uses CRSP MF holdings, which are month-end.
  Russell: pre = last report before T (end-May), post = first report on or after T (end-June; when T is the last
           session of June the 30 June report is already post-trade, checked on the adds). Rulers: IWM (Russell 2000),
           IWB (Russell 1000), IWV (Russell 3000).
  S&P 500: quarterly rebalance T = 3rd Friday of Mar/Jun/Sep/Dec; pre = end of the month before, post = end of the
           rebalance month. Ruler IVV. Additions use their own effective dates (Compustat spells, current members only).
Arithmetic as in s04: split basis from CRSP (total vs price return between the two reports), c' from one price vector
(CRSP close at T), fund trade = h_post/c' - h_pre, delta = (h_post/h_pre)/c - 1, D_vis = trade x AUM_visible / E_fund,
with AUM_visible = the family's ETF TNA (IWM+VTWO, IWB+VONE, IWV+VTHR, SPY+IVV+VOO; CRSP MF, pre month-end).
D_vis_all scales by all US-registered index funds on that index instead.
A month-end pair also carries off-cycle changes inside the month (IPOs, M&A, S&P weekly share changes); that noise
attenuates the volume calibration and is a stated limitation. THR comes from a placebo month pair (end-Mar -> end-Apr
for the Russell; the month after each S&P quarter for the S&P).
Outputs: us_demand.csv (Russell, one row per recon x permno), us_sp500_demand.csv (S&P 500 quarters and adds),
data/us_pairs.parquet.
"""
import numpy as np
import pandas as pd
import duckdb
from common import DATA, OUT, RUS, sessions

H = pd.read_parquet(DATA / "us_holdings.parquet")
H["report_dt"] = pd.to_datetime(H.report_dt)
H = H[H.permno.notna() & (H.nbr_shares > 0)]
H["permno"] = H.permno.astype(int)
PI = pd.read_parquet(DATA / "pi_us_monthly.parquet")
CAL = sessions("XNYS")
con = duckdb.connect()
FAMILY = {"IWM": "R2000", "IWB": "R1000", "IWV": "R3000", "IVV": "SP500"}


def dsf_window(d0, d1):
    return con.execute(f"""SELECT permno, dlycaldt, ret, prc, vol, shrout FROM read_parquet('{RUS}/_cache/dsf/*.parquet')
                           WHERE dlycaldt BETWEEN '{d0:%Y-%m-%d}' AND '{d1:%Y-%m-%d}'""").df()


def report_dates(fund):
    return np.array(sorted(H[H.fund == fund].report_dt.unique()), dtype="datetime64[ns]")


def pair(fund, pre, post, T, px):
    """Fund trades between two month-end reports, priced at T. px: DataFrame permno -> prc_T, ret_cum, pratio."""
    hb = H[(H.fund == fund) & (H.report_dt == pre)].groupby("permno").agg(h_b=("nbr_shares", "sum"), mv_b=("market_val", "sum"))
    ha = H[(H.fund == fund) & (H.report_dt == post)].groupby("permno").agg(h_a=("nbr_shares", "sum"), mv_a=("market_val", "sum"))
    if hb.empty or ha.empty:
        return None, None
    g = hb.join(ha, how="outer").fillna(0.0)
    g = g.join(px, how="left")
    f = (g.split_f.fillna(1.0)).values
    g["hb_s"] = g.h_b * f                              # pre holdings on the post (T) share basis
    p = g.prc_T.abs()
    p = p.fillna(g.mv_a / g.h_a.replace(0, np.nan)).fillna(g.mv_b / g.h_b.replace(0, np.nan) / f)
    cprime = (g.h_a * p).sum() / (g.hb_s * p).sum()
    cont = (g.hb_s > 0) & (g.h_a > 0)
    r = g.h_a / g.hb_s
    c = np.median(r[cont])
    g["delta"] = np.where(cont, r / c - 1, np.nan)
    g["trade"] = g.h_a / cprime - g.hb_s
    g["funding"] = np.where(cont, g.hb_s * (c / cprime - 1), 0.0)
    g["p_T"] = p
    E_b = hb.mv_b.sum()
    st = dict(fund=fund, pre=pre, post=post, T=T, n_b=len(hb), n_a=len(ha), cprime=cprime, c_med=c, m_index=cprime / c,
              E_b=E_b, n_add=int(((g.hb_s == 0) & (g.h_a > 0)).sum()), n_del=int(((g.hb_s > 0) & (g.h_a == 0)).sum()))
    return g, st


def px_table(d, pre, post, T):
    """per permno: close at T, split factor between the two report dates (total vs price return)."""
    x = d[(d.dlycaldt > pre) & (d.dlycaldt <= post)].sort_values("dlycaldt")
    cum = x.groupby("permno").ret.apply(lambda s: np.prod(1 + s.fillna(0)))
    pb = d[d.dlycaldt <= pre].sort_values("dlycaldt").groupby("permno").prc.last().abs()
    pa = d[d.dlycaldt <= post].sort_values("dlycaldt").groupby("permno").prc.last().abs()
    pT = d[d.dlycaldt <= T].sort_values("dlycaldt").groupby("permno").prc.last().abs()
    f = cum / (pa / pb)
    f = f.where((f > 1.25) | (f < 0.8), 1.0)                # only splits / reverse splits, not dividends
    return pd.DataFrame({"prc_T": pT, "split_f": f})


def vol_fields(d, T):
    """ADV60 (median of T-65..T-6), V_T, med20, shares outstanding and price at T-1 per permno."""
    i = CAL.get_loc(T)
    w = d[(d.dlycaldt >= CAL[i - 65]) & (d.dlycaldt <= CAL[i - 6])]
    adv = w.groupby("permno").vol.agg(lambda s: s.median() if s.notna().sum() >= 20 else np.nan)
    m20 = d[(d.dlycaldt >= CAL[i - 20]) & (d.dlycaldt <= CAL[i - 1])].groupby("permno").vol.median()
    vT = d[d.dlycaldt == T].set_index("permno").vol
    pm1 = d[d.dlycaldt == CAL[i - 1]].set_index("permno")
    return pd.DataFrame({"adv60": adv, "med20": m20, "vol_T": vT, "price_Tm1": pm1.prc.abs(),
                         "shares_out": pm1.shrout * 1000})


def aum(fam, pre, kind):
    x = PI[(PI.family == fam) & (PI.date == pre + pd.offsets.MonthEnd(0))]
    if x.empty:
        return np.nan
    return float(x["tna_etf" if kind == "etf" else "tna_all"].iloc[0])


def run_russell():
    rus = pd.read_csv(RUS / "output" / "events_actual.csv")
    rus = rus[rus.valid]
    import importlib.util
    spec = importlib.util.spec_from_file_location("rc", RUS / "common.py"); rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
    CALR = rc.CAL
    rows, pairs, plac = [], [], []
    for y in range(2011, 2026):
        T = pd.Timestamp(CALR.loc[y, "T"])
        d = dsf_window(CAL[CAL.get_loc(T) - 80], CAL[CAL.get_loc(T) + 30])
        per = {}
        for fund in ("IWM", "IWB", "IWV"):
            rd = report_dates(fund)
            pre = pd.Timestamp(rd[rd < np.datetime64(T)].max())
            # post = end-July: the end-June report is not always post-trade (2024: no adds yet; 2025: 115 of 152
            # deletes still partly held at 30 Jun), so every year uses the first report after end-June
            post = pd.Timestamp(rd[rd > np.datetime64(T + pd.offsets.MonthEnd(0))].min())
            p1 = pd.Timestamp(rd[rd >= np.datetime64(T)].min())
            px = px_table(d, pre, post, T)
            g, st = pair(fund, pre, post, T, px)
            fam = FAMILY[fund]
            sc_e = aum(fam, pre, "etf") / st["E_b"]
            sc_a = aum(fam, pre, "all") / st["E_b"]
            st.update(year=y, family=fam, scale_etf=sc_e, scale_all=sc_a)
            per[fund] = g.assign(D_vis=g.trade * sc_e, D_all=g.trade * sc_a)
            # share of the year's adds / deletes already visible in the first report on or after T
            g1, s1 = pair(fund, pre, p1, T, px_table(d, pre, p1, T))
            st["june_n_add"], st["june_n_del"] = s1["n_add"], s1["n_del"]
            # placebo: end-Mar -> end-May of the same year (two months, no reconstitution)
            rdp = [pd.Timestamp(v) for v in rd if pd.Timestamp(v).year == y and pd.Timestamp(v).month in (3, 5)]
            if len(rdp) == 2:
                dp = dsf_window(rdp[0] - pd.Timedelta(days=5), rdp[1] + pd.Timedelta(days=2))
                gp, _ = pair(fund, rdp[0], rdp[1], rdp[1], px_table(dp, rdp[0], rdp[1], rdp[1]))
                if gp is not None:
                    plac.append(gp.assign(fund=fund, year=y, D_pl=gp.trade * sc_e)[["fund", "year", "delta", "D_pl", "hb_s"]])
            pairs.append(st)
        vf = vol_fields(d, T)
        allp = sorted(set().union(*[set(g.index) for g in per.values()]))
        for pn in allp:
            row = dict(year=y, T=T, permno=pn)
            for fund, g in per.items():
                f = FAMILY[fund]
                if pn in g.index:
                    x = g.loc[pn]
                    row.update({f"h_b_{f}": x.hb_s, f"h_a_{f}": x.h_a, f"delta_{f}": x.delta, f"D_vis_{f}": x.D_vis,
                                f"D_all_{f}": x.D_all, "p_T": x.p_T})
                else:
                    row.update({f"h_b_{f}": 0.0, f"h_a_{f}": 0.0, f"D_vis_{f}": 0.0, f"D_all_{f}": 0.0})
            rows.append(row)
        print(y, T.date(), {f: (p["n_add"], p["n_del"], round(p["cprime"], 4), round(p["m_index"], 4)) for f, p in
                              zip(("IWM", "IWB", "IWV"), pairs[-3:])}, flush=True)
    D = pd.DataFrame(rows)
    PL = pd.concat(plac)
    PL.to_parquet(DATA / "us_placebo.parquet", index=False)
    thr = PL.groupby("fund").delta.apply(lambda s: np.nanquantile(np.abs(s), 0.95)).to_dict()
    print("placebo |delta| quantiles:", PL.groupby("fund").delta.apply(lambda s: np.nanquantile(np.abs(s), [.5, .9, .95, .99]).round(4).tolist()).to_dict())
    THR = max(0.01, max(thr.values()))
    print("THR (max over rulers of placebo p95):", THR)
    for f in ("R2000", "R1000", "R3000"):
        hb, ha, dl = D[f"h_b_{f}"].fillna(0), D[f"h_a_{f}"].fillna(0), D[f"delta_{f}"]
        D[f"cls_{f}"] = np.select([(hb == 0) & (ha > 0), (hb > 0) & (ha == 0), (hb > 0) & (dl.abs() > THR), hb > 0],
                                  ["add", "delete", "weight", "unchanged"], "none")
    D["THR"] = THR
    D["D_net"] = D.D_vis_R2000 + D.D_vis_R1000 + D.D_vis_R3000
    D["D_net_all"] = D.D_all_R2000 + D.D_all_R1000 + D.D_all_R3000
    D["D_gross"] = D.D_vis_R2000.abs() + D.D_vis_R1000.abs() + D.D_vis_R3000.abs()
    # sister event groups (actual Russell changes, ACT-P): permno x year
    rus = rus.rename(columns={"permno": "permno_s"})
    gmap = {(int(y), int(p)): g for y, p, g in zip(rus.year, rus.permno_s, rus.group) if pd.notna(p)}
    D["sister_group"] = [gmap.get((y, p), "") for y, p in zip(D.year, D.permno)]
    lab = np.select([D.sister_group != "", (D.cls_R2000 == "weight") | (D.cls_R1000 == "weight"),
                     (D.cls_R2000 == "unchanged") | (D.cls_R1000 == "unchanged")],
                    [D.sister_group, "weight", "unchanged"], "other")
    D["event"] = lab
    # volume fields
    vv = []
    for y in D.year.unique():
        T = pd.Timestamp(CALR.loc[y, "T"])
        d = dsf_window(CAL[CAL.get_loc(T) - 80], CAL[CAL.get_loc(T) + 2])
        vv.append(vol_fields(d, T).assign(year=y).reset_index())
    V = pd.concat(vv)
    D = D.merge(V, on=["year", "permno"], how="left")
    D["notional_net"] = D.D_net * D.price_Tm1
    D["net_adv"] = D.D_net / D.adv60
    D.to_csv(OUT / "us_demand.csv", index=False)
    pd.DataFrame(pairs).to_parquet(DATA / "us_pairs.parquet", index=False)
    print(D.groupby("event").size())
    return D


def run_sp500():
    sp = pd.read_parquet(DATA / "sp500_spells.parquet")
    ccm = pd.read_parquet(RUS / "_cache" / "ccm.parquet")
    rd = report_dates("IVV")
    rows, pairs = [], []
    for y in range(2010, 2026):
        for m in (3, 6, 9, 12):
            t_nom = pd.Timestamp(y, m, 1) + pd.Timedelta(days=(4 - pd.Timestamp(y, m, 1).weekday()) % 7 + 14)
            T = CAL[CAL.searchsorted(t_nom, side="right") - 1]
            pre_c = rd[rd < np.datetime64(T)]
            post_c = rd[rd >= np.datetime64(T)]
            if not len(pre_c) or not len(post_c):
                continue
            pre, post = pd.Timestamp(pre_c.max()), pd.Timestamp(post_c.min())
            if (post - pre).days > 40:
                continue
            d = dsf_window(CAL[CAL.get_loc(T) - 80], CAL[CAL.get_loc(T) + 30])
            g, st = pair("IVV", pre, post, T, px_table(d, pre, post, T))
            if g is None:
                continue
            sc = aum("SP500", pre, "etf") / st["E_b"]
            sca = aum("SP500", pre, "all") / st["E_b"]
            st.update(quarter=f"{y}-{m:02d}", scale_etf=sc, scale_all=sca)
            pairs.append(st)
            vf = vol_fields(d, T)
            g = g.assign(D_vis=g.trade * sc, D_all=g.trade * sca).join(vf, how="left")
            g["quarter"] = f"{y}-{m:02d}"; g["T"] = T
            rows.append(g.reset_index())
    D = pd.concat(rows)
    # placebo threshold: distribution of |delta| is wide for month pairs; use 99.5th pct of the median-quarter pairs
    THR = max(0.01, float(np.nanquantile(np.abs(D.delta), 0.75)) * 4)
    D["cls"] = np.select([(D.hb_s == 0) & (D.h_a > 0), (D.hb_s > 0) & (D.h_a == 0), (D.hb_s > 0) & (D.delta.abs() > THR),
                          D.hb_s > 0], ["add", "delete", "weight", "unchanged"], "none")
    D["THR"] = THR
    # additions with their own effective date (Compustat spells -> CRSP permno via ccm)
    sp["dfrom"] = pd.to_datetime(sp.dfrom)
    adds = sp[(sp.dfrom >= "2010-07-01") & (sp.dfrom <= "2025-12-31")].copy()
    c = ccm.copy()
    for k in ("linkdt", "linkenddt"):
        if k in c:
            c[k] = pd.to_datetime(c[k])
    c = c[c.linkprim.isin(["P", "C"])]
    adds = adds.merge(c[["gvkey", "permno", "linkdt", "linkenddt"]], on="gvkey", how="left")
    adds = adds[(adds.dfrom >= adds.linkdt) & ((adds.dfrom <= adds.linkenddt) | adds.linkenddt.isna())]
    arow = []
    for a in adds.itertuples():
        T = CAL[CAL.searchsorted(a.dfrom) - 1]
        prc = rd[rd >= np.datetime64(T)]
        if not len(prc):
            continue
        post = pd.Timestamp(prc.min())
        h = H[(H.fund == "IVV") & (H.report_dt == post) & (H.permno == int(a.permno))]
        if h.empty:
            continue
        d = dsf_window(CAL[CAL.get_loc(T) - 80], post + pd.Timedelta(days=3))
        vf = vol_fields(d, T)
        x = d[(d.permno == int(a.permno)) & (d.dlycaldt > T) & (d.dlycaldt <= post)]
        ret_cum = np.prod(1 + x.ret.fillna(0)); pr = d[(d.permno == int(a.permno))].set_index("dlycaldt").prc.abs()
        pT, pP = pr.get(T, np.nan), pr[pr.index <= post].iloc[-1] if (pr.index <= post).any() else np.nan
        f = ret_cum / (pP / pT) if pT and pP else 1.0
        f = f if (f > 1.25 or f < 0.8) else 1.0
        # IVV holdings at post, on the T share basis, deflated by IVV's own AUM change T -> post (creations)
        pre_rd = pd.Timestamp(rd[rd < np.datetime64(T)].max())
        E_pre = H[(H.fund == "IVV") & (H.report_dt == pre_rd)].market_val.sum()
        E_post = H[(H.fund == "IVV") & (H.report_dt == post)].market_val.sum()
        hT = h.nbr_shares.sum() / f
        sc = aum("SP500", pre_rd, "etf") / E_pre
        # creations between pre and post at roughly constant weights: scale by E_pre/E_post x index price ratio ~ use ratio of
        # the fund's units, proxied by AUM ratio deflated by the S&P price move (SPY total return from CRSP if present)
        spy = d[d.permno == 84398].set_index("dlycaldt").prc.abs()
        px_ratio = (spy[spy.index <= post].iloc[-1] / spy[spy.index <= pre_rd].iloc[-1]) if len(spy) else 1.0
        units_ratio = (E_post / E_pre) / px_ratio
        D_vis = hT / units_ratio * sc
        r = dict(gvkey=a.gvkey, permno=int(a.permno), tic=a.tic, conm=a.conm, eff=a.dfrom, T=T, post=post, h_post=h.nbr_shares.sum(),
                 units_ratio=units_ratio, D_vis=D_vis, D_all=hT / units_ratio * aum("SP500", pre_rd, "all") / E_pre)
        if int(a.permno) in vf.index:
            r.update(vf.loc[int(a.permno)].to_dict())
        arow.append(r)
    A = pd.DataFrame(arow)
    D.to_csv(OUT / "us_sp500_demand.csv", index=False)
    A.to_csv(OUT / "us_sp500_adds.csv", index=False)
    pd.DataFrame(pairs).to_parquet(DATA / "us_sp500_pairs.parquet", index=False)
    print(D.groupby("cls").size(), len(A), "S&P adds with IVV holdings")
    return D, A


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["russell", "sp500"]
    if "russell" in todo:
        run_russell()
    if "sp500" in todo:
        run_sp500()
