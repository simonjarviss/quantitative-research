"""Step 4: returns of the capturable (predicted) and hindsight (actual) portfolios, matched-control abnormal returns.

Portfolios (formation date f):
  PRED-H6 / PRED-H3 / PRED-HR : names predicted at f to be BUY_add / BUY_mig / SELL_del / SELL_mig (incl. false positives)
  ACT-H6 / ACT-H3 / ACT-HR    : the actual changes (hindsight reference), names already trading at f
  ACT-P                       : the actual changes, formed at the preliminary-list close (= public-study start point)
Returns: CRSP dsf_v2 daily total returns, delisting-aware (the delisting return is a dsf_v2 row);
buy-and-hold from the close of the window start to the close of the window end; missing days = 0;
after a delisting the position is cash.
Controls: companies ranked 600-5000 in the PIT rebuild at the horizon, not predicted to change at that horizon and
not actually changed this year. Cells: 10 size deciles x 5 prior-6-month (126 trading days) return quintiles,
breakpoints from the controls, formed at f.  AR = stock BHR - equal-weight BHR of the controls in its cell
(cell < 3 controls -> size-decile controls; no 6-month history -> size-decile controls).

Outputs (output/)
  car_stock_level.csv.gz   one row per (portfolio, year, company, window)
  car_summary.csv          pooled stats per (portfolio, group, window)
  car_by_year.csv          per-year means
  spread_by_year.csv       dollar-neutral long-BUY / short-SELL EW spread per year
  car_paths.csv            daily mean AR and CAR, T-130..T+60, per (portfolio, group, year) and pooled (year = 0)
Usage: python3 s04_returns.py [years...]
"""
import sys
import numpy as np
import pandas as pd
from common import CACHE, OUT, CAL, IWM_PERMNO, MAIN_YEARS, EXT_YEARS, SIDE
import lib

EV = pd.read_csv(OUT / "events_actual.csv")
GRP = ["BUY_add", "BUY_mig", "SELL_del", "SELL_mig"]
WINDOWS = [("H6_T", "H6", "T"), ("H3_T", "H3", "T"), ("HR_T", "HR", "T"), ("P_T", "P", "T"),
           ("H6_H3", "H6", "H3"), ("H3_HR", "H3", "HR"), ("HR_P", "HR", "P"), ("P_Tm1", "P", "Tm1"),
           ("Tm1_T", "Tm1", "T"), ("T_Tp1", "T", "Tp1"), ("T_Tp20", "T", "Tp20"), ("T_Tp60", "T", "Tp60")]
ORDER = {"H6": 0, "H3": 1, "HR": 2, "P": 3, "Tm1": 4, "T": 5}


def year_panel(y, hd):
    start = lib.tshift(hd["H6"], -135)
    d = lib.daily(start, hd["Tp60"], "permno, permco, dlycaldt, ret, prc, cap")
    R = d.pivot_table(index="dlycaldt", columns="permno", values="ret", aggfunc="first")
    Pm = d.pivot_table(index="dlycaldt", columns="permno", values="prc", aggfunc="first")
    td = lib.tdays()
    days = td[(td >= start) & (td <= hd["Tp60"])]
    R = R.reindex(days)
    Pm = Pm.reindex(days)
    alive = Pm.notna() | R.notna()
    ccap = d.groupby(["dlycaldt", "permco"]).cap.sum()
    return R, Pm, alive, ccap


def bhr(R, alive, a, b, cols):
    cols = [c for c in cols if c in R.columns]
    ok = alive.loc[a, cols]
    seg = R.loc[(R.index > a) & (R.index <= b), cols].fillna(0.0)
    out = (1 + seg).prod() - 1
    out[~ok] = np.nan
    return out


def mom(R, alive, f, cols):
    cols = [c for c in cols if c in R.columns]
    i = R.index.get_loc(f)
    seg = R.iloc[max(0, i - 125): i + 1][cols]
    n = seg.notna().sum()
    m = (1 + seg.fillna(0)).prod() - 1
    m[n < 60] = np.nan
    return m


def cells(ctrl, ev):
    """size decile x momentum quintile, breakpoints from controls"""
    qs = np.nanquantile(ctrl.ccap, np.linspace(0.1, 0.9, 9))
    qm = np.nanquantile(ctrl.mom.dropna(), np.linspace(0.2, 0.8, 4))
    for df in (ctrl, ev):
        df["sz"] = np.searchsorted(qs, df.ccap.values)
        df["mq"] = np.where(df.mom.isna(), -1, np.searchsorted(qm, df.mom.fillna(0).values))
    return ctrl, ev


def run_year(y):
    hd = lib.horizon_dates(y)
    R, Pm, alive, ccap = year_panel(y, hd)
    iwm = R[IWM_PERMNO] if IWM_PERMNO in R.columns else None
    act = EV[(EV.year == y) & EV.valid & EV.group.isin(GRP)].dropna(subset=["permno"])
    changed = set(EV.loc[EV.year == y, "permco"])
    preds = {h: pd.read_parquet(CACHE / f"pred_{y}_{h}.parquet") for h in ["H6", "H3", "HR"]}
    ports = []
    for h in ["H6", "H3", "HR"]:
        p = preds[h]
        ctrl = p[(p["rank"].between(600, 5000)) & (p.pgroup == "none") & ~p.permco.isin(changed)][["permco", "permno"]]
        pe = p[p.pgroup.isin(GRP)][["permco", "permno", "pgroup"]].rename(columns={"pgroup": "group"})
        ports.append((f"PRED-{h}", h, pe, ctrl))
        ports.append((f"ACT-{h}", h, act[["permco", "permno", "group"]], ctrl))
    ctrl_hr = ports[-1][3]
    ports.append(("ACT-P", "P", act[["permco", "permno", "group"]], ctrl_hr))

    rows, paths, cellrows = [], [], []
    rel = R.index.get_loc(hd["T"])
    pidx = R.index[rel - 130: rel + 61]
    for name, fh, ev, ctrl in ports:
        f = hd[fh]
        ev = ev.copy(); ctrl = ctrl.copy()
        ev["permno"] = ev.permno.astype(int); ctrl["permno"] = ctrl.permno.astype(int)
        ev = ev[ev.permno.isin(R.columns)]
        ev = ev[alive.loc[f, ev.permno].values]                 # must be trading at formation
        ctrl = ctrl[ctrl.permno.isin(R.columns)]
        ctrl = ctrl[alive.loc[f, ctrl.permno].values]
        cc = ccap.loc[f] if f in ccap.index.get_level_values(0) else pd.Series(dtype=float)
        for df in (ev, ctrl):
            df["ccap"] = df.permco.map(cc)
            df["mom"] = df.permno.map(mom(R, alive, f, df.permno.tolist()))
        ctrl = ctrl.dropna(subset=["ccap"])
        ctrl, ev = cells(ctrl, ev)
        cellrows.append(pd.concat([ev.assign(role="event"), ctrl.assign(role="control", group="control")])
                        .assign(portfolio=name, year=y, fdate=f))
        for wname, a, b in WINDOWS:
            if ORDER[a] < ORDER[fh]:
                continue
            A, B = hd[a], hd[b]
            re_ = bhr(R, alive, A, B, ev.permno.tolist())
            rc = bhr(R, alive, A, B, ctrl.permno.tolist())
            c = ctrl.assign(r=ctrl.permno.map(rc))
            full = c.groupby(["sz", "mq"]).r.agg(["mean", "count"])
            szonly = c.groupby("sz").r.mean()
            e = ev.copy()
            e["ret"] = e.permno.map(re_)
            key = list(zip(e.sz, e.mq))
            cm = pd.Series([full["mean"].get(k, np.nan) if full["count"].get(k, 0) >= 3 else np.nan for k in key], index=e.index)
            cm = cm.fillna(e.sz.map(szonly))
            e["ctrl_ret"] = cm
            e["ar"] = e.ret - e.ctrl_ret
            e["iwm"] = float((1 + iwm.loc[(iwm.index > A) & (iwm.index <= B)].fillna(0)).prod() - 1) if iwm is not None else np.nan
            e["ar_iwm"] = e.ret - e.iwm
            e["portfolio"], e["year"], e["window"], e["start"], e["end"] = name, y, wname, A.date(), B.date()
            rows.append(e)
        # daily CAR path (controls in the same cell, daily EW mean)
        Rp = R.loc[pidx]
        cm_daily = {}
        for (sz, mq), g in ctrl.groupby(["sz", "mq"]):
            cm_daily[(sz, mq)] = (Rp[g.permno].mean(axis=1), len(g))
        sz_daily = {sz: Rp[g.permno].mean(axis=1) for sz, g in ctrl.groupby("sz")}
        for grp, g in ev.groupby("group"):
            ars = []
            for _, r_ in g.iterrows():
                k = (r_.sz, r_.mq)
                base = cm_daily[k][0] if k in cm_daily and cm_daily[k][1] >= 3 else sz_daily.get(r_.sz)
                if base is None:
                    continue
                ars.append(Rp[r_.permno] - base)
            if not ars:
                continue
            A_ = pd.concat(ars, axis=1)
            m = A_.mean(axis=1)
            paths.append(pd.DataFrame({"portfolio": name, "group": grp, "year": y, "rel_day": np.arange(-130, len(pidx) - 130),
                                       "date": pidx.date, "n": A_.notna().sum(axis=1).values, "mean_ar": m.values,
                                       "car": m.fillna(0).cumsum().values}))
    pd.concat(cellrows, ignore_index=True).to_parquet(CACHE / f"cells_{y}.parquet", index=False)
    return pd.concat(rows, ignore_index=True), pd.concat(paths, ignore_index=True)


def summarize(S):
    out = []
    S = S.copy()
    S["side"] = S.group.map(SIDE)
    pooled = pd.concat([S, S.assign(group=np.where(S.side > 0, "BUY_all", "SELL_all"))])
    for (p, g, w), d in pooled.groupby(["portfolio", "group", "window"]):
        d = d.dropna(subset=["ar"])
        if d.empty:
            continue
        side = d.side.iloc[0]
        m, se, t = lib.cluster_t(d.ar, d.year)
        ym = d.groupby("year").ar.mean()
        lo = d.ar.abs().quantile(0.99)
        tr = d[d.ar.abs() <= lo]
        mt, _, tt = lib.cluster_t(tr.ar, tr.year)
        out.append(dict(portfolio=p, group=g, window=w, n=len(d), n_years=d.year.nunique(), mean_car=m, t_cluster_year=t,
                        median_car=d.ar.median(), mean_raw=d.ret.mean(), mean_minus_iwm=d.ar_iwm.mean(),
                        mean_ctrl=d.ctrl_ret.mean(), years_in_trade_dir=int((np.sign(ym) == side).sum()),
                        hit_rate=float((np.sign(d.ar) == side).mean()), mean_of_year_means=ym.mean(),
                        t_year_means=ym.mean() / (ym.std(ddof=1) / np.sqrt(len(ym))) if len(ym) > 1 else np.nan,
                        mean_car_trim1pct=mt, t_trim=tt))
    return pd.DataFrame(out)


def spreads(S):
    S = S.dropna(subset=["ar"]).copy()
    S["side"] = S.group.map(SIDE)
    g = S.groupby(["portfolio", "window", "year", "side"]).agg(ar=("ar", "mean"), ret=("ret", "mean"), n=("ar", "size")).unstack("side")
    out = pd.DataFrame({"n_long": g[("n", 1)], "n_short": g[("n", -1)],
                        "spread_ar": g[("ar", 1)] - g[("ar", -1)], "spread_raw": g[("ret", 1)] - g[("ret", -1)],
                        "long_ar": g[("ar", 1)], "short_ar": g[("ar", -1)]}).reset_index()
    return out


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:] if a.isdigit()] or MAIN_YEARS + EXT_YEARS
    S, P = [], []
    for y in yrs:
        s, p = run_year(y)
        S.append(s); P.append(p)
        x = s[(s.portfolio.str.startswith("PRED")) & s.window.isin(["H6_T", "H3_T", "HR_T", "T_Tp20"])]
        print(y, x.groupby(["portfolio", "window", "group"]).ar.mean().unstack().round(3).to_string(), flush=True)
    S = pd.concat(S, ignore_index=True)
    P = pd.concat(P, ignore_index=True)
    keep = ["portfolio", "year", "window", "start", "end", "group", "permco", "permno", "ccap", "mom", "sz", "mq",
            "ret", "ctrl_ret", "ar", "iwm", "ar_iwm"]
    S[keep].to_csv(OUT / "car_stock_level.csv.gz", index=False)
    for lab, yy in [("", MAIN_YEARS), ("_2008_2015", EXT_YEARS), ("_2008_2025", MAIN_YEARS + EXT_YEARS)]:
        sub = S[S.year.isin(yy)]
        if sub.empty:
            continue
        summarize(sub).to_csv(OUT / f"car_summary{lab}.csv", index=False)
    S["side"] = S.group.map(SIDE)
    by = S.groupby(["portfolio", "window", "group", "year"]).agg(n=("ar", "count"), mean_car=("ar", "mean"), median_car=("ar", "median"),
                                                                   mean_raw=("ret", "mean"), mean_minus_iwm=("ar_iwm", "mean")).reset_index()
    by.to_csv(OUT / "car_by_year.csv", index=False)
    spreads(S).to_csv(OUT / "spread_by_year.csv", index=False)
    # pooled paths: equal-weight across years (main years; extension years separately)
    pool = []
    for lab, yy in [(0, MAIN_YEARS), (1, EXT_YEARS)]:
        q = P[P.year.isin(yy)].groupby(["portfolio", "group", "rel_day"]).agg(mean_ar=("mean_ar", "mean"), n=("n", "sum")).reset_index()
        q["car"] = q.groupby(["portfolio", "group"]).mean_ar.cumsum()
        q["year"] = lab
        pool.append(q)
    pd.concat([P] + pool, ignore_index=True).to_csv(OUT / "car_paths.csv", index=False)
    print("done")
