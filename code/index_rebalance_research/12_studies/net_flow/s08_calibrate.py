"""Step 8: calibrate the true-to-visible multiplier R per index family, and write pi_calibration.csv.

R is the slope of excess trade-day volume on holdings-implied VISIBLE passive demand, both in shares and both scaled
by the line's ADV60 (so every line counts on a comparable scale):
    (V_T - ADV60) / ADV60 = a_T + sum_f R_f |D_vis,f| / ADV60 + b1 log mcap + b2 log(ADV60 / shares out)
                            + price-bucket dummies + e
with event-date fixed effects a_T (they absorb the quad-witching / same-close volume of everything else), y and x
winsorised at 1% / 99%, SEs clustered by event date. If all of the trade-day excess volume caused by index trading
were proportional to visible demand, R = true passive demand / visible demand (plus liquidity-provider round trips;
the two cannot be separated with daily volume). Robustness: the same in shares-outstanding units; adds/deletes only;
weight changes only; 2016+; and the audit-style median ratio (V_T - median V_{T-20..T-1}) / |D_vis| for additions.
Samples:
  TSX, TSX60  Canada, daily-mode review pairs (XIC/XIU T-1 -> T), consolidated volume from 2014 (TSX-only before)
  R2000, R1000, R3000  Russell June recon 2011-2025 (month-end pairs), CRSP consolidated volume
  SP500       IVV month-end pairs around the quarterly rebalance (weight changes) and additions at their own T
  STOXX, DAX, MDAX, TecDAX, SDAX, FTSE100, FTSE250   quarter-end s12 pairs, PRIMARY-venue volume (understates)
pi_calibration.csv columns (stable; read by 16_flow_engine):
  family      index family code: TSX, TSX60, SP500, R1000, R2000, R3000, STOXX, DAX, MDAX, SDAX, TecDAX, FTSE100, FTSE250
  date        observation date (each family's quarterly review trade dates T and every quarter-end, plus the last date
              with tracker data)
  pi_visible  visible passive share: AUM of the headline physical trackers / index float MV on that date (s03)
  R           calibrated multiplier true / visible (constant per family; see R_window)
  pi_true     R x pi_visible
  source      where pi_visible and R come from
  R_se, R_n, R_window   standard error, observations and sample window of the R fit; R_basis = volume basis
Outputs: table_R.csv (all fits), pi_calibration.csv.
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, fe_ols, winsor

FAM_SRC = {"TSX": "Datastream AUM XIC+ZCN+HXCN / MV 48552", "TSX60": "Datastream AUM XIU+ZIU+HXT / MV 48553",
           "SP500": "Datastream AUM SPY+IVV+VOO / MV 41620", "STOXX": "Datastream AUM 7 physical STOXX 600 UCITS / MV 13635",
           "DAX": "Datastream AUM 13 DAX ETFs / MV 13095", "MDAX": "Datastream AUM 6 MDAX ETFs / MV 40753",
           "SDAX": "Datastream AUM 2 SDAX ETFs / MV 46391", "TecDAX": "Datastream AUM 3 TecDAX ETFs / MV 41379",
           "FTSE100": "Datastream AUM ISF+VUKE / MV 36888", "FTSE250": "Datastream AUM MIDD+VMID / MV 36890",
           "R2000": "CRSP MF TNA IWM+VTWO / MV implied by IWM holdings", "R1000": "CRSP MF TNA IWB+VONE / MV implied by IWB holdings",
           "R3000": "CRSP MF TNA IWV+VTHR / MV implied by IWV holdings"}
VOL_BASIS = {"TSX": "CA consolidated (TSX-only before 2014)", "TSX60": "CA consolidated (TSX-only before 2014)",
             "SP500": "CRSP consolidated", "R2000": "CRSP consolidated", "R1000": "CRSP consolidated", "R3000": "CRSP consolidated"}


def prep(d, xcols, price_cut=5.0):
    d = d[(d.adv60 > 0) & d.vol_T.notna() & (d.shares_out > 0) & (d.price_Tm1 > 0)].copy()
    d["y"] = winsor((d.vol_T - d.adv60) / d.adv60)
    for c in xcols:
        x = (d[c].abs() / d.adv60).values
        hi = np.nanquantile(x, 0.99)
        d["x_" + c] = np.clip(x, 0, hi)
        xf = (d[c].abs() / d.shares_out).values
        d["xf_" + c] = np.clip(xf, 0, np.nanquantile(xf, 0.99))
    d["yf"] = winsor((d.vol_T - d.adv60) / d.shares_out)
    d["lmcap"] = np.log(d.price_Tm1 * d.shares_out)
    d["lturn"] = np.log(d.adv60 / d.shares_out)
    d["pb_lo"] = (d.price_Tm1 < price_cut).astype(float)
    d["pb_mid"] = ((d.price_Tm1 >= price_cut) & (d.price_Tm1 < 4 * price_cut)).astype(float)
    return d


def fit(d, xcols, label, fam_names, units="adv", date_col="T"):
    xs = [("x_" if units == "adv" else "xf_") + c for c in xcols]
    yv = d.y if units == "adv" else d.yf
    X = d[xs + ["lmcap", "lturn", "pb_lo", "pb_mid"]].values
    tab, (n, G, r2) = fe_ols(yv.values, X, d[date_col].astype(str).values, d[date_col].astype(str).values,
                             xs + ["lmcap", "lturn", "pb_lo", "pb_mid"])
    out = []
    if tab is None:
        return out
    for c, fam in zip(xs, fam_names):
        t = tab.set_index("term").loc[c]
        out.append(dict(sample=label, family=fam, units=units, R=t.coef, se=t.se, t=t.t, n=n, dates=G, r2_within=r2))
    return out


WINDOWS = {"CA": [(2012, 2015), (2016, 2019), (2020, 2022), (2023, 2026)],
           "US": [(2010, 2015), (2016, 2019), (2020, 2022), (2023, 2025)],
           "EU": [(2018, 2020), (2021, 2022), (2023, 2025)]}


def windows(d, xcols, fam_names, mkt, label):
    out = []
    for a, b in WINDOWS[mkt]:
        x = d[(d["T"].dt.year >= a) & (d["T"].dt.year <= b)]
        if x["T"].nunique() < 3:
            continue
        for r in fit(x, xcols, f"{label} {a}-{b}", fam_names):
            r["window"] = f"{a}-{b}"
            out.append(r)
    return out


def elasticity(d, col, label, fam):
    """log(excess / ADV) on log(|D| / ADV) with date FE, lines with positive excess and demand (shape check)."""
    x = d[(d.vol_T > d.adv60) & (d[col].abs() > 0)].copy()
    yv = np.log((x.vol_T - x.adv60) / x.adv60).values
    xv = np.log(x[col].abs() / x.adv60).values
    tab, (n, G, r2) = fe_ols(yv, np.column_stack([xv, x.lmcap, x.lturn]), x["T"].astype(str).values, x["T"].astype(str).values,
                             ["lx", "lmcap", "lturn"])
    t = tab.set_index("term").loc["lx"]
    return dict(sample=label, family=fam, units="log-log elasticity", R=t.coef, se=t.se, t=t.t, n=n, dates=G, r2_within=r2)


def med_ratio(d, col, label, fam):
    x = d[(d[col].abs() > 0) & d.med20.notna()]
    r = (x.vol_T - x.med20) / x[col].abs()
    return dict(sample=label, family=fam, units="median ratio (V_T - med20)/|D_vis|", R=float(np.median(r)),
                se=np.nan, t=np.nan, n=len(r), dates=x["T"].nunique() if "T" in x else np.nan,
                r2_within=np.nan, iqr=f"{np.quantile(r, .25):.2f}-{np.quantile(r, .75):.2f}")


def canada():
    D = pd.read_csv(OUT / "ca_demand.csv", parse_dates=["T"])
    D = D[D.mode_TSX.str.startswith("daily") & D.event.isin(["add", "delete", "tsx60_add", "tsx60_delete", "weight", "unchanged"])]
    d = prep(D, ["D_vis_TSX_sh", "D_vis_TSX60_sh", "D_net_sh"], price_cut=5.0)
    rows = []
    rows += fit(d, ["D_vis_TSX_sh", "D_vis_TSX60_sh"], "CA daily 2012-2026, all lines", ["TSX", "TSX60"])
    rows += fit(d, ["D_net_sh"], "CA daily, all lines, net demand", ["CA net"])
    rows += fit(d, ["D_vis_TSX_sh", "D_vis_TSX60_sh"], "CA daily, all lines", ["TSX", "TSX60"], units="float")
    ad = d[d.event.isin(["add", "delete", "tsx60_add", "tsx60_delete"])]
    rows += fit(ad, ["D_vis_TSX_sh", "D_vis_TSX60_sh"], "CA daily, adds/deletes/TSX60 moves", ["TSX", "TSX60"])
    wc = d[d.event.isin(["weight", "unchanged"])]
    rows += fit(wc, ["D_net_sh"], "CA daily, existing members (weight + unchanged)", ["CA net"])
    rows += fit(d[d["T"] >= "2016-01-01"], ["D_vis_TSX_sh", "D_vis_TSX60_sh"], "CA daily 2016+, all lines", ["TSX", "TSX60"])
    rows += windows(d, ["D_vis_TSX_sh", "D_vis_TSX60_sh"], ["TSX", "TSX60"], "CA", "CA daily, all lines,")
    rows += windows(d, ["D_net_sh"], ["CA net"], "CA", "CA daily, all lines, net demand,")
    x = d[d.event.isin(["tsx60_add", "tsx60_delete"])]
    rows.append(med_ratio(x, "D_vis_TSX60_sh", "CA daily, TSX 60 moves (audit-style, TSX 60 visible demand)", "TSX60"))
    rows.append(elasticity(d, "D_net_sh", "CA daily, all lines with excess volume", "CA net"))
    rows.append(elasticity(d[d.event.isin(["add", "delete"])], "D_net_sh", "CA daily, adds/deletes with excess volume", "CA net"))
    rows.append(med_ratio(d[d.event == "add"], "D_net_sh", "CA daily, Composite adds (audit-style)", "TSX"))
    for a, b in WINDOWS["CA"]:
        x = d[(d.event == "add") & (d["T"].dt.year >= a) & (d["T"].dt.year <= b)]
        if len(x) >= 10:
            rows.append(med_ratio(x, "D_net_sh", f"CA daily, Composite adds {a}-{b} (audit-style)", "TSX") | {"window": f"{a}-{b}"})
    rows.append(med_ratio(d[(d.event == "add") & (d["T"] >= "2024-01-01")], "D_net_sh", "CA daily, Composite adds 2024+ (audit-style)", "TSX"))
    return rows


def russell():
    D = pd.read_csv(OUT / "us_demand.csv", parse_dates=["T"])
    D = D[D.event.isin(["BUY_add", "SELL_del", "BUY_mig", "SELL_mig", "weight", "unchanged"])]
    d = prep(D, ["D_vis_R2000", "D_vis_R1000", "D_vis_R3000", "D_net"], price_cut=5.0)
    rows = []
    rows += fit(d, ["D_vis_R2000", "D_vis_R1000", "D_vis_R3000"], "US Russell recon 2011-2025, all lines", ["R2000", "R1000", "R3000"])
    rows += fit(d, ["D_net"], "US Russell, all lines, net demand", ["US Russell net"])
    rows += fit(d, ["D_vis_R2000", "D_vis_R1000", "D_vis_R3000"], "US Russell, all lines", ["R2000", "R1000", "R3000"], units="float")
    ad = d[d.event.isin(["BUY_add", "SELL_del", "BUY_mig", "SELL_mig"])]
    rows += fit(ad, ["D_vis_R2000", "D_vis_R1000", "D_vis_R3000"], "US Russell, adds/deletes/migrations", ["R2000", "R1000", "R3000"])
    wc = d[d.event.isin(["weight", "unchanged"])]
    rows += fit(wc, ["D_net"], "US Russell, existing members (weight + unchanged)", ["US Russell net"])
    rows += fit(d[d["T"] >= "2016-01-01"], ["D_vis_R2000", "D_vis_R1000", "D_vis_R3000"], "US Russell 2016+, all lines", ["R2000", "R1000", "R3000"])
    rows += windows(d, ["D_vis_R2000", "D_vis_R1000", "D_vis_R3000"], ["R2000", "R1000", "R3000"], "US", "US Russell, all lines,")
    rows.append(elasticity(d, "D_net", "US Russell, all lines with excess volume", "US Russell net"))
    rows.append(med_ratio(d[d.event == "BUY_add"], "D_net", "US Russell, R2000 adds (audit-style)", "R2000"))
    return rows


def sp500():
    Q = pd.read_csv(OUT / "us_sp500_demand.csv", parse_dates=["T"])
    Q = Q[Q.cls.isin(["weight", "unchanged"])]
    d = prep(Q, ["D_vis"], price_cut=5.0)
    rows = fit(d, ["D_vis"], "S&P 500 quarterly rebalances 2010-2025, existing members", ["SP500"])
    rows += windows(d, ["D_vis"], ["SP500"], "US", "S&P 500 quarterly, existing members,")
    A = pd.read_csv(OUT / "us_sp500_adds.csv", parse_dates=["T"])
    A = A[(A.adv60 > 0) & A.vol_T.notna()]
    rows.append(med_ratio(A, "D_vis", "S&P 500 additions 2010-2025 (audit-style)", "SP500"))
    rows.append(med_ratio(A[A["T"] >= "2024-01-01"], "D_vis", "S&P 500 additions 2024-25 (audit-style, as flow_sizing §4.4)", "SP500"))
    a = A.copy()
    a["y"] = (a.vol_T - a.adv60)
    rows.append(dict(sample="S&P 500 additions, OLS through origin of (V_T-ADV60) on |D_vis|", family="SP500", units="shares",
                     R=float((a.y * a.D_vis.abs()).sum() / (a.D_vis ** 2).sum()), se=np.nan, t=np.nan, n=len(a),
                     dates=a["T"].nunique(), r2_within=np.nan))
    return rows


def europe():
    D = pd.read_csv(OUT / "eu_demand.csv", parse_dates=["T"])
    D = D[D.col >= 0]
    fams = ["STOXX", "DAX", "MDAX", "TecDAX", "SDAX", "FTSE100", "FTSE250"]
    W = D.pivot_table(index=["T", "col"], columns="family", values="D_vis_sh", aggfunc="sum").fillna(0.0)
    W.columns = ["D_" + c for c in W.columns]
    F = D.groupby(["T", "col"]).agg(adv60=("adv60", "first"), vol_T=("vol_T", "first"), med20=("med20", "first"),
                                    price_Tm1=("price_Tm1", "first"), shares_out=("shares_out", "first"),
                                    cls=("cls", lambda s: "+".join(sorted(set(s)))))
    E = F.join(W).reset_index()
    for f in fams:
        if "D_" + f not in E:
            E["D_" + f] = 0.0
    E["D_net"] = E[["D_" + f for f in fams]].sum(1)
    d = prep(E, ["D_" + f for f in fams] + ["D_net"], price_cut=5.0)
    rows = fit(d, ["D_" + f for f in fams], "EU quarter-end pairs 2011-2025, all lines, primary volume", fams)
    rows += fit(d, ["D_net"], "EU, all lines, net demand, primary volume", ["EU net"])
    d["D_DAXfam"] = d[["D_DAX", "D_MDAX", "D_TecDAX", "D_SDAX"]].sum(1)
    d["D_FTSE"] = d[["D_FTSE100", "D_FTSE250"]].sum(1)
    for c in ("D_DAXfam", "D_FTSE"):
        x = (d[c].abs() / d.adv60).values
        d["x_" + c] = np.clip(x, 0, np.nanquantile(x, 0.99))
    rows += fit(d, ["D_STOXX", "D_DAXfam", "D_FTSE"], "EU, all lines, pooled DAX family and FTSE UK, primary volume",
                ["STOXX", "DAX family pooled", "FTSE UK pooled"])
    rows += windows(d, ["D_STOXX", "D_DAXfam", "D_FTSE"], ["STOXX", "DAX family pooled", "FTSE UK pooled"], "EU", "EU pooled, all lines,")
    rows.append(elasticity(d, "D_net", "EU, all lines with excess volume (primary)", "EU net"))
    ad = d[d.cls.str.contains("add|delete")]
    rows += fit(ad, ["D_" + f for f in fams], "EU, adds/deletes, primary volume", fams)
    rows.append(med_ratio(d[d.cls == "add"], "D_net", "EU STOXX-only adds (audit-style, primary volume)", "STOXX"))
    return rows


# R choice per family for pi_calibration.csv: (sample label prefix of the windowed fits, family label in table_R,
# fallback note). Windowed fits give a time-varying R; dates outside the windows take the nearest window.
RCHOICE = {"TSX": ("CA daily, all lines, net demand,", "CA net", "one R for both Canadian families (net demand fit)"),
           "TSX60": ("CA daily, all lines, net demand,", "CA net", "one R for both Canadian families (net demand fit); the separate TSX 60 slope is a large-cap premium collinear with Composite demand"),
           "R2000": ("US Russell, all lines,", "R2000", None),
           "R1000": ("S&P 500 quarterly, existing members,", "SP500", "R1000 not identified (t 1.2): S&P 500 R used"),
           "R3000": ("US Russell, all lines,", "R2000", "R3000 not identified separately from R2000 demand: R2000 R used"),
           "SP500": ("S&P 500 quarterly, existing members,", "SP500", None),
           "STOXX": ("EU pooled, all lines,", "STOXX", None)}
EU_FULL = "EU quarter-end pairs 2011-2025, all lines, primary volume"
EU_POOL = "EU, all lines, pooled DAX family and FTSE UK, primary volume"


def r_for(T, fam, d):
    """(R, se, n, window label, t, note) for family fam on date d."""
    if fam in RCHOICE:
        pre, lab, note = RCHOICE[fam]
        x = T[(T["sample"].str.startswith(pre)) & (T.family == lab) & (T.units == "adv") & T.window.notna()].copy()
        x["a"] = x.window.str[:4].astype(int); x["b"] = x.window.str[5:].astype(int)
        y = d.year
        hit = x[(x.a <= y) & (x.b >= y)]
        if hit.empty and y < x.a.min():
            return np.nan, np.nan, np.nan, "", np.nan, "before the first calibration window: not extrapolated backward"
        if hit.empty:
            hit = x.iloc[[-1]]
            note = (note + "; " if note else "") + "after the last fitted window: latest window used"
        r = hit.iloc[0]
        return r.R, r.se, r.n, r["sample"], r.t, note
    # Europe, single families (2018-2025 only): full-sample fit; TecDAX and SDAX not identified -> pooled DAX family
    if fam in ("TecDAX", "SDAX"):
        if d.year < 2018:
            return np.nan, np.nan, np.nan, "", np.nan, "before the 2018-2025 calibration window: not extrapolated backward"
        r = T[(T["sample"] == EU_POOL) & (T.family == "DAX family pooled")].iloc[0]
        return r.R, r.se, r.n, r["sample"], r.t, f"{fam} not identified (t < 1): pooled DAX-family R used"
    if d.year < 2018:
        return np.nan, np.nan, np.nan, "", np.nan, "before the 2018-2025 calibration window: not extrapolated backward"
    r = T[(T["sample"] == EU_FULL) & (T.family == fam) & (T.units == "adv")].iloc[0]
    return r.R, r.se, r.n, r["sample"], r.t, None


def pi_csv(T):
    """pi_calibration.csv: one row per family x date (review trade dates, quarter-ends, last tracker date)."""
    pd_ = pd.read_parquet(DATA / "pi_daily.parquet")
    pm = pd.read_parquet(DATA / "pi_us_monthly.parquet")
    rows = []
    qe = pd.date_range("2006-03-31", "2026-06-30", freq="QE")
    tds = []
    for y in range(2006, 2027):
        for m in (3, 6, 9, 12):
            d1 = pd.Timestamp(y, m, 1)
            tds.append(d1 + pd.Timedelta(days=(4 - d1.weekday()) % 7 + 14))

    def add(fam, d, piv, stale, src_pi):
        R, se, n, win, t, note = r_for(T, fam, pd.Timestamp(d))
        src = f"pi: {src_pi}" + (f" (stale Datastream share counts: {stale:.0%} of AUM)" if stale > 0.05 else "")
        src += (f"; R: {win}, t {t:.1f}" if np.isfinite(R) else "; R: none") + (f" [{note}]" if note else "")
        rows.append(dict(family=fam, date=pd.Timestamp(d).date(), pi_visible=piv, R=R, pi_true=piv * R, source=src,
                         R_se=se, R_n=n, R_window=win, R_basis=VOL_BASIS.get(fam, "primary-venue volume (understates)")))

    for fam in ["TSX", "TSX60", "SP500", "STOXX", "DAX", "MDAX", "SDAX", "TecDAX", "FTSE100", "FTSE250"]:
        s = pd_[pd_.family == fam].set_index("date").sort_index()
        dates = sorted(set([d for d in tds + list(qe) if s.index.min() <= d <= s.index.max()] + [s.index.max()]))
        for d in dates:
            k = s.index.searchsorted(d, side="right") - 1
            if k < 0:
                continue
            v = s.iloc[k]
            add(fam, s.index[k], v.pi_visible, v.stale_share, FAM_SRC[fam])
    for fam in ["R2000", "R1000", "R3000"]:
        s = pm[pm.family == fam].dropna(subset=["pi_etf"]).sort_values("date")
        for v in s.itertuples():
            add(fam, v.date, v.pi_etf, 0.0, FAM_SRC[fam] + " (month-end)")
    out = pd.DataFrame(rows).sort_values(["family", "date"]).drop_duplicates(["family", "date"])
    out.to_csv(OUT / "pi_calibration.csv", index=False)
    return out


SEL_FULL = {"TSX": ("CA daily, all lines, net demand", "CA net"), "TSX60": ("CA daily, all lines, net demand", "CA net"),
            "R2000": ("US Russell recon 2011-2025, all lines", "R2000"),
            "R1000": ("S&P 500 quarterly rebalances 2010-2025, existing members", "SP500"),
            "R3000": ("US Russell recon 2011-2025, all lines", "R2000"),
            "SP500": ("S&P 500 quarterly rebalances 2010-2025, existing members", "SP500"),
            "STOXX": ("EU quarter-end pairs 2011-2025, all lines, primary volume", "STOXX"),
            "DAX": ("EU quarter-end pairs 2011-2025, all lines, primary volume", "DAX"),
            "MDAX": ("EU quarter-end pairs 2011-2025, all lines, primary volume", "MDAX"),
            "TecDAX": ("EU, all lines, pooled DAX family and FTSE UK, primary volume", "DAX family pooled"),
            "SDAX": ("EU, all lines, pooled DAX family and FTSE UK, primary volume", "DAX family pooled"),
            "FTSE100": ("EU quarter-end pairs 2011-2025, all lines, primary volume", "FTSE100"),
            "FTSE250": ("EU quarter-end pairs 2011-2025, all lines, primary volume", "FTSE250")}


def selected(T):
    rows = []
    for fam, (smp, lab) in SEL_FULL.items():
        x = T[(T["sample"] == smp) & (T.family == lab) & (T.units == "adv")].iloc[0]
        rows.append(dict(family=fam, R=x.R, se=x.se, t=x.t, n=x.n, sample=smp, fitted_as=lab))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "table_R_selected.csv", index=False)
    return S


if __name__ == "__main__":
    rows = canada() + russell() + sp500() + europe()
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_R.csv", index=False)
    print(selected(T).round(3).to_string())
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(T[["sample", "family", "units", "R", "se", "t", "n", "dates"]].round(3).to_string())
    pc = pi_csv(T)
    print(pc.groupby("family").tail(1).to_string())
