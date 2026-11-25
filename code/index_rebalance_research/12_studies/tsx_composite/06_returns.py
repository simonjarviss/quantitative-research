"""Step 6: abnormal returns of capturable (predicted) and hindsight (actual) portfolios.

AR(i,t) = r(i,t) - r(cell,t), r = daily total return (Compustat prccd, ajexdi, trfd).
Cells: size decile x prior-6-month-momentum tercile, formed at the portfolio's formation date among
TSX-listed controls: Canadian, GICS-classified, non-Venture lines with a price in the last 10 sessions,
close >= C$1 and market cap >= 0.005% of the estimated index float cap, excluding every name predicted
(any horizon, any model variant) or actually changed at that review. Cell return = equal-weighted mean of
control returns (daily returns clipped to [-50%, +100%]). Fewer than 5 controls -> size-decile cell.
Index-adjusted AR = r(i,t) - r(index proxy,t) (cap-weighted Composite members, total return).
CAR(a,b] = sum of AR over sessions a+1..b; a name that stops trading contributes 0 afterwards.

Portfolios per review q:
  hind_<side>       actual quarterly-review changes, formed at H6 / H3 / R and at T-130 (for paths)
  pred_<side>       capturable: main model predictions at H6 / H3 / R (false positives included)
  rules_<side>      robustness: rules-only predictions
  naive_<side>      naive benchmark at each horizon
Outputs: car_events.csv, car_paths.csv
"""
import numpy as np, pandas as pd
from common import DATA, OUT, tsx_calendar, review_dates

cal = tsx_calendar()
rev = review_dates(cal).reset_index(drop=True)
z = np.load(DATA / "panel.npz")
P, RET, SH, AJ = z["P"], z["RET"], z["SH"], z["AJ"]
AJf = pd.DataFrame(AJ).ffill().values
nd, nl = P.shape
L = pd.read_parquet(DATA / "panel_lines.parquet")
lid_ix = pd.Index(L.lid)
MCAP = P * SH / AJf
R_cl = np.clip(RET, -0.5, 1.0)
TRI = np.nancumprod(np.nan_to_num(RET) + 1, axis=0)
mkt = pd.read_parquet(DATA / "index_proxy.parquet").proxy_tr.fillna(0).values
dom = ((L["loc"] == "CAN") | (L.fic == "CAN")).values & L.gsubind.notna().values & (L.exchg != 9).values
MEM60 = None
m = pd.read_parquet(DATA / "members.parquet")
m60 = m[m.gvkeyx == "118341"].copy()
m60["fromdate"] = pd.to_datetime(m60.fromdate); m60["thru"] = pd.to_datetime(m60.thru)

ev = pd.read_csv(OUT / "events.csv", parse_dates=["eff_date", "T"])
ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
ev["col"] = lid_ix.get_indexer(ev.lid)
pr = pd.read_csv(OUT / "predictions_final.csv", parse_dates=["date"])
diag = pd.read_csv(OUT / "prediction_diagnostics.csv")
Mtot = diag.set_index(["review", "horizon"]).M_bn * 1e9

WIN = [("H6", "H3"), ("H3", "R"), ("R", "A"), ("A", "Tm1"), ("Tm1", "T"), ("T", "Tp1"), ("T", "Tp20"), ("T", "Tp60"),
       ("H6", "T"), ("H3", "T"), ("R", "T"), ("A", "T"), ("Tp1", "Tp20"), ("Tp1", "Tp60")]
rows, paths = [], []


def cells_at(f, Mref, exclude):
    """size decile x momentum tercile codes for all lines at session f; control mask."""
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
    return sdec * 3 + mter, sdec, ctrl, mc, mom


for q in rev.itertuples():
    iT = cal.get_loc(q.T)
    pos = dict(H6=cal.get_loc(q.H6), H3=cal.get_loc(q.H3), R=cal.get_loc(q.R), A=cal.get_loc(q.A), Tm1=iT - 1, T=iT,
               Tp1=iT + 1, Tp20=iT + 20, Tp60=iT + 60, P130=iT - 130)
    lo, hi = iT - 135, min(iT + 60, nd - 1)
    e = ev[(ev.review == q.review) & (ev.etype == "review") & (ev.col >= 0)]
    groups = {"hind_add": e.loc[e.side == "add", "col"].values, "hind_del": e.loc[e.side == "delete", "col"].values}
    x = pr[pr.review == q.review]
    excl = np.zeros(nl, bool)
    excl[groups["hind_add"]] = True; excl[groups["hind_del"]] = True
    excl[x.loc[x.pred_add_rules | x.pred_del_rules | x.pred_add | x.pred_del, "col"].values] = True
    Mref = Mtot.get((q.review, "R"), np.nan)
    tp_add = set(groups["hind_add"]); tp_del = set(groups["hind_del"])
    for form in ["H6", "H3", "R", "P130"]:
        f = pos[form]
        code, sdec, ctrl, mc, mom = cells_at(f, Mref, excl)
        # control cell returns over the window
        cr = {}
        for c in np.unique(code[ctrl]):
            mbr = ctrl & (code == c)
            if mbr.sum() >= 5:
                cr[c] = np.nanmean(R_cl[lo:hi + 1, mbr], 1)
        dr = {d: np.nanmean(R_cl[lo:hi + 1, ctrl & (sdec == d)], 1) for d in np.unique(sdec[ctrl])}
        ports = {}
        hz = form if form != "P130" else "R"
        xs = x[x.horizon == hz]
        if form == "P130":
            ports["hind_add"] = groups["hind_add"]; ports["hind_del"] = groups["hind_del"]
            for side, col in [("add", "pred_add"), ("del", "pred_del")]:
                for h in ["H6", "H3", "R"]:
                    ports[f"pred{h}_{side}"] = x.loc[(x.horizon == h) & x[col], "col"].values
        else:
            ports["hind_add"] = groups["hind_add"]; ports["hind_del"] = groups["hind_del"]
            ports["pred_add"] = xs.loc[xs.pred_add, "col"].values; ports["pred_del"] = xs.loc[xs.pred_del, "col"].values
            ports["rules_add"] = xs.loc[xs.pred_add_rules, "col"].values; ports["rules_del"] = xs.loc[xs.pred_del_rules, "col"].values
            ports["naive_add"] = xs.loc[xs.naive_add, "col"].values; ports["naive_del"] = xs.loc[xs.naive_del, "col"].values
        in60 = set(lid_ix.get_indexer(m60[(m60.fromdate <= cal[f]) & (m60.thru.isna() | (m60.thru >= cal[f]))].gvkey + "_" +
                                      m60[(m60.fromdate <= cal[f]) & (m60.thru.isna() | (m60.thru >= cal[f]))].iid))
        for pname, cols in ports.items():
            for c in cols:
                ri = R_cl[lo:hi + 1, c]
                cc = cr.get(code[c], dr.get(sdec[c], np.full(hi - lo + 1, np.nan)))
                ar = np.nan_to_num(ri - cc)          # missing stock return -> 0 (position ended)
                ari = np.nan_to_num(ri - mkt[lo:hi + 1])
                has = ~np.isnan(ri)
                cum = np.concatenate([[0], np.cumsum(ar)]); cumi = np.concatenate([[0], np.cumsum(ari)])
                # cum[k] = sum of AR over sessions lo..lo+k-1 -> CAR(a,b] = cum[b-lo+1] - cum[a-lo+1]
                def car(a, b, cm):
                    if pos[b] > hi or pos[a] < lo or pos[a] >= pos[b]:
                        return np.nan
                    return cm[pos[b] - lo + 1] - cm[pos[a] - lo + 1]
                row = dict(review=q.review, year=q.year, form=form, portfolio=pname, col=c, lid=L.lid[c], tic=L.tic[c],
                           hit=(c in tp_add) if pname.endswith("add") else (c in tp_del),
                           mcap=mc[c], size_rel=mc[c] / Mref if Mref else np.nan, mom6=mom[c], sdec=sdec[c],
                           tsx60=c in in60, n_ret_form_T=int(has[max(0, pos[form] - lo + 1):iT - lo + 1].sum()))
                for a, b in WIN:
                    row[f"car_{a}_{b}"] = car(a, b, cum) if pos[a] >= pos[form] or form == "P130" else np.nan
                    row[f"cari_{a}_{b}"] = car(a, b, cumi) if pos[a] >= pos[form] or form == "P130" else np.nan
                rows.append(row)
                if form == "P130":
                    k0 = iT - 130 - lo
                    seg = cum[k0 + 1:] - cum[k0 + 1]     # cumulative AR from close of T-130
                    segi = cumi[k0 + 1:] - cumi[k0 + 1]
                    ks = np.arange(-130, -130 + len(seg))
                    paths.append(pd.DataFrame({"review": q.review, "portfolio": pname, "col": c, "k": ks, "car": seg,
                                               "cari": segi}))
    print(q.review, len(rows))

ce = pd.DataFrame(rows)
ce.to_csv(OUT / "car_events.csv", index=False)
pa = pd.concat(paths, ignore_index=True)
pa = pa[pa.k <= 60]
pa["period"] = np.where(pa.review.str[:4].astype(int) <= 2015, "2006-2015", "2016-2026")
outs = []
for per, d in [("2006-2026", pa), ("2006-2015", pa[pa.period == "2006-2015"]), ("2016-2026", pa[pa.period == "2016-2026"])]:
    g = d.groupby(["portfolio", "k"])
    o = pd.DataFrame({"mean_car": g.car.mean(), "median_car": g.car.median(), "n": g.car.size(),
                      "mean_car_index_adj": g.cari.mean()}).reset_index()
    o.insert(0, "period", per)
    outs.append(o)
pd.concat(outs).to_csv(OUT / "car_paths.csv", index=False)
