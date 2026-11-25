"""Step 7: abnormal returns of capturable (predicted) and hindsight (actual) portfolios, STOXX 600 and DAX family.

AR(i,t) = r(i,t) - r(cell,t), r = daily total return in local currency (Compustat Global prccd, ajexdi, trfd).
Cells (formed at the portfolio's formation date, among controls):
  STOXX 600: controls = one home line per company in a STOXX Europe country, operating company, priced in the
     last 10 sessions, full EUR market cap within [0.15x, 6x] of the review's rank-550 free-float cap (C screen),
     excluding every company predicted (any horizon, any variant, naive included) or changed at that review.
     Cell = country x size decile x 6-month-momentum tercile; fallbacks (>= 5 controls each): country x size
     quintile x momentum, currency bloc x size decile x momentum, bloc x size decile, all x size decile x momentum,
     all x size decile.
  DAX family: controls = German operating companies (home listing in Germany) with full market cap >= EUR 100m,
     not predicted or changed in the DAX family at that review. Cell = size decile x momentum tercile; fallback size
     decile. (All one country.)
Cell return = equal-weighted mean of control returns (daily returns clipped to [-50%, +100%]).
Index-adjusted AR = r(i,t) - index: STOXX = cap-weighted member proxy in local currency (03_panel.py);
  DAX family = DAX (moves touching the DAX), MDAX (MDAX/SDAX moves), TecDAX (TecDAX-only moves), all total-return
  levels from Compustat Global.
CAR(a,b] = sum of AR over sessions a+1..b; a name that stops trading contributes 0 afterwards.
Formation points: H6, H3, C (cut-off) and T-130 (paths). Segments as in the TSX study, with R -> C.
Portfolios per side: hind (actual changes), pred (rules model), mem (rules + memory), naive (size benchmark) and, for
STOXX at H3 and C, v0 (rules without the eligibility proxies of 05_predict_stoxx.py, which were chosen after seeing
the v0 false positives).
Usage: python3 07_returns.py [stoxx] [dax]
Outputs: car_events_stoxx.csv, car_events_dax.csv, car_paths.csv (data/paths_*.parquet intermediate)
"""
import numpy as np, pandas as pd
from common import DATA, OUT, eu_calendar, xetra_calendar, review_dates, stoxx_country_ok, bloc, buy_tax

cal = eu_calendar()
z = np.load(DATA / "panel.npz")
P, RET, SH, AJ, FX, VAL = z["P"], z["RET"], z["SH"], z["AJ"], z["FX"], z["VAL"]
AJf = pd.DataFrame(AJ).ffill().values
nd, nl = P.shape
L = pd.read_parquet(DATA / "panel_lines.parquet")
lid_ix = pd.Index(L.lid)
MCAP = P * SH / AJf * FX
R_cl = np.clip(RET, -0.5, 1.0)
TRI = np.nancumprod(np.nan_to_num(RET) + 1, axis=0)
ip = pd.read_parquet(DATA / "index_proxy.parquet")
IDX = {k: ip[k].fillna(0).values for k in ["stoxx_proxy_tr_local", "dax_tr", "mdax_tr", "tecdax_tr"]}
operating = L.gsector.notna().values & (L.tpci == "0").values
cty = L.home_cty.values
gv = L.gvkey.values
WIN = [("H6", "H3"), ("H3", "C"), ("C", "A"), ("A", "Tm1"), ("Tm1", "T"), ("T", "Tp1"), ("T", "Tp20"), ("T", "Tp60"),
       ("H6", "T"), ("H3", "T"), ("C", "T"), ("A", "T"), ("Tp1", "Tp20"), ("Tp1", "Tp60")]


def one_line_per_company(mask, f):
    """Among lines in mask, keep the most liquid line (3-month primary value traded) of each company."""
    adtv = np.nan_to_num(np.nanmean(VAL[max(0, f - 62):f + 1], 0), nan=-1)
    idx = np.where(mask)[0]
    df = pd.DataFrame({"g": gv[idx], "c": idx, "a": adtv[idx]}).sort_values(["g", "a"], ascending=[True, False])
    m = np.zeros(nl, bool); m[df.drop_duplicates("g").c.values] = True
    return m


def cells(f, univ, ctrl, group_key):
    """Codes for the fallback chain. group_key: array of country codes (or None for one group)."""
    mc = MCAP[f]
    mc = np.where(np.isnan(mc), pd.DataFrame(MCAP[max(0, f - 9):f + 1]).ffill().values[-1], mc)
    mom = TRI[f] / TRI[max(0, f - 126)] - 1
    sb = np.nanquantile(mc[univ], np.linspace(0.1, 0.9, 9))
    mb = np.nanquantile(mom[univ], [1 / 3, 2 / 3])
    sdec = np.searchsorted(sb, np.nan_to_num(mc, nan=0.0))
    squi = sdec // 2
    mter = np.searchsorted(mb, np.nan_to_num(mom, nan=0.0))
    return mc, mom, sdec, squi, mter


def cell_returns(lo, hi, ctrl, keys):
    out = {}
    df = pd.DataFrame({"k": keys[ctrl], "c": np.where(ctrl)[0]})
    for k, g in df.groupby("k"):
        if len(g) >= 5:
            out[k] = np.nanmean(R_cl[lo:hi + 1][:, g.c.values], 1)
    return out


def run(family):
    if family == "stoxx":
        cal_f = cal
        rev = review_dates(cal, "stoxx")
        ev = pd.read_csv(OUT / "events_stoxx.csv", dtype={"gvkey": str})
        ev = ev[(ev.etype == "review") & (ev.col >= 0)]
        pr = pd.read_csv(OUT / "predictions_stoxx.csv", dtype={"gvkey": str})
        diag = pd.read_csv(OUT / "prediction_diagnostics_stoxx.csv").set_index(["review", "horizon"])
        sides = [("add", "add"), ("del", "delete")]
    else:
        rev = review_dates(xetra_calendar(), "dax")
        ev = pd.read_csv(OUT / "events_dax.csv", dtype={"gvkey": str})
        ev = ev[(ev.etype == "review") & (ev.col >= 0)]
        rev = rev[rev.review <= ev.review.max()]
        pr = pd.read_csv(OUT / "predictions_dax.csv", dtype={"gvkey": str})
        sides = [("up", "up"), ("down", "down")]
    rows, paths = [], []
    for q in rev.itertuples():
        iT = cal.get_loc(q.T)
        pos = dict(H6=cal.get_loc(q.H6), H3=cal.get_loc(q.H3), C=cal.get_loc(q.C), A=cal.get_loc(q.A), Tm1=iT - 1, T=iT,
                   Tp1=iT + 1, Tp20=iT + 20, Tp60=iT + 60, P130=iT - 130)
        lo, hi = iT - 135, min(iT + 60, nd - 1)
        e = ev[ev.review == q.review]
        x = pr[pr.review == q.review]
        if family == "stoxx":
            groups = {"add": e.loc[e.side == "add", "col"].values, "del": e.loc[e.side == "delete", "col"].values}
            flagcols = [c for c in x.columns if c.startswith(("pred_", "naive_"))]
            excl_g = set(x.loc[x[flagcols].any(axis=1), "gvkey"]) | set(e.gvkey)
            L550 = diag.loc[(q.review, "C"), "L550_bn"] * 1e9
            ctry_ok = np.array([stoxx_country_ok(c, q.review) for c in cty])
        else:
            groups = {"up": e.loc[e.direction == "up", "col"].values, "down": e.loc[e.direction == "down", "col"].values}
            excl_g = set(x.loc[x.pred_up | x.pred_down | x.naive_up | x.naive_down | x.pred_up_mem | x.pred_down_mem, "gvkey"]) | set(e.gvkey)
            # index benchmark per event line
            bench = {}
            for r_ in e.itertuples():
                bench[r_.col] = "dax_tr" if 3 in (r_.lvl_pre, r_.lvl_post) else \
                    ("mdax_tr" if max(r_.lvl_pre, r_.lvl_post) > 0 else "tecdax_tr")
        excl = np.isin(gv, list(excl_g))
        ev_g = dict(zip(e.col, e.gvkey))
        for form in ["H6", "H3", "C", "P130"]:
            f = pos[form]
            if f - 126 < 0 or f < lo:
                continue
            live = (~np.isnan(P[max(0, f - 9):f + 1])).any(0)
            mc0 = MCAP[f]
            if family == "stoxx":
                base = live & operating & L.home_line.values & ctry_ok
                base = one_line_per_company(base, f)
                univ = base & (mc0 >= 0.15 * L550) & (mc0 <= 6 * L550)
                keyset = cty
            else:
                base = live & operating & L.home_line.values & (cty == "DEU")
                base = one_line_per_company(base, f)
                univ = base & (mc0 >= 1e8)
                keyset = np.full(nl, "DEU", dtype=object)
            ok_hist = (~np.isnan(P[max(0, f - 126):f + 1])).sum(0) >= 60
            ctrl = univ & ~excl & ok_hist
            mc, mom, sdec, squi, mter = cells(f, univ, ctrl, keyset)
            blocs = np.array([bloc(c) for c in keyset])
            chain = []
            if family == "stoxx":
                chain = [("cty_dec_mom", np.char.add(np.char.add(keyset.astype(str), "|"), (sdec * 3 + mter).astype(str))),
                         ("cty_qui_mom", np.char.add(np.char.add(keyset.astype(str), "|q"), (squi * 3 + mter).astype(str))),
                         ("bloc_dec_mom", np.char.add(np.char.add(blocs.astype(str), "|"), (sdec * 3 + mter).astype(str))),
                         ("bloc_dec", np.char.add(np.char.add(blocs.astype(str), "|d"), sdec.astype(str))),
                         ("all_dec_mom", (sdec * 3 + mter).astype(str)),
                         ("all_dec", np.char.add("d", sdec.astype(str)))]
            else:
                chain = [("dec_mom", (sdec * 3 + mter).astype(str)), ("dec", np.char.add("d", sdec.astype(str)))]
            cr = [(nm, keys, cell_returns(lo, hi, ctrl, keys)) for nm, keys in chain]
            hz = form if form != "P130" else "C"
            xs = x[x.horizon == hz]
            ports = {}
            for s, _ in sides:
                ports[f"hind_{s}"] = groups[s]
            if form == "P130":
                for s, _ in sides:
                    for h in ["H6", "H3", "C"]:
                        ports[f"pred{h}_{s}"] = x.loc[(x.horizon == h) & x[f"pred_{s}"], "col"].values
            else:
                for s, _ in sides:
                    ports[f"pred_{s}"] = xs.loc[xs[f"pred_{s}"], "col"].values
                    ports[f"mem_{s}"] = xs.loc[xs[f"pred_{s}_mem"], "col"].values
                    ports[f"naive_{s}"] = xs.loc[xs[f"naive_{s}"], "col"].values
                    if family == "stoxx" and form != "H6":      # v0 = rules without the post-hoc eligibility proxies
                        ports[f"v0_{s}"] = xs.loc[xs[f"pred_{s}_v0"], "col"].values
            hit_sets = {s: set(groups[s]) for s, _ in sides}
            for pname, cols in ports.items():
                s = pname.split("_")[-1]
                for c in np.unique(cols):
                    ri = R_cl[lo:hi + 1, c]
                    cc, lvl_used = None, "none"
                    for nm, keys, crd in cr:
                        if keys[c] in crd:
                            cc, lvl_used = crd[keys[c]], nm
                            break
                    if cc is None:
                        cc = np.full(hi - lo + 1, np.nan)
                    ar = np.nan_to_num(ri - cc)
                    if family == "stoxx":
                        bk = "stoxx_proxy_tr_local"
                    else:
                        bk = bench.get(c, "mdax_tr")
                    ari = np.nan_to_num(ri - IDX[bk][lo:hi + 1])
                    cum = np.concatenate([[0], np.cumsum(ar)]); cumi = np.concatenate([[0], np.cumsum(ari)])

                    def car(a, b, cm):
                        if pos[b] > hi or pos[a] < lo or pos[a] >= pos[b]:
                            return np.nan
                        return cm[pos[b] - lo + 1] - cm[pos[a] - lo + 1]
                    xr = x[(x.col == c)]
                    row = dict(review=q.review, year=q.year, form=form, portfolio=pname, col=c, lid=L.lid[c], gvkey=gv[c],
                               conm=L.conm[c], cty=cty[c], hit=c in hit_sets[s], mcap=mc[c], mom6=mom[c], sdec=sdec[c],
                               cell=lvl_used, n_ctrl=int(ctrl.sum()), tax=buy_tax(cty[c], cal[f]),
                               ff_src=xr.ff_src.iloc[0] if len(xr) else "")
                    if family == "dax":
                        row["bench"] = bk
                        er = e[e.col == c]
                        row["move"] = (er.move.fillna("").iloc[0] + " " + er.tec.fillna("").iloc[0]).strip() if len(er) else ""
                    for a, b in WIN:
                        ok = pos[a] >= pos[form] or form == "P130"
                        row[f"car_{a}_{b}"] = car(a, b, cum) if ok else np.nan
                        row[f"cari_{a}_{b}"] = car(a, b, cumi) if ok else np.nan
                    rows.append(row)
                    if form == "P130":
                        k0 = iT - 130 - lo
                        seg = cum[k0 + 1:] - cum[k0 + 1]; segi = cumi[k0 + 1:] - cumi[k0 + 1]
                        ks = np.arange(-130, -130 + len(seg))
                        paths.append(pd.DataFrame({"family": family, "review": q.review, "portfolio": pname, "col": c,
                                                   "k": ks, "car": seg, "cari": segi}))
        print(family, q.review, len(rows), flush=True)
    ce = pd.DataFrame(rows)
    ce.to_csv(OUT / f"car_events_{family}.csv", index=False)
    return pd.concat(paths, ignore_index=True)


import sys
fams = sys.argv[1:] or ["stoxx", "dax"]
for fam in fams:
    run(fam).to_parquet(DATA / f"paths_{fam}.parquet", index=False)
parts = [pd.read_parquet(DATA / f"paths_{f}.parquet") for f in ["stoxx", "dax"] if (DATA / f"paths_{f}.parquet").exists()]
pa = pd.concat(parts, ignore_index=True)
pa = pa[pa.k <= 60]
pa["period"] = np.where(pa.review.str[:4].astype(int) <= 2015, "2008-2015", "2016-2026")
outs = []
for per, d in [("2008-2026", pa), ("2008-2015", pa[pa.period == "2008-2015"]), ("2016-2026", pa[pa.period == "2016-2026"])]:
    g = d.groupby(["family", "portfolio", "k"])
    o = pd.DataFrame({"mean_car": g.car.mean(), "median_car": g.car.median(), "n": g.car.size(),
                      "mean_car_index_adj": g.cari.mean()}).reset_index()
    o.insert(0, "period", per)
    outs.append(o)
pd.concat(outs).to_csv(OUT / "car_paths.csv", index=False)
