"""Step 4: point-in-time predictions of each quarterly review's adds and deletes at H6, H3 and R.

At each horizon date d only data dated <= d is used (prices, volumes, shares, float estimates, membership
and review announcements already made). Rules: sp_tsx_composite.md §2.2/§10, parameters by era from
common.rules_for (change log §2.4; earlier years = current rule applied backward).

H3 and R: literal screen ("if the review were held today"), membership as known at d.
H6: a review (q-1) sits between d and review q. A literal screen at H6 predicts q-1, not q. For q we use the
    spec's Step 7 price-risk model: log relative size follows a driftless random walk with the name's
    trailing 120-session volatility relative to the index proxy. Predicted add at q if
    P(fail at R_{q-1} and pass at R_q) >= 0.10 and >= P(pass at R_{q-1}); deletes symmetric on the 0.025% buffer.
    Names blocked at q-1 (seasoning, re-entry ban) but eligible at q use P(pass at R_q) >= 0.5.
Naive benchmark: top-N non-members by estimated float cap and bottom-N members, N = mean count of
review adds (deletes) in the last 4 announced reviews (min 1).

Outputs: predictions.csv (every predicted, naive or actual name per review x horizon), accuracy tables.
Pass FLOAT=all100 in the environment to rerun with every float set to 100% (sensitivity).
"""
import os, numpy as np, pandas as pd
from scipy.stats import norm, multivariate_normal
from common import DATA, OUT, tsx_calendar, review_dates, rules_for

FLOAT_MODE = os.environ.get("FLOAT", "est")
cal = tsx_calendar()
rev = review_dates(cal).reset_index(drop=True)
rev["qn"] = np.arange(len(rev))
z = np.load(DATA / "panel.npz")
P, RET, VT, VC, VU, SH, AJ = (z[k] for k in ["P", "RET", "VT", "VC", "VU", "SH", "AJ"])
AJf = pd.DataFrame(AJ).ffill().values.astype(np.float32)
MEM = np.load(DATA / "member_matrix.npy")
L = pd.read_parquet(DATA / "panel_lines.parquet")
nd, nl = P.shape
ip = pd.read_parquet(DATA / "index_proxy.parquet")
mkt = ip.proxy_tr.fillna(0).values
fm = pd.read_parquet(DATA / "float_monthly.parquet")
ev = pd.read_csv(OUT / "events.csv", parse_dates=["eff_date", "T"])
ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
lid_ix = pd.Index(L.lid)
ev["col"] = lid_ix.get_indexer(ev.lid)
rq = ev[ev.etype == "review"].merge(rev[["review", "qn"]], on="review")

dom = ((L["loc"] == "CAN") | (L.fic == "CAN")).values
gics = L.gsubind.notna().values
not_v = (L.exchg != 9).values
has_px = ~np.isnan(P)
first_pos = np.where(has_px.any(0), has_px.argmax(0), nd)
first_date = cal[np.minimum(first_pos, nd - 1)]
Pa = P / AJf                                   # split-adjusted price
CAv = np.where(np.isnan(VC), VT, np.fmax(VC, VT))   # Canadian volume: consolidated where available
CAv = np.nan_to_num(CAv); VTn = np.nan_to_num(VT); VUn = np.nan_to_num(VU)

# float matrix lookup by month
fm["month"] = pd.to_datetime(fm.month)
months = pd.DatetimeIndex(sorted(fm.month.unique()))
FDS = np.full((len(months), nl), np.nan); FWS = np.full((len(months), nl), np.nan)
mi = months.get_indexer(fm.month)
FDS[mi, fm.line.values] = fm.ff_ds.values; FWS[mi, fm.line.values] = fm.ff_ws.values


def float_at(d):
    i = months.searchsorted(d, side="right") - 1
    ds, ws = FDS[i], FWS[i]
    ff = np.where(~np.isnan(ds), ds, np.where(~np.isnan(ws), ws, 1.0))
    src = np.where(~np.isnan(ds), "DS", np.where(~np.isnan(ws), "WS", "default100"))
    ff = np.round(ff, 2); ff = np.where(ff >= 0.96, 1.0, ff)          # S&P-style rounding (F10/F11)
    if FLOAT_MODE == "all100":
        ff = np.ones(nl); src = np.full(nl, "all100")
    return ff, src


def known_members(di, d):
    m = MEM[di].copy()
    for r in rev.itertuples():
        if r.A <= d < r.eff:                    # announced, not yet effective
            x = rq[rq.review == r.review]
            m[x.loc[(x.side == "add") & (x.col >= 0), "col"].values] = True
            m[x.loc[(x.side == "delete") & (x.col >= 0), "col"].values] = False
    return m


def seasoned_at(Rdate):
    lim = (pd.Timestamp(Rdate.year, Rdate.month, 1) - pd.DateOffset(months=6))
    return first_date <= lim


def banned_at(qn):
    b = np.zeros(nl, bool)
    x = rq[(rq.side == "delete") & (rq.qn >= qn - 4) & (rq.qn <= qn - 1) & (rq.col >= 0)]
    b[x.col.values] = True
    return b


def features(q, d, kind):
    """All screens at horizon date d for review q (row of rev). kind in H6/H3/R."""
    rules = rules_for(q.review)
    di = cal.get_loc(d)
    end = di - 1 if (kind == "R" and q.regime == "F1/R10") else di     # VWAP window before R (new regime)
    k = rules["vwap_days"]
    sl = slice(end - k + 1, end + 1)
    v = VTn[sl]; pa = Pa[sl]
    num = np.nansum(pa * v, 0); den = v.sum(0)
    vw_adj = np.where(den > 0, num / np.where(den > 0, den, 1), np.nanmean(pa, 0))
    aj = AJf[di]
    vw = vw_adj * aj                                                 # unadjusted C$ VWAP
    sl3 = slice(max(0, di - 62), di + 1)
    v3 = VTn[sl3]; p3 = Pa[sl3]
    d3 = v3.sum(0)
    vw3 = np.where(d3 > 0, np.nansum(p3 * v3, 0) / np.where(d3 > 0, d3, 1), np.nanmean(p3, 0)) * aj
    ff, src = float_at(d)
    sh = SH[di]
    qmv = ff * sh * vw_adj
    live = has_px[max(0, di - 9):di + 1].any(0) & ~np.isnan(qmv) & (qmv > 0)
    mem = known_members(di, d) & live
    # denominator, with the >10% Capped Composite switch (App. G reading, spec §10 step 3)
    M = np.nansum(qmv[mem]); wmax = np.nanmax(qmv[mem]) / M
    capped = 0
    if wmax > 0.10:
        q_sorted = np.sort(qmv[mem])[::-1]
        for kcap in range(1, 6):
            Mc = q_sorted[kcap:].sum() / (1 - 0.10 * kcap)
            if q_sorted[kcap] <= 0.10 * Mc:
                M, capped = Mc, kcap
                break
    w = np.where(mem, qmv / M, qmv / (M + qmv))
    # 12-month turnover on float shares
    d0 = cal.searchsorted(d - pd.Timedelta(days=365), side="right")
    ca12 = CAv[d0:di + 1].sum(0); us12 = VUn[d0:di + 1].sum(0)
    F = ff * sh
    allv = ca12 + (us12 if rules["us_volume"] else 0)
    turn = allv / F; turn_ca = ca12 / F
    dual = VUn[max(0, di - 62):di + 1].sum(0) > 0
    seas = seasoned_at(q.R); ban = banned_at(q.qn)
    elig = dom & gics & not_v & live
    price_in = (vw3 >= 1) & (vw >= 1)
    turn_in = (turn >= rules["turn_entry"]) & (~(dual & rules["us_volume"]) | (turn_ca >= rules["ca_only_entry"]))
    turn_st = (turn >= rules["turn_stay"]) & (~(dual & rules["us_volume"]) | (turn_ca >= rules["ca_only_stay"]))
    nonsize_in = elig & ~mem & seas & ~ban & price_in & turn_in
    nonsize_st = mem & (vw3 >= 1) & turn_st
    entry = nonsize_in & (w >= rules["size_entry"])
    stay_fail = mem & ~(nonsize_st & (w >= rules["size_stay"]))
    # trailing relative volatility for the H6 model
    r = RET[max(0, di - 119):di + 1]; mk = mkt[max(0, di - 119):di + 1][:, None]
    rel = np.log1p(np.clip(r, -0.95, 5)) - np.log1p(mk)
    nobs = (~np.isnan(rel)).sum(0)
    sig = np.where(nobs >= 60, np.nanstd(rel, 0), np.nan)
    sig = np.clip(np.where(np.isnan(sig), 0.03, sig), 0.005, 0.10)
    mcap = P[di] * sh / aj
    tri_now = np.nan  # momentum computed in the returns step
    return dict(di=di, w=w, M=M, capped=capped, wmax=wmax, qmv=qmv, ff=ff, src=src, turn=turn, turn_ca=turn_ca,
                dual=dual, mem=mem, elig=elig, seas=seas, ban=ban, vw=vw, vw3=vw3, live=live, entry=entry,
                stay_fail=stay_fail, nonsize_in=nonsize_in, nonsize_st=nonsize_st, sig=sig, mcap=mcap, rules=rules)


def bvn_cdf(a, b, rho):
    return multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(np.column_stack([a, b]))


rows = []
diag = []
for q in rev.itertuples():
    act_add = set(rq.loc[(rq.review == q.review) & (rq.side == "add"), "col"]) - {-1}
    act_del = set(rq.loc[(rq.review == q.review) & (rq.side == "delete"), "col"]) - {-1}
    prev = rev[(rev.A <= q.R) & (rev.qn < q.qn)].tail(4)
    for kind in ["H6", "H3", "R"]:
        d = getattr(q, kind)
        f = features(q, d, kind)
        rl = f["rules"]
        pred_add = f["entry"].copy(); pred_del = f["stay_fail"].copy()
        p_add = np.where(pred_add, 1.0, 0.0); p_del = np.where(pred_del, 1.0, 0.0)
        if kind == "H6":
            q1 = rev.iloc[q.qn - 1] if q.qn > 0 else None
            t2 = cal.get_loc(q.R) - f["di"]
            t1 = (cal.get_loc(q1.R) - f["di"]) if (q1 is not None and q1.R > d) else 0
            sig = f["sig"]
            # adds
            cand = f["nonsize_in"] & (f["w"] > 0)
            x0 = np.log(np.where(cand, f["w"], 1) / rl["size_entry"])
            el1 = f["seas"] if q1 is None else seasoned_at(q1.R) & ~banned_at(q1.qn)
            p = np.zeros(nl); p1 = np.zeros(nl)
            two = cand & el1 & (t1 > 0)
            one = cand & ~two
            if two.any():
                a = x0[two] / (sig[two] * np.sqrt(t1)); b = x0[two] / (sig[two] * np.sqrt(t2))
                p[two] = norm.cdf(-a) - bvn_cdf(-a, -b, np.sqrt(t1 / t2))
                p1[two] = norm.cdf(a)
            p[one] = norm.cdf(x0[one] / (sig[one] * np.sqrt(t2)))
            pred_add = (two & (p >= 0.10) & (p >= p1)) | (one & (p >= 0.5))
            p_add = p
            # deletes: members passing the non-size buffer tests now
            candd = f["mem"] & f["nonsize_st"] & (f["w"] > 0)
            y0 = np.log(np.where(candd, f["w"], 1) / rl["size_stay"])
            pd_ = np.zeros(nl); pd1 = np.zeros(nl)
            if t1 > 0 and candd.any():
                a = y0[candd] / (sig[candd] * np.sqrt(t1)); b = y0[candd] / (sig[candd] * np.sqrt(t2))
                pd_[candd] = norm.cdf(-b) - bvn_cdf(-a, -b, np.sqrt(t1 / t2))
                pd1[candd] = norm.cdf(-a)
                pred_del = candd & (pd_ >= 0.10) & (pd_ >= pd1)
            else:
                pd_[candd] = norm.cdf(-y0[candd] / (sig[candd] * np.sqrt(t2)))
                pred_del = candd & (pd_ >= 0.5)
            p_del = pd_
        # naive benchmark
        n_add = max(1, int(round(rq[rq.review.isin(prev.review) & (rq.side == "add")].shape[0] / max(len(prev), 1))))
        n_del = max(1, int(round(rq[rq.review.isin(prev.review) & (rq.side == "delete")].shape[0] / max(len(prev), 1))))
        pool = f["elig"] & ~f["mem"] & (f["vw"] >= 1)
        naive_add = np.zeros(nl, bool); naive_del = np.zeros(nl, bool)
        if pool.any():
            naive_add[np.argsort(-np.where(pool, f["qmv"], -1))[:n_add]] = True
        mm = f["mem"]
        naive_del[np.argsort(np.where(mm, f["qmv"], np.inf))[:n_del]] = True
        keep = pred_add | pred_del | naive_add | naive_del | np.isin(np.arange(nl), list(act_add | act_del))
        for c in np.where(keep)[0]:
            rows.append(dict(review=q.review, year=q.year, horizon=kind, date=d, col=c, lid=L.lid[c], tic=L.tic[c],
                             conm=L.conm[c], member=bool(f["mem"][c]), pred_add=bool(pred_add[c]), pred_del=bool(pred_del[c]),
                             p_add=float(p_add[c]), p_del=float(p_del[c]), naive_add=bool(naive_add[c]), naive_del=bool(naive_del[c]),
                             act_add=c in act_add, act_del=c in act_del, w=float(f["w"][c]), ff=float(f["ff"][c]),
                             ff_src=f["src"][c], turn=float(f["turn"][c]), turn_ca=float(f["turn_ca"][c]), dual=bool(f["dual"][c]),
                             seasoned=bool(f["seas"][c]), banned=bool(f["ban"][c]), elig=bool(f["elig"][c]),
                             vwap=float(f["vw"][c]), vwap3m=float(f["vw3"][c]), mcap=float(f["mcap"][c]),
                             qmv=float(f["qmv"][c]), sigma=float(f["sig"][c]), live=bool(f["live"][c])))
        diag.append(dict(review=q.review, horizon=kind, date=d, M_bn=f["M"] / 1e9, wmax=f["wmax"], capped=f["capped"],
                         n_members=int(f["mem"].sum()), n_pred_add=int(pred_add.sum()), n_pred_del=int(pred_del.sum()),
                         n_act_add=len(act_add), n_act_del=len(act_del), size_entry=rl["size_entry"]))
    print(q.review, {k: v for k, v in diag[-1].items() if k.startswith("n_")}, "M=%.0fbn" % diag[-1]["M_bn"])

suffix = "" if FLOAT_MODE == "est" else "_" + FLOAT_MODE
pr = pd.DataFrame(rows)
pr.to_csv(OUT / f"predictions{suffix}.csv", index=False)
pd.DataFrame(diag).to_csv(OUT / f"prediction_diagnostics{suffix}.csv", index=False)
