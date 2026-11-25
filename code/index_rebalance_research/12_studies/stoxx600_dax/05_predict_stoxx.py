"""Step 5: point-in-time predictions of each STOXX Europe 600 quarterly review's adds and deletes at H6, H3, C.

At each horizon date d only data dated <= d is used (prices, primary value traded, shares, float estimates,
membership and review results already announced). Rules: stoxx_europe_600.md §2-3 and §10(a):
  universe   one line per company (index-member line for members, else the company's most liquid home-exchange
             line by 3-month primary value traded), common stock, listed in a STOXX Europe country at that review
             (common.stoxx_country_ok), priced in the last 10 sessions, first priced before the end of month M-2
             (TMI-entry proxy for new listings; no seasoning rule in the spec)
  liquidity  3-month ADTV (primary venue, EUR) > EUR 1m  (current rule applied to all years)
  rank       free-float market cap in EUR = close x shares x float estimate x FX (float: Datastream freefloatpct
             when strategic holdings are reported, else Worldscope 1 - closely held %, else 100%)
  select     ranks 1-550; then current members ranked 551-750 in rank order up to 600; then largest remaining
H3 and C: literal screen ("if the review were held today"), membership as known at d.
H6: review q-1 sits between d and review q, so the literal screen predicts q-1. For q we use the random-walk
    model of the TSX study: log(FFMC / entry level) follows a driftless random walk with the name's trailing
    120-session volatility relative to the STOXX proxy; entry level = FFMC at rank 550, exit level = FFMC of the
    lowest selected member at d. Predicted add at q if P(fail at C_{q-1}, pass at C_q) >= 0.10 and >= P(pass at
    C_{q-1}); deletes symmetric. (H6 literal screen kept as column pred_*_lit.)
Variants: 'rules' (above, the headline model) and 'rules+MA' (also drops names with a pending control bid
    announced by d in SDC, matched on SEDOL or ISIN-derived SEDOL; not tuned on outcomes).
Naive benchmark: top-N non-members by full EUR market cap and bottom-N members, N = mean review adds (deletes)
    in the last 4 reviews announced on or before the horizon date.
Outputs: predictions_stoxx.csv, prediction_diagnostics_stoxx.csv
"""
import numpy as np, pandas as pd
from scipy.stats import norm, multivariate_normal
from common import DATA, OUT, eu_calendar, review_dates, stoxx_country_ok

cal = eu_calendar()
rev = review_dates(cal, "stoxx").reset_index(drop=True)
rev["qn"] = np.arange(len(rev))
z = np.load(DATA / "panel.npz")
P, RET, VAL, VP, SH, AJ, FX = (z[k] for k in ["P", "RET", "VAL", "VP", "SH", "AJ", "FX"])
MS = np.load(DATA / "member_stoxx.npy")
L = pd.read_parquet(DATA / "panel_lines.parquet")
nd, nl = P.shape
mkt = pd.read_parquet(DATA / "index_proxy.parquet").stoxx_proxy_tr_local.fillna(0).values
ev = pd.read_csv(OUT / "events_stoxx.csv", parse_dates=["eff_date", "T"], dtype={"gvkey": str})
rq = ev[ev.etype == "review"].merge(rev[["review", "qn"]], on="review")
gv = L.gvkey.values
home = L.home_line.values
common_ = L.tpci.isin(["0", "1"]).values      # common and preferred shares ("equities with similar characteristics")
cty = L.home_cty.values
has_px = ~np.isnan(P)
first_pos = np.where(has_px.any(0), has_px.argmax(0), nd)
MCAP = P * SH / AJ * FX            # EUR full market cap of the line

fm = pd.read_parquet(DATA / "float_monthly.parquet")
fm["month"] = pd.to_datetime(fm.month)
months = pd.DatetimeIndex(sorted(fm.month.unique()))
FDS = np.full((len(months), nl), np.nan); FWS = np.full((len(months), nl), np.nan)
mi = months.get_indexer(fm.month)
FDS[mi, fm.line.values] = fm.ff_ds.values; FWS[mi, fm.line.values] = fm.ff_ws.values

# ---- SDC pending control bids (rules+MA variant)
sdc = pd.read_parquet(DATA / "sdc_ma_all.parquet")
sdc = sdc[(sdc.psought.fillna(sdc.pctacq) >= 50) & sdc.status.isin(["Completed", "Pending", "Unconditional"])].copy()
for c in ["dateann", "dateeff", "datewith"]:
    sdc[c] = pd.to_datetime(sdc[c])
L["sedol_isin"] = np.where(L["isin"].str[:2].isin(["GB", "IE"]), L["isin"].str[4:11], None)
sed_of_gv = pd.concat([L[["gvkey", "sedol"]].rename(columns={"sedol": "s"}),
                       L[["gvkey", "sedol_isin"]].rename(columns={"sedol_isin": "s"})]).dropna().drop_duplicates()
sdc = sdc[sdc.target_sedol.notna()].merge(sed_of_gv, left_on="target_sedol", right_on="s")
sdc_g = dict(tuple(sdc.groupby("gvkey")))
print("SDC control deals linked to the universe by SEDOL:", sdc.master_deal_no.nunique(), "companies:", sdc.gvkey.nunique())


def pending(g, d):
    x = sdc_g.get(g)
    if x is None:
        return False
    live = (x.dateann <= d) & (x.datewith.isna() | (x.datewith > d)) & \
           ((x.dateeff > d) | (x.dateeff.isna() & (x.status == "Pending") & (x.dateann >= d - pd.Timedelta(days=365))))
    return bool(live.any())


def float_at(d):
    i = months.searchsorted(d, side="right") - 1
    ds, ws = FDS[i], FWS[i]
    ff = np.where(~np.isnan(ds), ds, np.where(~np.isnan(ws), ws, 1.0))
    src = np.where(~np.isnan(ds), "DS", np.where(~np.isnan(ws), "WS", "default100"))
    return ff, src


def known_members(di, d):
    m = MS[di].copy()
    for r in rev.itertuples():
        if r.A <= d < r.eff:
            x = rq[(rq.review == r.review) & (rq.col >= 0)]
            m[x.loc[x.side == "add", "col"].values] = True
            m[x.loc[x.side == "delete", "col"].values] = False
    return m


EU_HQ = {"GBR", "DEU", "FRA", "CHE", "SWE", "NLD", "ITA", "ESP", "DNK", "NOR", "FIN", "BEL", "AUT", "IRL", "PRT", "LUX",
         "POL", "GRC", "ISL", "CZE", "JEY", "GGY", "IMN", "BMU"}
hq_eu = L["loc"].isin(EU_HQ).values
operating = L.gsector.notna().values


def screen(q, d, v0=False):
    """Shadow selection at session d for review q. Returns dict of per-line arrays (company-level on chosen line).
    v0=True: first-run universe without the three eligibility proxies below (kept for transparency).
    Proxies for spec eligibility rules that Compustat cannot express directly, applied to non-members only:
      - ICB 30204000/30205000 exclusion (investment trusts, closed-end funds): company has no GICS sector in
        Compustat (all 1,198 STOXX 600 members since 2008 have one)
      - STOXX country assignment ("nationality analysis"): headquarters (g_company.loc) in Europe or a UK-linked
        offshore centre (JEY, GGY, IMN, BMU); drops US/other companies whose primary non-US line is European
      - TMI trading-days rule: at most 10 non-trading sessions (no price or zero volume) in the last 63
      - liquidity: ADTV must be observed (>= 20 sessions with volume) as well as > EUR 1m
    These proxies were chosen after inspecting the v0 run's false positives (RESULTS.md reports both)."""
    di = cal.get_loc(d)
    mem = known_members(di, d)
    adtv = np.nanmean(VAL[max(0, di - 62):di + 1], 0)
    nobs = (~np.isnan(VAL[max(0, di - 62):di + 1])).sum(0)
    live = has_px[max(0, di - 9):di + 1].any(0)
    ff, src = float_at(d)
    mc = np.where(np.isnan(MCAP[di]), pd.DataFrame(MCAP[max(0, di - 9):di + 1]).ffill().values[-1], MCAP[di])
    ffmc = mc * ff
    # one line per company: member line if a member, else most liquid home line
    df = pd.DataFrame({"gv": gv, "col": np.arange(nl), "mem": mem, "adtv": np.nan_to_num(adtv, nan=-1),
                       "ok": live & ~np.isnan(ffmc) & (ffmc > 0) & (home | mem)})
    df = df[df.ok].sort_values(["gv", "mem", "adtv"], ascending=[True, False, False])
    chosen = np.zeros(nl, bool); chosen[df.drop_duplicates("gv").col.values] = True
    C_month = pd.Timestamp(q.C.year, q.C.month, 1)
    listed_ok = cal[np.minimum(first_pos, nd - 1)] < C_month - pd.DateOffset(months=1)
    listed_ok = listed_ok | mem
    ctry = np.array([stoxx_country_ok(c, q.review) for c in cty])
    if v0:
        liq = (adtv > 1e6) | (nobs < 20)
        elig = chosen & ctry & (common_ | mem) & listed_ok & liq
    else:
        w = slice(max(0, di - 62), di + 1)
        nontrade = (np.isnan(P[w]) | (VP[w] == 0)).sum(0)
        entry_ok = operating & hq_eu & (nontrade <= 10) & (nobs >= 20) & (adtv > 1e6)
        liq = (adtv > 1e6) | (nobs < 20)
        elig = chosen & ctry & (common_ | mem) & listed_ok & liq & (mem | entry_ok)
    order = np.argsort(-np.where(elig, ffmc, -np.inf))
    rank = np.full(nl, np.nan)
    ne = int(elig.sum())
    rank[order[:ne]] = np.arange(1, ne + 1)
    sel = elig & (rank <= 550)
    k = 600 - sel.sum()
    buf = np.where(elig & mem & (rank > 550) & (rank <= 750))[0]
    buf = buf[np.argsort(rank[buf])][:max(k, 0)]
    sel[buf] = True
    k = 600 - sel.sum()
    if k > 0:
        rest = np.where(elig & ~sel)[0]
        sel[rest[np.argsort(rank[rest])][:k]] = True
    L550 = ffmc[order[549]] if ne >= 550 else np.nan
    selm = sel & mem
    Lexit = np.nanmin(np.where(selm, ffmc, np.nan))
    return dict(di=di, mem=mem, elig=elig, chosen=chosen, rank=rank, sel=sel, ffmc=ffmc, ff=ff, src=src, mc=mc,
                adtv=adtv, L550=L550, Lexit=Lexit, ctry=ctry, live=live)


def bvn_cdf(a, b, rho):
    return multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(np.column_stack([a, b]))


def rel_sigma(di):
    r = RET[max(0, di - 119):di + 1]; mk = mkt[max(0, di - 119):di + 1][:, None]
    rel = np.log1p(np.clip(r, -0.95, 5)) - np.log1p(mk)
    nobs = (~np.isnan(rel)).sum(0)
    sig = np.where(nobs >= 60, np.nanstd(rel, 0), np.nan)
    return np.clip(np.where(np.isnan(sig), 0.03, sig), 0.005, 0.10)


rows, diag = [], []
for q in rev.itertuples():
    act = rq[rq.review == q.review]
    act_add_g = set(act.loc[act.side == "add", "gvkey"]); act_del_g = set(act.loc[act.side == "delete", "gvkey"])
    for kind in ["H6", "H3", "C"]:
        d = getattr(q, kind)
        prev = rev[(rev.A <= d) & (rev.qn < q.qn)].tail(4)      # reviews already announced at d (was: by C)
        f = screen(q, d)
        mem, sel, elig = f["mem"], f["sel"], f["elig"]
        f0 = screen(q, d, v0=True)
        v0_add = f0["sel"] & ~mem; v0_del = mem & ~f0["sel"] & f0["live"]
        lit_add = sel & ~mem
        lit_del = mem & ~sel & f["live"]
        pred_add, pred_del = lit_add.copy(), lit_del.copy()
        p_add = lit_add.astype(float); p_del = lit_del.astype(float)
        if kind == "H6" and q.qn > 0:
            q1 = rev.iloc[q.qn - 1]
            t2 = cal.get_loc(q.C) - f["di"]; t1 = cal.get_loc(q1.C) - f["di"]
            sig = rel_sigma(f["di"])
            p_add = np.zeros(nl); p_del = np.zeros(nl)
            cand = elig & ~mem & (f["ffmc"] > 0)
            x0 = np.log(np.where(cand, f["ffmc"], 1) / f["L550"])
            a = x0[cand] / (sig[cand] * np.sqrt(t1)); b = x0[cand] / (sig[cand] * np.sqrt(t2))
            pp = norm.cdf(-a) - bvn_cdf(-a, -b, np.sqrt(t1 / t2)); p1 = norm.cdf(a)
            p_add[cand] = pp
            pred_add = np.zeros(nl, bool); pred_add[np.where(cand)[0][(pp >= 0.10) & (pp >= p1)]] = True
            candd = mem & elig
            y0 = np.log(np.where(candd, f["ffmc"], 1) / f["Lexit"])
            a = y0[candd] / (sig[candd] * np.sqrt(t1)); b = y0[candd] / (sig[candd] * np.sqrt(t2))
            pp = norm.cdf(-b) - bvn_cdf(-a, -b, np.sqrt(t1 / t2)); p1 = norm.cdf(-a)
            p_del[candd] = pp
            pred_del = np.zeros(nl, bool); pred_del[np.where(candd)[0][(pp >= 0.10) & (pp >= p1)]] = True
        # naive benchmark
        n_add = max(1, int(round((rq.review.isin(prev.review) & (rq.side == "add")).sum() / max(len(prev), 1))))
        n_del = max(1, int(round((rq.review.isin(prev.review) & (rq.side == "delete")).sum() / max(len(prev), 1))))
        pool = f["chosen"] & f["ctry"] & ~mem & f["live"] & (common_)
        naive_add = np.zeros(nl, bool); naive_del = np.zeros(nl, bool)
        naive_add[np.argsort(-np.where(pool, f["mc"], -np.inf))[:n_add]] = True
        naive_del[np.argsort(np.where(mem & f["live"], f["mc"], np.inf))[:n_del]] = True
        keep_cols = set(np.where(pred_add | pred_del | lit_add | lit_del | naive_add | naive_del | v0_add | v0_del)[0])
        # actual events: by company; record on the event's ret line
        for e in act.itertuples():
            if e.col >= 0:
                keep_cols.add(e.col)
        for c in sorted(keep_cols):
            g = gv[c]
            rows.append(dict(review=q.review, year=q.year, horizon=kind, date=d, col=c, lid=L.lid[c], gvkey=g,
                             conm=L.conm[c], cty=cty[c], member=bool(mem[c]),
                             pred_add=bool(pred_add[c]), pred_del=bool(pred_del[c]),
                             pred_add_lit=bool(lit_add[c]), pred_del_lit=bool(lit_del[c]),
                             p_add=float(p_add[c]), p_del=float(p_del[c]),
                             naive_add=bool(naive_add[c]), naive_del=bool(naive_del[c]),
                             pred_add_v0=bool(v0_add[c]), pred_del_v0=bool(v0_del[c]),
                             act_add=g in act_add_g, act_del=g in act_del_g, rank=f["rank"][c], ffmc=f["ffmc"][c],
                             ff=f["ff"][c], ff_src=f["src"][c], mcap=f["mc"][c], adtv=f["adtv"][c], elig=bool(elig[c]),
                             selected=bool(sel[c]), ma_pending=pending(g, d)))
        diag.append(dict(review=q.review, horizon=kind, date=d, n_known_members=int(mem.sum()), n_elig=int(elig.sum()),
                         L550_bn=f["L550"] / 1e9, Lexit_bn=f["Lexit"] / 1e9,
                         n_pred_add=int(pred_add.sum()), n_pred_del=int(pred_del.sum()),
                         n_lit_add=int(lit_add.sum()), n_lit_del=int(lit_del.sum()),
                         n_act_add=len(act_add_g), n_act_del=len(act_del_g)))
    print(q.review, {k: v for k, v in diag[-1].items() if k.startswith("n_")}, flush=True)

pr = pd.DataFrame(rows)
# a company can appear on two lines (e.g. ret line of the actual event and its ranking line): collapse to company
pr = pr.sort_values(["review", "horizon", "gvkey", "member"], ascending=[True, True, True, False])
agg = {c: "max" for c in ["pred_add", "pred_del", "pred_add_lit", "pred_del_lit", "naive_add", "naive_del", "act_add",
                          "pred_add_v0", "pred_del_v0",
                          "act_del", "ma_pending", "member"]}
first = pr.groupby(["review", "horizon", "gvkey"]).first().reset_index()
flags = pr.groupby(["review", "horizon", "gvkey"]).agg(agg).reset_index()
pr = first.drop(columns=list(agg)).merge(flags, on=["review", "horizon", "gvkey"])
# return line: the actual event's ret line when the company changed, else the ranking line
evcol = rq.drop_duplicates(["review", "gvkey"]).set_index(["review", "gvkey"]).col
pr["col"] = [int(evcol.get((r, g), c)) for r, g, c in zip(pr.review, pr.gvkey, pr.col)]
pr["pred_add_ma"] = pr.pred_add & ~pr.ma_pending
pr["pred_del_ma"] = pr.pred_del & ~pr.ma_pending
# 'memory' variant (point-in-time, as the TSX study's committee memory; threshold fixed a priori, not tuned):
#   drop a predicted add that our cut-off screen ranked <= 500 at one of the last two reviews already announced at d
#   without STOXX adding it (a structural exclusion we cannot see: AIM/growth segment, US primary listing, fund);
#   drop a predicted delete that our cut-off screen called a delete at one of the last two announced reviews
#   without STOXX deleting it.
pC = pr[pr.horizon == "C"]
rej = pC[pC.pred_add & ~pC.act_add & (pC["rank"] <= 500)].groupby("gvkey").review.apply(set).to_dict()
prot = pC[pC.pred_del & ~pC.act_del].groupby("gvkey").review.apply(set).to_dict()
A_of = dict(zip(rev.review, rev.A))


def recent(sets, g, d):
    s_ = sets.get(g)
    if not s_:
        return False
    known = sorted(r for r, a in A_of.items() if a <= d)[-2:]
    return any(r in s_ for r in known)


pr["mem_reject"] = [recent(rej, g, d) for g, d in zip(pr.gvkey, pd.to_datetime(pr.date))]
pr["mem_protect"] = [recent(prot, g, d) for g, d in zip(pr.gvkey, pd.to_datetime(pr.date))]
pr["pred_add_mem"] = pr.pred_add & ~pr.mem_reject
pr["pred_del_mem"] = pr.pred_del & ~pr.mem_protect
pr.to_csv(OUT / "predictions_stoxx.csv", index=False)
pd.DataFrame(diag).to_csv(OUT / "prediction_diagnostics_stoxx.csv", index=False)


def acc(df, pc, ac):
    tp = (df[pc] & df[ac]).sum(); fp = (df[pc] & ~df[ac]).sum(); fn = (~df[pc] & df[ac]).sum()
    return int(tp + fp), int(tp + fn), int(tp), round(tp / max(tp + fp, 1), 3), round(tp / max(tp + fn, 1), 3)


for h in ["H6", "H3", "C"]:
    x = pr[pr.horizon == h]
    for v in ["", "_ma", "_mem", "_lit", "_v0"]:
        print(h, v or "rules", "ADD npred/nact/tp/P/R", acc(x, "pred_add" + v, "act_add"), "| DEL", acc(x, "pred_del" + v, "act_del"))
    print(h, "naive", acc(x, "naive_add", "act_add"), acc(x, "naive_del", "act_del"))
