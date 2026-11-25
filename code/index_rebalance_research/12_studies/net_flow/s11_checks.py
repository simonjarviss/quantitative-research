"""Step 11: checks. Writes table_checks.csv (check, scope, value, detail) and table_known_events.csv.

1  Holdings-implied adds/deletes vs the sister studies' actual change lists (recall, precision):
   CA by pair mode; US Russell (IWM) by year; S&P 500 (IVV quarter pairs vs Compustat spells); EU STOXX (s12 vs
   sister review events, mapped lines).
2  Known events: XIC/XIU trades at the effective date for named S&P/TSX changes; S&P 500 adds vs the audit's
   float x pi predictions (raw/audits/passive_share_estimates.md §5.2).
3  Tracker AUM: Datastream vs the holdings files (XIC, XIU) and vs CRSP TNA (IVV, IWM); stale share counts.
4  pi_visible at 2026-06-23 vs flow_sizing.md §4.4.
5  Placebo (CA, no-event day pairs) vs the review pairs: |delta| tail and spurious adds/deletes.
6  EU cross-ruler: delta from iShares vs Xtrackers STOXX 600 holdings on the same review.
7  Headline numbers recomputed with independent code (explicit date dummies, own cluster SEs) from the event-level
   files: the Canadian net R and three flow-test coefficients.
8  US month-end timing: share of the year's adds/deletes already in the end-June report.
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, TSX, RUS, EU, CD

rows = []


def add(check, scope, value, detail=""):
    rows.append(dict(check=check, scope=scope, value=value, detail=detail))


# ---------------------------------------------------------------- 1. adds/deletes vs sister lists
D = pd.read_csv(OUT / "ca_demand.csv")
ev = pd.read_csv(TSX / "events.csv")
ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
ev = ev[ev.etype == "review"]
for mode in ["daily", "daily (T+1 booking)", "month-end"]:
    rv = set(D[D.mode_TSX == mode].review)
    for side in ("add", "delete"):
        s = set(zip(ev[ev.review.isin(rv) & (ev.side == side)].review, ev[ev.review.isin(rv) & (ev.side == side)].lid))
        h = set(zip(D[(D.mode_TSX == mode) & (D.cls_TSX == side)].review, D[(D.mode_TSX == mode) & (D.cls_TSX == side)].lid))
        add("CA holdings vs sister review changes", f"{mode} {side}s", f"recall {len(s & h)}/{len(s)}, precision {len(s & h)}/{len(h)}",
            "precision counts off-cycle changes inside month-end pairs as misses" if mode == "month-end" else "")
U = pd.read_csv(OUT / "us_demand.csv")
r = pd.read_csv(RUS / "output" / "events_actual.csv")
r = r[r.valid & r.permno.notna()]
for grp, cls, fam in (("BUY_add", "add", "R2000"), ("SELL_del", "delete", "R2000"), ("BUY_mig", "add", "R2000"), ("SELL_mig", "delete", "R2000")):
    s = set(zip(r[(r.group == grp) & (r.year >= 2011)].year, r[(r.group == grp) & (r.year >= 2011)].permno.astype(int)))
    h = set(zip(U[U[f"cls_{fam}"] == cls].year, U[U[f"cls_{fam}"] == cls].permno))
    add("US IWM holdings vs sister Russell changes 2011-2025", grp, f"recall {len(s & h)}/{len(s)} = {len(s & h) / max(len(s), 1):.1%}")
sp = pd.read_parquet(DATA / "sp500_spells.parquet")
A = pd.read_csv(OUT / "us_sp500_adds.csv")
add("S&P 500 adds (Compustat spells, current members) found in IVV after the effective date", "2010-07..2025-12",
    f"{len(A)} of {int(((pd.to_datetime(sp.dfrom) >= '2010-07-01') & (pd.to_datetime(sp.dfrom) <= '2025-12-31')).sum())}",
    "spells are current members only (survivor-biased)")
E = pd.read_csv(OUT / "eu_demand.csv")
es = pd.read_csv(EU / "events_stoxx.csv")
es = es[es.etype == "review"]
for side in ("add", "delete"):
    sub = E[(E.family == "STOXX") & (E.col >= 0)]
    rvs = set(sub.review)
    s = set(zip(es[(es.side == side) & es.review.isin(rvs)].review, es[(es.side == side) & es.review.isin(rvs)].rlid))
    h = set(zip(sub[sub.cls == side].review, sub[sub.cls == side].lid))
    add("EU iShares STOXX 600 (s12) vs sister STOXX review changes", f"{side}s, reviews with a quarter pair",
        f"recall {len(s & h)}/{len(s)}, precision {len(s & h)}/{len(h)}", "quarter pairs also hold off-cycle changes")

# ---------------------------------------------------------------- 2. known events
K = []
for rv, tics in (("2026-06", ["ALS.", "EXE.", "SDE.", "SXGC.", "HPS.A", "GSY.", "PET.", "TCL.A", "WPK.", "VITL.UN"]),
                 ("2019-03", ["SHOP."]), ("2025-12", ["FFH.", "CLS.", "CAR.UN", "AQN."])):
    x = D[(D.review == rv) & D.tic.isin(tics)]
    for v in x.itertuples():
        K.append(dict(market="CA", review=rv, name=v.name, event=v.event, mode=v.mode_TSX, xic_before=v.h_b_TSX,
                      xic_after=v.h_a_TSX, xiu_before=v.h_b_TSX60, xiu_after=v.h_a_TSX60, D_vis_sh=v.D_net_sh,
                      D_vis_over_adv=v.D_net_sh / v.adv60 if v.adv60 else np.nan,
                      excess_over_D=(v.vol_T - v.med20) / abs(v.D_net_sh) if v.D_net_sh else np.nan))
aud = {"KKR": 21.92e6, "PLTR": 68.24e6, "DASH": 12.42e6, "HOOD": 28.91e6}
for tic, pred in aud.items():
    x = A[A.tic == tic]
    if len(x):
        v = x.iloc[0]
        K.append(dict(market="US", review=str(v["T"])[:10], name=tic, event="S&P 500 add", mode="IVV month-end after T",
                      D_vis_sh=v.D_vis, audit_float_x_pi=pred, ratio_to_audit=v.D_vis / pred,
                      excess_over_D=(v.vol_T - v.med20) / v.D_vis))
KE = pd.DataFrame(K)
KE.to_csv(OUT / "table_known_events.csv", index=False)
for v in KE[KE.market == "US"].itertuples():
    add("S&P 500 add: holdings-implied visible demand / audit float x pi", v.name, round(v.ratio_to_audit, 3))
ca_adds = KE[(KE.market == "CA") & (KE.event == "add")]
add("CA Jun 2026 adds: XIC holds 0 before and > 0 after the T close", "ALS, EXE, SDE, SXGC, HPS.A",
    f"{int(((ca_adds.xic_before == 0) & (ca_adds.xic_after > 0)).sum())} of {len(ca_adds)}")

# ---------------------------------------------------------------- 3. AUM checks
P = pd.read_parquet(DATA / "ca_pairs.parquet")
a = pd.read_parquet(CD / "data" / "etf_aum.parquet")
a["marketdate"] = pd.to_datetime(a.marketdate)
b = pd.read_parquet(DATA / "ds_etf_aum_extra.parquet")
b["marketdate"] = pd.to_datetime(b.marketdate)
for etf, ic, src in (("XIC", 150287, a), ("XIU", 150280, b)):
    s = src[src.infocode == ic].set_index("marketdate").mktcap.sort_index()
    x = P[P.etf == etf].copy()
    x["ds"] = [s[s.index <= d].iloc[-1] if (s.index <= d).any() else np.nan for d in x.b]
    x["ratio"] = x.ds / x.E_b
    x["y"] = pd.to_datetime(x.b).dt.year
    by = x.groupby(x.y // 5 * 5).ratio.median()
    add(f"Datastream AUM / {etf} holdings-file equity value at b", "median by 5-year bucket",
        "; ".join(f"{int(k)}: {v:.3f}" for k, v in by.items()), f"overall median {x.ratio.median():.3f}, min {x.ratio.min():.3f}, max {x.ratio.max():.3f}")
pi = pd.read_parquet(DATA / "pi_daily.parquet")
pi["y"] = pi.date.dt.year
st = pi.groupby(["family", "y"]).stale_share.median().unstack(0)
bad = {f: [int(y) for y in st.index if st.loc[y, f] > 0.5] for f in st.columns}
add("Datastream tracker share counts unchanged >= 120 sessions (stale) on > 50% of AUM", "years by family",
    "; ".join(f"{f}: {v[0]}-{v[-1]}" for f, v in bad.items() if v), "pi_visible in those years understates")
tna = pd.read_parquet(DATA / "us_index_tna.parquet")
fh = pd.read_parquet(DATA / "us_index_funds.parquet")
tna = tna.merge(fh[["fundno", "ticker"]], on="fundno")
tna["caldt"] = pd.to_datetime(tna.caldt)
for tk, ic in (("IVV", 73929), ("IWM", 73898)):
    t = tna[tna.ticker == tk].set_index("caldt").mtna * 1e6
    s = b[b.infocode == ic].set_index("marketdate").mktcap.sort_index()
    r_ = [s[s.index <= d].iloc[-1] / v for d, v in t.items() if (s.index <= d).any() and v > 0 and d >= pd.Timestamp("2008-01-01")]
    add(f"Datastream AUM / CRSP MF TNA, {tk}", "month-ends 2008-2026", f"median {np.median(r_):.3f}, p5 {np.quantile(r_, .05):.3f}, p95 {np.quantile(r_, .95):.3f}")

# ---------------------------------------------------------------- 4. pi vs flow_sizing
ref = {"SP500": 0.0411, "TSX60": 0.0072, "TSX": 0.0100, "STOXX": 0.0027}
for f, v in ref.items():
    x = pi[(pi.family == f) & (pi.date <= "2026-06-23")].sort_values("date").iloc[-1]
    add("pi_visible at 2026-06-23 vs flow_sizing.md §4.4", f, f"{x.pi_visible:.4f} vs {v:.4f}")
pm = pd.read_parquet(DATA / "pi_us_monthly.parquet")
x = pm[(pm.family == "R2000")].dropna(subset=["pi_etf"]).sort_values("date").iloc[-1]
add("pi_visible R2000 (IWM+VTWO, CRSP, IWM-implied MV) latest vs flow_sizing 2.87% (Jun 2026)", str(x.date.date()), f"{x.pi_etf:.4f}")
x = pm[(pm.family == "SP500") & (pm.date == "2025-12-31")]
if len(x):
    add("S&P 500 MV implied by IVV holdings / Datastream MV 41620", "2025-12-31", f"{float(x.mv.iloc[0] / x.mv_official.iloc[0]):.3f}")

# ---------------------------------------------------------------- 5. placebo
PL = pd.read_parquet(DATA / "ca_placebo.parquet")
thr = float(D.THR.iloc[0])
cont = PL[(PL.hb_s > 0) & (PL.ha_s > 0)]
dd = D[D.mode_TSX.str.startswith("daily")]
for etf, fam in (("XIC", "TSX"), ("XIU", "TSX60")):
    c = cont[cont.etf == etf]
    e = dd[dd[f"h_b_{fam}"] > 0][f"delta_{fam}"].dropna()
    add(f"CA placebo vs review pair, {etf}", "|delta| > THR share; p99 |delta|",
        f"placebo {np.mean(np.abs(c.delta) > thr):.2%}, p99 {np.nanquantile(np.abs(c.delta), .99):.4f}; review {np.mean(np.abs(e) > thr):.2%}, p99 {np.nanquantile(np.abs(e), .99):.4f}",
        f"THR = {thr:.3%}")
add("CA placebo pairs: spurious adds + deletes per pair (XIC)", "daily reviews",
    f"{P[P.etf == 'XIC'][['placebo_n_add', 'placebo_n_del']].sum(1).mean():.2f}", "off-cycle changes on T-1 count here")

# ---------------------------------------------------------------- 6. EU cross-ruler
try:
    C = pd.read_parquet(DATA / "eu_ruler_check.parquet")
    c = C[C.cls == "cont"].dropna(subset=["delta", "delta_chk"])
    big = c[(c.delta.abs() > 0.01) | (c.delta_chk.abs() > 0.01)]
    add("EU STOXX 600: delta from iShares (45947) vs Xtrackers (138577) holdings, same review", "continuing names",
        f"corr {np.corrcoef(c.delta.clip(-1, 1), c.delta_chk.clip(-1, 1))[0, 1]:.3f} (n {len(c)}); |delta|>1%: corr {np.corrcoef(big.delta.clip(-1, 1), big.delta_chk.clip(-1, 1))[0, 1]:.3f} (n {len(big)})",
        f"{C.review.nunique()} reviews")
except FileNotFoundError:
    add("EU cross-ruler", "", "not available")

# ---------------------------------------------------------------- 7. independent recomputation
def ols_dummies(y, X, g):
    """OLS with explicit group dummies; CR1 cluster SE by g. Independent of common.fe_ols (no demeaning)."""
    G = pd.get_dummies(pd.Series(g).astype(str)).values.astype(float)
    Z = np.column_stack([X, G])
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    ZtZi = np.linalg.pinv(Z.T @ Z)
    k = Z.shape[1]; n = len(y)
    meat = np.zeros((k, k))
    gs = pd.Series(g).astype(str).values
    for u in np.unique(gs):
        m = gs == u
        sv = Z[m].T @ e[m]
        meat += np.outer(sv, sv)
    nG = len(np.unique(gs))
    V = nG / (nG - 1) * (n - 1) / (n - k) * ZtZi @ meat @ ZtZi
    return b[:X.shape[1]], np.sqrt(np.diag(V))[:X.shape[1]]


TR = pd.read_csv(OUT / "table_R.csv")
dca = D[D.mode_TSX.str.startswith("daily") & D.event.isin(["add", "delete", "tsx60_add", "tsx60_delete", "weight", "unchanged"])].copy()
dca = dca[(dca.adv60 > 0) & dca.vol_T.notna() & (dca.shares_out > 0) & (dca.price_Tm1 > 0)]
yv = (dca.vol_T - dca.adv60) / dca.adv60
yv = yv.clip(*yv.quantile([.01, .99])).values
xv = (dca.D_net_sh.abs() / dca.adv60)
xv = xv.clip(0, xv.quantile(.99)).values
ctr = np.column_stack([xv, np.log(dca.price_Tm1 * dca.shares_out), np.log(dca.adv60 / dca.shares_out),
                       (dca.price_Tm1 < 5).astype(float), ((dca.price_Tm1 >= 5) & (dca.price_Tm1 < 20)).astype(float)])
bb, se = ols_dummies(yv, ctr, dca["T"].values)
ref = TR[(TR["sample"] == "CA daily, all lines, net demand") & (TR.units == "adv")].iloc[0]
add("Recompute: Canadian net R (all lines, daily), dummies vs within estimator", "R (se)",
    f"{bb[0]:.3f} ({se[0]:.3f}) vs table {ref.R:.3f} ({ref.se:.3f})")
F = pd.read_parquet(DATA / "flow_sample.parquet")
FT = pd.read_csv(OUT / "table_flow_test.csv")
for mkt, grp, lab in (("CA", "weight", "CA S&P/TSX"), ("US", "add/delete", "US Russell"), ("EU", "weight", "EU STOXX 600 + DAX family (+FTSE)")):
    d = F[(F.market == mkt) & (F.grp == grp)]
    d = d[np.isfinite(d.lflow) & np.isfinite(d.lturn) & np.isfinite(d.lmcap_) & np.isfinite(d.y_d0)]
    def z(v):
        v = pd.Series(v).clip(*pd.Series(v).quantile([.01, .99]))
        return ((v - v.mean()) / v.std()).values
    y = pd.Series(d.y_d0.values).clip(*d.y_d0.quantile([.01, .99])).values
    X = np.column_stack([z(d.lflow.values), z(d.lturn.values), z(d.lmcap_.values), d.pb_lo.values, d.pb_mid.values,
                         (d.side_ < 0).astype(float).values])
    bb, se = ols_dummies(y, X, d["T"].astype(str).values)
    ref = FT[(FT.market == lab) & (FT.group == grp) & (FT.side == "all") & (FT.period == "all") & (FT.leg == "d0")
             & (FT.spec == "flow + turnover + size + price, date FE")].iloc[0]
    add("Recompute: flow coefficient, trade day (bp per SD)", f"{lab} {grp}",
        f"{bb[0]:.2f} (t {bb[0] / se[0]:.2f}) vs table {ref.coef_flow:.2f} (t {ref.t_flow:.2f})")

# ---------------------------------------------------------------- 8. US timing
up = pd.read_parquet(DATA / "us_pairs.parquet")
x = up[up.fund == "IWM"]
add("US IWM: adds / deletes already in the first report on or after T (end-June)", "2011-2025",
    f"adds {x.june_n_add.sum()}/{x.n_add.sum()}, deletes {x.june_n_del.sum()}/{x.n_del.sum()}",
    "2024: 0 of 228 adds at 30 Jun (T = 28 Jun); 2025: 48 of 190 deletes complete at 30 Jun; hence end-July is used")

C = pd.DataFrame(rows)
C.to_csv(OUT / "table_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140); pd.set_option("display.max_rows", 200)
print(C.to_string())
print(KE.round(3).to_string())
