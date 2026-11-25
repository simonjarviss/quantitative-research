"""Step 4 (Canada): holdings-implied passive demand per stock at every S&P/TSX quarterly review, 2006-12 .. 2026-09.

Inputs: data/ishca/{XIC,XIU}_YYYYMMDD.csv (s01), data/pi_daily.parquet (s03), the S&P/TSX sister panel
(../tsx_composite/data/panel.npz: P = Compustat close, AJ = ajexdi, VC/VT = consolidated / TSX volume and SH = shares
outstanding, the last three in split-adjusted units = x ajexdi), its member matrix and float_monthly.

Method, per review and ETF (XIC = S&P/TSX Capped Composite ruler, XIU = S&P/TSX 60 ruler):
  pair (b, a): daily mode b = T-1, a = T (the fund books the review trades at the T close; checked on Jun 2026).
               month-end mode (before daily files exist, about 2006-12 .. 2011): b = last session of the month before
               the review month, a = last session of the review month.
  h = shares held (summed per line), put on a common split basis: h* = h x ajexdi.
  c' = sum_i h*_a P_i / sum_i h*_b P_i at one price vector (the a-date close): the fund's creation/redemption factor.
       The review trades are self-financing at fixed prices, so c' removes fund flows but keeps the index trades.
  fund trade_i = h*_a / c' - h*_b   (shares, on the b-date fund size). For an unchanged member this is the small
       uniform "funding" trade h*_b (c/c' - 1), with c the median h*_a/h*_b of continuing members.
  delta_i = (h*_a / h*_b) / c - 1 = the change in the line's index float shares (share, IWF or capping update).
  visible demand D_vis_i = trade_i x AUM_visible(family, b) / E_fund(b), E_fund = the fund's equity value on b and
       AUM_visible = Datastream AUM of all visible trackers (s03; TSX: XIC+ZCN+HXCN, TSX60: XIU+ZIU+HXT).
  Net Canadian visible demand per line D_net = D_TSX + D_TSX60 (both families trade the same close).
Classes per family: add (h_b = 0), delete (h_a = 0), weight (|delta| > THR), unchanged. THR is set from the placebo
pair (T-2 -> T-1) in daily mode: the 99.5th percentile of |delta| on a no-event day (printed).
Weight changes are split into "share update" (delta within 0.25|delta| + 0.2 pt of the quarter's change in Compustat
shares outstanding since the previous review) and "IWF / other"; "capping" if the line's XIC weight before is >= 9.5%.
Month-end mode adds/deletes not in the sister study's review event list are labelled off-cycle.

Outputs: data/ca_holdings.parquet (parsed and mapped holdings), data/ca_pairs.parquet (pair-level diagnostics),
ca_demand.csv (one row per review x line: holdings, classes, delta, D_vis per family, D_net, ADV60, V_T, ...).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from common import DATA, OUT, TSX, tsx_reviews, tsx_sessions

ISH = DATA / "ishca"
PZ = np.load(TSX / "data" / "panel.npz")
P, AJ, VC, VT, SH = PZ["P"], PZ["AJ"], PZ["VC"], PZ["VT"], PZ["SH"]
AJf = pd.DataFrame(AJ).ffill().bfill().values
PD = pd.DatetimeIndex(PZ["dates"])
L = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
MEM = np.load(TSX / "data" / "member_matrix.npy")
nl = len(L)


def norm_tic(t):
    t = str(t).upper().strip().rstrip(".")
    t = re.sub(r"\.(Z|\d)$", "", t)
    return t.replace("-", ".")


L["ntic"] = L.tic.map(norm_tic)
TIC2COL = L.groupby("ntic").apply(lambda g: list(g.index)).to_dict()
NAME_TOK = [set(re.findall(r"[A-Z]{3,}", str(n).upper())) - {"INC", "LTD", "CORP", "CORPORATION", "THE", "TRUST", "FUND",
                                                               "CLASS", "COMPANY", "LIMITED", "HOLDINGS", "GROUP", "UNITS"}
            for n in L.conm]


def parse(p: Path):
    first = open(p, encoding="utf-8-sig").readline()
    asof = pd.Timestamp(re.search(r'"(.+)"', first).group(1))
    x = pd.read_csv(p, skiprows=2, thousands=",", keep_default_na=False, na_values=["", "-"], dtype={"Ticker": str})
    x = x[(x["Asset Class"] == "Equity")]
    x = x.assign(shares=pd.to_numeric(x["Shares"], errors="coerce"), price=pd.to_numeric(x["Price"], errors="coerce"),
                 mv=pd.to_numeric(x["Market Value"], errors="coerce"), wt=pd.to_numeric(x["Weight (%)"], errors="coerce"))
    x = x[(x.shares > 0) & (x.price > 0)]
    return asof, x[["Ticker", "Name", "shares", "price", "mv", "wt"]].rename(columns={"Ticker": "ticker", "Name": "name"})


def match(d, x):
    """Map holdings rows of date d to panel columns. Returns col array and quality labels."""
    if d not in PD:
        return np.full(len(x), -1), np.array(["no_panel_date"] * len(x), dtype=object)
    i = PD.get_loc(d)
    Pd = P[i]
    mem = MEM[max(0, i - 5):i + 6].any(0)
    cols, qs = [], []
    for t, n, p in zip(x.ticker, x.name, x.price):
        c, q = -1, "none"
        cand = TIC2COL.get(norm_tic(t), [])
        ok = [k for k in cand if np.isfinite(Pd[k]) and abs(Pd[k] / p - 1) < 0.02]
        if ok:
            c, q = ok[0], "tic"
        else:
            rel = np.abs(Pd / p - 1)
            pc = np.where(np.isfinite(rel) & (rel < 0.003))[0]
            toks = set(re.findall(r"[A-Z]{3,}", str(n).upper()))
            pm = [k for k in pc if mem[k]]
            if len(pm) == 1 and (len(toks & NAME_TOK[pm[0]]) > 0 or rel[pm[0]] < 0.0005):
                c, q = pm[0], "price_member"
            elif len(pc):
                sc = [len(toks & NAME_TOK[k]) for k in pc]
                if max(sc) > 0:
                    c, q = pc[int(np.argmax(sc))], "price_name"
            if c < 0:
                pc3 = np.where(np.isfinite(rel) & (rel < 0.03))[0]
                if len(pc3):
                    sc = [len(toks & NAME_TOK[k]) for k in pc3]
                    if max(sc) >= 2 or (max(sc) == 1 and len(toks) == 1):
                        c, q = pc3[int(np.argmax(sc))], "name_px3"
        cols.append(c); qs.append(q)
    return np.array(cols), np.array(qs, dtype=object)


def load_holdings():
    p = DATA / "ca_holdings.parquet"
    if p.exists():
        return pd.read_parquet(p)
    rows = []
    for f in sorted(ISH.glob("*.csv")):
        etf, ds = f.stem.split("_")
        asof, x = parse(f)
        d = pd.Timestamp(ds)
        assert asof.normalize() == d, (f, asof)
        col, q = match(d, x)
        x = x.assign(etf=etf, date=d, col=col, q=q)
        rows.append(x)
    H = pd.concat(rows, ignore_index=True)
    # consistency: a (ticker, name) pair that is matched on most dates fills its unmatched dates (same line)
    key = H.ticker.fillna("") + "|" + H.name.fillna("")
    mode = H[H.col >= 0].groupby(key[H.col >= 0]).col.agg(lambda s: s.value_counts().index[0])
    fill = (H.col < 0) & key.isin(mode.index)
    H.loc[fill, "col"] = key[fill].map(mode).values
    H.loc[fill, "q"] = "filled_same_ticker_name"
    H.to_parquet(p, index=False)
    return H


def aum_visible(fam):
    pi = pd.read_parquet(DATA / "pi_daily.parquet")
    pi = pi[pi.family == fam].set_index("date").sort_index()
    return pi


def pair_trades(H, etf, b, a):
    """Fund trades between holdings dates b and a (per panel column). Returns DataFrame and pair stats."""
    hb = H[(H.etf == etf) & (H.date == b)]
    ha = H[(H.etf == etf) & (H.date == a)]
    if hb.empty or ha.empty:
        return None, None
    um_b = hb.col.lt(0).mean(); um_a = ha.col.lt(0).mean()
    Eb = hb.mv.sum()
    gb = hb[hb.col >= 0].groupby("col").agg(h_b=("shares", "sum"), p_b=("price", "first"), w_b=("wt", "sum"))
    ga = ha[ha.col >= 0].groupby("col").agg(h_a=("shares", "sum"), p_a=("price", "first"), w_a=("wt", "sum"))
    g = gb.join(ga, how="outer").fillna({"h_b": 0.0, "h_a": 0.0, "w_b": 0.0, "w_a": 0.0})
    ib = PD.get_loc(b) if b in PD else None
    ia = PD.get_loc(a) if a in PD else None
    cols = g.index.values.astype(int)
    ajb = AJf[ib, cols] if ib is not None else np.ones(len(g))
    aja = AJf[ia, cols] if ia is not None else np.ones(len(g))
    g["hb_s"] = g.h_b * ajb                     # split-adjusted units (panel volume units)
    g["ha_s"] = g.h_a * aja
    # one price vector: a-date close per adjusted share; deletes use their a-date panel close when they still trade
    pa = g.p_a.copy()
    if ia is not None:
        pa = pa.fillna(pd.Series(P[ia, cols], index=g.index))
    pa = pa.fillna(g.p_b * (aja / ajb))         # last resort: b price on the a basis
    pas = pa / aja
    cprime = (g.ha_s * pas).sum() / (g.hb_s * pas).sum()
    cont = (g.hb_s > 0) & (g.ha_s > 0)
    r = g.ha_s / g.hb_s
    c = np.median(r[cont])
    g["delta"] = np.where(cont, r / c - 1, np.nan)
    g["trade_s"] = g.ha_s / cprime - g.hb_s
    g["funding_s"] = np.where(cont, g.hb_s * (c / cprime - 1), 0.0)
    g["pa"] = pa
    st = dict(etf=etf, b=b, a=a, n_b=len(hb), n_a=len(ha), unmatched_b=um_b, unmatched_a=um_a, E_b=Eb,
              E_a=ha.mv.sum(), cprime=cprime, c_med=c, m_index=cprime / c,
              n_add=int(((g.hb_s == 0) & (g.ha_s > 0)).sum()), n_del=int(((g.hb_s > 0) & (g.ha_s == 0)).sum()))
    return g, st


def main():
    H = load_holdings()
    q = H.groupby(["etf", H.date.dt.year]).q.value_counts(normalize=True).unstack().fillna(0).round(3)
    print("match quality by ETF x year\n", q.to_string())
    cal = tsx_sessions()
    rv = tsx_reviews()
    dates = set(H.date.unique())
    ev = pd.read_csv(TSX / "events.csv", parse_dates=["eff_date", "T"])
    ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
    ev_rev = ev[ev.etype.isin(["review", "offcycle_at_review"])]
    revset = set(zip(ev_rev.review, ev_rev.lid, ev_rev.side))
    lid_ix = pd.Index(L.gvkey + "_" + L.iid)
    av = {"XIC": aum_visible("TSX"), "XIU": aum_visible("TSX60")}
    fam = {"XIC": "TSX", "XIU": "TSX60"}
    pairs, rows, placebo = [], [], []
    PD_end = PD[-1]
    for r in rv.itertuples():
        if r.T > PD[-1]:
            continue                                   # beyond the R2 panel (Sep 2026): no prices to map or test
        i = cal.get_loc(r.T)
        per = {}
        Tm1, T, Tm2, Tp1 = cal[i - 1], cal[i], cal[i - 2], cal[i + 1]
        # booking lag: in 2012-06 .. 2014-09 the files dated T do not yet show the T-close trades (they appear in the
        # T+1 file). Decided on XIC, applied to both ETFs: lagged if T -> T+1 shows more adds/deletes, or a much wider
        # |delta| tail, than T-1 -> T.
        lag = False
        hx = set(H[H.etf == "XIC"].date)
        if {Tm1, T, Tp1} <= hx:
            g0, s0 = pair_trades(H, "XIC", Tm1, T)
            g1, s1 = pair_trades(H, "XIC", T, Tp1)
            q0 = np.nanquantile(np.abs(g0.delta), 0.99); q1 = np.nanquantile(np.abs(g1.delta), 0.99)
            lag = (s1["n_add"] + s1["n_del"] > s0["n_add"] + s0["n_del"]) or (q1 > 5 * q0 and q1 > 0.02)
        for etf in ("XIC", "XIU"):
            have = set(H[H.etf == etf].date)
            if lag and {Tm1, T, Tp1} <= have:
                mode, b, a, pb, pa_ = "daily (T+1 booking)", T, Tp1, Tm1, T
            elif {Tm1, T} <= have:
                mode, b, a, pb, pa_ = "daily", Tm1, T, Tm2, Tm1
            else:
                pb = pa_ = None
                m0 = pd.Timestamp(r.T.year, r.T.month, 1)
                b = cal[cal.searchsorted(m0) - 1]
                a = cal[cal.searchsorted(m0 + pd.offsets.MonthEnd(0), side="right") - 1]
                mode = "month-end"
                if b not in have or a not in have:
                    bb = [cal[i - k] for k in range(1, 7) if cal[i - k] in have]
                    aa = [cal[i + k] for k in range(0, 6) if cal[i + k] in have]
                    if bb and aa:
                        b, a, mode = bb[0], aa[0], "nearest"
            g, st = pair_trades(H, etf, b, a)
            if g is None:
                continue
            # visible AUM on b (last available Datastream day on or before b)
            A = av[etf]
            k = A.index.searchsorted(b, side="right") - 1
            aum_vis = A.aum.iloc[k] if k >= 0 else np.nan
            scale = aum_vis / st["E_b"]
            st.update(review=r.review, mode=mode, aum_visible=aum_vis, scale=scale, family=fam[etf])
            if pb is not None and pb in have and pa_ in have:
                gp, sp = pair_trades(H, etf, pb, pa_)
                if gp is not None:
                    placebo.append(gp.assign(etf=etf, review=r.review, D_vis_pl=gp.trade_s * scale,
                                             notional_pl=gp.trade_s * scale * gp.pa / AJf[PD.get_loc(pa_), gp.index.values.astype(int)]
                                             if pa_ in PD else np.nan).reset_index()[["etf", "review", "col", "delta", "D_vis_pl", "notional_pl", "hb_s", "ha_s"]])
                    st["placebo_cprime"] = sp["cprime"]; st["placebo_n_add"] = sp["n_add"]; st["placebo_n_del"] = sp["n_del"]
                nxt = cal[cal.get_loc(a) + 1]
                gl, sl = pair_trades(H, etf, a, nxt) if nxt in have else (None, None)
                if gl is not None:
                    st["late_n_add"] = sl["n_add"]; st["late_n_del"] = sl["n_del"]
                    st["late_abs_delta_p99"] = np.nanquantile(np.abs(gl.delta), 0.99)
            pairs.append(st)
            g = g.assign(D_vis=g.trade_s * scale, F_vis=g.funding_s * scale)
            per[etf] = g
        if not per:
            continue
        allc = sorted(set().union(*[set(g.index) for g in per.values()]))
        for c in allc:
            row = dict(review=r.review, T=r.T, col=c, lid=lid_ix[c], tic=L.tic.iloc[c], name=L.conm.iloc[c])
            for etf, g in per.items():
                f = fam[etf]
                if c in g.index:
                    x = g.loc[c]
                    row.update({f"h_b_{f}": x.h_b, f"h_a_{f}": x.h_a, f"w_b_{f}": x.w_b, f"w_a_{f}": x.w_a,
                                f"delta_{f}": x.delta, f"D_vis_{f}": x.D_vis, f"F_vis_{f}": x.F_vis, f"price_{f}": x.pa})
                else:
                    row.update({f"h_b_{f}": 0.0, f"h_a_{f}": 0.0, f"D_vis_{f}": 0.0, f"F_vis_{f}": 0.0})
                row[f"mode_{f}"] = [p for p in pairs if p["review"] == r.review and p["etf"] == etf][0]["mode"]
            rows.append(row)
        print(r.review, {e: (p["mode"], p["n_add"], p["n_del"]) for e in per for p in pairs if p["review"] == r.review and p["etf"] == e}, flush=True)
    PR = pd.DataFrame(pairs)
    PR.to_parquet(DATA / "ca_pairs.parquet", index=False)
    PL = pd.concat(placebo)
    PL.to_parquet(DATA / "ca_placebo.parquet", index=False)
    thr = {e: float(np.nanquantile(np.abs(PL[PL.etf == e].delta), 0.995)) for e in ("XIC", "XIU")}
    THR = max(0.005, max(thr.values()))
    print("placebo |delta| p99.5:", thr, "-> THR", THR)
    D = pd.DataFrame(rows)
    for f in ("TSX", "TSX60"):
        hb, ha, dl = D[f"h_b_{f}"].fillna(0), D[f"h_a_{f}"].fillna(0), D[f"delta_{f}"]
        D[f"cls_{f}"] = np.select([(hb == 0) & (ha > 0), (hb > 0) & (ha == 0), (hb > 0) & (dl.abs() > THR), hb > 0],
                                  ["add", "delete", "weight", "unchanged"], "none")
        D[f"D_vis_{f}"] = D[f"D_vis_{f}"].fillna(0.0)
    D["D_net"] = D.D_vis_TSX + D.D_vis_TSX60
    D["THR"] = THR
    # panel fields at T (split-adjusted units -> shares at T)
    flt = pd.read_parquet(TSX / "data" / "float_monthly.parquet")
    flt["ff"] = flt.ff_ds.fillna(flt.ff_ws)
    FF = flt.pivot_table(index="month", columns="line", values="ff")
    prevT = dict(zip(rv.review, rv["T"].shift(1)))
    ext = []
    for x in D.itertuples():
        e = {}
        if x.T in PD:
            iT = PD.get_loc(x.T); c = x.col; aj = AJf[iT - 1, c]
            vc = VC[iT - 65:iT - 5, c]
            cons = (~np.isnan(vc)).sum() >= 20
            Vb = VC if cons else VT
            w = Vb[iT - 65:iT - 5, c]
            e["adv60"] = np.nanmedian(w) / aj if (~np.isnan(w)).sum() >= 20 else np.nan
            e["vol_T"] = Vb[iT, c] / aj
            e["med20"] = np.nanmedian(Vb[iT - 20:iT, c]) / aj
            e["vol_basis"] = "CA consolidated" if cons else "TSX only"
            e["shares_out"] = SH[iT - 1, c] / aj
            e["price_Tm1"] = P[iT - 1, c]
            pT = prevT.get(x.review)
            if pd.notna(pT) and pT in PD:
                e["sh_chg_q"] = SH[iT - 1, c] / SH[PD.get_loc(pT) - 1, c] - 1
            m = pd.Timestamp(x.T.year, x.T.month, 1)
            ffv = FF.loc[m, c] if (m in FF.index and c in FF.columns) else np.nan
            e["ff"] = ffv
            e["d_scale"] = 1.0 / aj                  # D_vis is in split-adjusted units -> shares at T-1
        ext.append(e)
    D = pd.concat([D, pd.DataFrame(ext, index=D.index)], axis=1)
    for f in ("TSX", "TSX60"):
        D[f"D_vis_{f}_sh"] = D[f"D_vis_{f}"] * D.d_scale.fillna(1.0)
    D["D_net_sh"] = D.D_vis_TSX_sh + D.D_vis_TSX60_sh
    # weight-change component labels
    dl = D.delta_TSX.where(D.cls_TSX == "weight", D.delta_TSX60.where(D.cls_TSX60 == "weight"))
    shr = D.sh_chg_q
    D["wc_type"] = np.where(dl.isna(), "",
                            np.where(D.w_b_TSX.fillna(0) >= 9.5, "capping",
                                     np.where((dl - shr).abs() <= 0.25 * dl.abs() + 0.002, "share update", "IWF / other")))
    # event label: one per line, priority add/delete in the Composite, then TSX 60 moves, then weight, unchanged
    lab = np.select([D.cls_TSX == "add", D.cls_TSX == "delete", D.cls_TSX60 == "add", D.cls_TSX60 == "delete",
                     (D.cls_TSX == "weight") | (D.cls_TSX60 == "weight"), (D.cls_TSX == "unchanged") | (D.cls_TSX60 == "unchanged")],
                    ["add", "delete", "tsx60_add", "tsx60_delete", "weight", "unchanged"], "none")
    D["event"] = lab
    side = np.where(D.event == "add", "add", np.where(D.event == "delete", "delete", ""))
    D["in_sister_review_events"] = [(rv_, l, s) in revset for rv_, l, s in zip(D.review, D.lid, side)]
    D.loc[(D.event.isin(["add", "delete"])) & ~D.in_sister_review_events, "event"] = D.event + "_offcycle"
    D["notional_net"] = D.D_net_sh * D.price_Tm1.fillna(D.price_TSX)
    D["net_adv"] = D.D_net_sh / D.adv60
    D.to_csv(OUT / "ca_demand.csv", index=False)
    print(D.groupby("event").size())
    print(PR[["review", "etf", "mode", "n_b", "n_a", "unmatched_b", "unmatched_a", "cprime", "c_med", "m_index",
              "n_add", "n_del", "scale"]].tail(12).to_string())


if __name__ == "__main__":
    main()
