"""Step 6 (Europe): holdings-implied passive demand per stock at the quarterly STOXX 600, DAX-family and FTSE UK
reviews, from Thomson s12 quarter-end holdings of fully replicating UCITS trackers (2011-2025).

iShares UCITS daily history is not available: the DE/UK/CH sites ignore asOfDate (the file returned is always the
latest; the product page lists only the latest, last quarter-end and last year-end dates; probe 2026-10-11).
R2's tr_mutualfunds.s12 carries quarter-end holdings of the same funds. A quarter-end pair (end of the previous
quarter -> end of the review quarter) brackets each 3rd-Friday review; it also carries off-cycle changes inside the
quarter (fast entries, M&A deletions, large share changes applied at once), which is a stated limitation.
Rulers (s12 fundno): STOXX 45947 iShares STOXX Europe 600 (DE) [cross-check 138577 Xtrackers STOXX Europe 600],
DAX 64843 iShares Core DAX, MDAX 64844, TecDAX 64845, SDAX 103374 ComStage SDAX (2017-2023), FTSE100 80260 iShares
Core FTSE 100, FTSE250 80252 iShares FTSE 250.
Identifiers: s12 8-character cusip/CINS -> Datastream ds2cusipchg (date-valid) -> security -> primary quote ->
the STOXX/DAX sister panel line (ds_pq / ds_ic). Trades are computed per cusip, so unmapped names still count in the
size totals; volume and returns need the panel line.
Arithmetic as in s04: split basis from the panel ajexdi (or the s12 shares-outstanding ratio when unmapped), c' at
the post-date USD prices, trade = h_post/c' - h_pre, delta = (h_post/h_pre)/c - 1, D_vis = trade x AUM_visible
(Datastream, local currency, pre date) / E_fund (local currency).
THR (weight change) = 1%. Measurement noise is checked by running the STOXX review pairs on two different rulers.
Outputs: eu_demand.csv (review x family x line), data/eu_pairs.parquet, data/eu_ruler_check.parquet.
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, EU, CD

RULER = {"STOXX": 45947, "DAX": 64843, "MDAX": 64844, "TecDAX": 64845, "SDAX": 103374, "FTSE100": 80260, "FTSE250": 80252}
CHECK = {"STOXX": 138577}
THR = 0.01
FX_IC = {"EUR": 151392, "GBP": 151139}
CCY = {"STOXX": "EUR", "DAX": "EUR", "MDAX": "EUR", "TecDAX": "EUR", "SDAX": "EUR", "FTSE100": "GBP", "FTSE250": "GBP"}

S = pd.read_parquet(DATA / "s12_holdings.parquet")
S["rdate"] = pd.to_datetime(S.rdate)
S["c8"] = S.cusip.str[:8]
S = S[S.shares > 0]
C = pd.read_parquet(DATA / "ds_cusip.parquet").dropna(subset=["prim_infocode"])
C["c8"] = C.cusip.str[:8]
C["startdate"] = pd.to_datetime(C.startdate); C["enddate"] = pd.to_datetime(C.enddate)
Z = np.load(EU / "data" / "panel.npz")
P, VP, SH, AJ, FXm = Z["P"], Z["VP"], Z["SH"], Z["AJ"], Z["FX"]
AJf = pd.DataFrame(AJ).ffill().bfill().values
PD = pd.DatetimeIndex(Z["dates"])
L = pd.read_parquet(EU / "data" / "panel_lines.parquet")
ic2col = {}
for k, (a, b) in enumerate(zip(L.ds_pq, L.ds_ic)):
    for v in (a, b):
        if pd.notna(v) and int(v) not in ic2col:
            ic2col[int(v)] = k
etf = pd.concat([pd.read_parquet(CD / "data" / "etf_aum.parquet"), pd.read_parquet(DATA / "ds_etf_aum_extra.parquet")])
etf["marketdate"] = pd.to_datetime(etf.marketdate)
FXR = {c: etf[etf.infocode == ic].set_index("marketdate").pipe(lambda d: d.mktcap / d.mktcap_usd).sort_index()
       for c, ic in FX_IC.items()}
PI = pd.read_parquet(DATA / "pi_daily.parquet")


def col_of(c8s, d):
    """cusip8 -> panel column on date d (date-valid cusip spell, first primary quote that is in the panel)."""
    x = C[(C.c8.isin(c8s)) & (C.startdate <= d) & (C.enddate >= d)]
    out = {}
    for c8, pic in zip(x.c8, x.prim_infocode):
        k = ic2col.get(int(pic))
        if k is not None and c8 not in out:
            out[c8] = k
    return out


def asof(series, d):
    s = series[series.index <= d]
    return s.iloc[-1] if len(s) else np.nan


def pair(fund, pre, post, T):
    hb = S[(S.fundno == fund) & (S.rdate == pre)].groupby("c8").agg(h_b=("shares", "sum"), p_b=("prc", "first"),
                                                                     so_b=("shrout2", "first"), name=("stkname", "first"))
    ha = S[(S.fundno == fund) & (S.rdate == post)].groupby("c8").agg(h_a=("shares", "sum"), p_a=("prc", "first"),
                                                                      so_a=("shrout2", "first"), name_a=("stkname", "first"))
    if len(hb) < 20 or len(ha) < 20:
        return None, None
    g = hb.join(ha, how="outer")
    g["name"] = g.name.fillna(g.name_a)
    g[["h_b", "h_a"]] = g[["h_b", "h_a"]].fillna(0.0)
    cm = col_of(list(g.index), T)
    g["col"] = [cm.get(c, -1) for c in g.index]
    # split basis: panel ajexdi between the report dates when mapped, else the s12 shares-outstanding ratio
    ib, ia = PD.searchsorted(pre, side="right") - 1, PD.searchsorted(post, side="right") - 1
    f = np.ones(len(g))
    m = g.col.values >= 0
    f[m] = AJf[ia, g.col.values[m]] / AJf[ib, g.col.values[m]]
    so = (g.so_a / g.so_b).values
    r0 = (g.h_a / g.h_b).values
    um = ~m & np.isfinite(so) & ((so > 1.4) | (so < 0.7)) & (np.abs(np.log(np.where(r0 > 0, r0, np.nan) / so)) < 0.25)
    f[um] = so[um]
    g["split_f"] = f
    g["hb_s"] = g.h_b * f
    cont = (g.hb_s > 0) & (g.h_a > 0)
    pr = (g.p_a / (g.p_b / f))[cont].median()
    pa = g.p_a.fillna(g.p_b / f * pr)
    cprime = (g.h_a * pa).sum() / (g.hb_s * pa).sum()
    r = g.h_a / g.hb_s
    c = np.median(r[cont])
    g["delta"] = np.where(cont, r / c - 1, np.nan)
    g["trade"] = g.h_a / cprime - g.hb_s
    g["funding"] = np.where(cont, g.hb_s * (c / cprime - 1), 0.0)
    g["p_usd"] = pa
    E_b_usd = (g.h_b * g.p_b).sum()
    st = dict(fund=fund, pre=pre, post=post, T=T, n_b=len(hb), n_a=len(ha), cprime=cprime, c_med=c, m_index=cprime / c,
              E_b_usd=E_b_usd, n_add=int(((g.hb_s == 0) & (g.h_a > 0)).sum()), n_del=int(((g.hb_s > 0) & (g.h_a == 0)).sum()),
              mapped=float(m.mean()))
    return g, st


def review_dates():
    rs = pd.read_csv(EU / "reviews_stoxx.csv", parse_dates=["T"])[["review", "T"]]
    rd = pd.read_csv(EU / "reviews_dax.csv", parse_dates=["T"])[["review", "T"]]
    return {"STOXX": rs, "DAX": rd, "MDAX": rd, "TecDAX": rd, "SDAX": rd, "FTSE100": rs, "FTSE250": rs}


def main():
    RV = review_dates()
    rows, pairs, chk = [], [], []
    for fam, fund in RULER.items():
        rdts = np.array(sorted(S[S.fundno == fund].rdate.unique()), dtype="datetime64[ns]")
        for r in RV[fam].itertuples():
            T = r.T
            pre_c = rdts[(rdts < np.datetime64(T)) & (rdts >= np.datetime64(T - pd.Timedelta(days=100)))]
            post_c = rdts[(rdts >= np.datetime64(T)) & (rdts <= np.datetime64(T + pd.Timedelta(days=20)))]
            if not len(pre_c) or not len(post_c):
                continue
            pre, post = pd.Timestamp(pre_c.max()), pd.Timestamp(post_c.min())
            g, st = pair(fund, pre, post, T)
            if g is None:
                continue
            ccy = CCY[fam]
            fx = asof(FXR[ccy], pre)                          # local per USD
            E_loc = st["E_b_usd"] * fx
            pi = PI[(PI.family == fam) & (PI.date <= pre)].sort_values("date").tail(1)
            aum = float(pi.aum.iloc[0]) if len(pi) else np.nan
            scale = aum / E_loc
            st.update(family=fam, review=r.review, fx=fx, aum_visible=aum, scale=scale)
            pairs.append(st)
            g = g.assign(D_vis=g.trade * scale, family=fam, review=r.review, T=T, fund=fund, pre=pre, post=post)
            rows.append(g.reset_index())
            if fam in CHECK:
                rdc = np.array(sorted(S[S.fundno == CHECK[fam]].rdate.unique()), dtype="datetime64[ns]")
                if np.datetime64(pre) in rdc and np.datetime64(post) in rdc:
                    g2, _ = pair(CHECK[fam], pre, post, T)
                    if g2 is not None:
                        j = g[["delta", "trade", "hb_s"]].join(g2[["delta", "hb_s"]], rsuffix="_chk", how="inner")
                        j["review"] = r.review
                        j["cls"] = np.where(j.hb_s == 0, "add", np.where(j.delta.isna(), "delete", "cont"))
                        chk.append(j.reset_index())
        print(fam, len([p for p in pairs if p["family"] == fam]), "review pairs", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D["cls"] = np.select([(D.hb_s == 0) & (D.h_a > 0), (D.hb_s > 0) & (D.h_a == 0), (D.hb_s > 0) & (D.delta.abs() > THR),
                          D.hb_s > 0], ["add", "delete", "weight", "unchanged"], "none")
    D["THR"] = THR
    # panel fields at T
    ext = []
    for x in D.itertuples():
        e = {}
        if x.col >= 0 and x.T in PD:
            iT = PD.get_loc(x.T); c = x.col; aj = AJf[iT - 1, c]
            w = VP[iT - 65:iT - 5, c]
            e["adv60"] = np.nanmedian(w) / aj if (~np.isnan(w)).sum() >= 20 else np.nan
            e["vol_T"] = VP[iT, c] / aj
            e["med20"] = np.nanmedian(VP[iT - 20:iT, c]) / aj
            e["price_Tm1"] = P[iT - 1, c]
            e["fx_eur"] = FXm[iT - 1, c]
            e["shares_out"] = SH[iT - 1, c] / aj
            e["aj_ratio"] = AJf[PD.searchsorted(x.post, side="right") - 1, c] / aj   # holdings basis (post) -> T-1 shares
        ext.append(e)
    D = pd.concat([D, pd.DataFrame(ext, index=D.index)], axis=1)
    D["D_vis_sh"] = D.D_vis / D.aj_ratio.fillna(1.0)          # shares on the T-1 basis
    D["lid"] = np.where(D.col >= 0, L.lid.reindex(D.col.clip(lower=0)).values, "")
    D.loc[D.col < 0, "lid"] = ""
    D["notional_eur"] = D.D_vis_sh * D.price_Tm1 * D.fx_eur
    D.to_csv(OUT / "eu_demand.csv", index=False)
    pd.DataFrame(pairs).to_parquet(DATA / "eu_pairs.parquet", index=False)
    if chk:
        pd.concat(chk).to_parquet(DATA / "eu_ruler_check.parquet", index=False)
    print(D.groupby(["family", "cls"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
