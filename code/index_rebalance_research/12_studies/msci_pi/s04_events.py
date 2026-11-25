"""Step 4: holdings-implied index changes at every MSCI review 2018-02 .. 2026-08, from iShares UCITS daily holdings
(ishares.com/ch, s01) of CSCA (MSCI Canada), IMEA (MSCI Europe), WSML (MSCI World Small Cap) and SWDA (MSCI World,
cross-check), as net_flow/s04 did with XIC.

For each review T and fund: pre = last holdings file before T (T-1, else T-2/T-3), post-trade file = T. If the pre->T
pair shows no adds/deletes but T->T+1 does, the fund booked a day late and T->T+1 is used (mode 'T+1').
    c       = median over continuing names of h_T / h_pre          fund creation/redemption factor
    k_pre   = 90th percentile of h_pre / shares outstanding         the fund's ownership of a FIF = 1 name
    F_pre   = h_pre / k_pre,   F_T = h_T / (k_pre x c)              index float shares before / after (fund's basis)
Segments: Standard from CSCA (Canadian lines) and IMEA (DM Europe lines); Small Cap from the Canadian / European lines
of WSML (World Small Cap is sampled: a Small Cap member WSML does not hold looks like 'not in Small Cap').
Event types per line (market x review):
    ADD_STD   enters Standard from outside the IMI           DEL_STD   leaves Standard and the IMI
    MIG_UP    Small Cap -> Standard                          MIG_DOWN  Standard -> Small Cap
    ADD_SC    enters Small Cap from outside the IMI          DEL_SC    leaves Small Cap and the IMI
    W_STD / W_SC   continuing member with |delta| > 1%   (FIF / NOS update)   U_STD / U_SC   unchanged
Shares outstanding, prices and volume come from the sister panels (Canada: consolidated volume VC, Datastream 295;
Europe: primary-venue volume VP), matched by ISIN. Output: events.csv (review x market x line) and data/pairs.csv.
"""
import numpy as np
import pandas as pd

import holdings_ch as H
from common import DATA, EU, OUT, TSX, reviews

THR = 0.01
EU_C = H.EU_LOC

_z = np.load(TSX / "data" / "panel.npz")
Zc = {k: _z[k] for k in ("P", "VC", "SH", "AJ", "dates")}          # load once (npz members decompress on every access)
Dc = pd.DatetimeIndex(Zc["dates"])
Lc = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
_z = np.load(EU / "data" / "panel.npz")
Ze = {k: _z[k] for k in ("P", "VP", "SH", "AJ", "FX", "dates")}
De = pd.DatetimeIndex(Ze["dates"])
Le = pd.read_parquet(EU / "data" / "panel_lines.parquet")


def _col_map(L, prefer_col=None):
    m = {}
    for k, isin in enumerate(L["isin"].values):
        if isinstance(isin, str) and isin not in m:
            m[isin] = k
    return m


COLc = _col_map(Lc)
# Canadian ISIN = CA + 9-character CUSIP + check digit: interlisted names often carry no ISIN in the panel, so map
# the CUSIP too (first 8 characters; the line with the most priced days wins)
_npx = np.isfinite(Zc["P"]).sum(0)
for k in np.argsort(-_npx):
    cu = Lc["cusip"].values[k]
    if isinstance(cu, str) and len(cu) >= 8:
        COLc.setdefault("CA" + cu[:8], k)

# Europe: prefer the home/primary row of each ISIN
_le = Le.reset_index().sort_values(["is_prirow", "home_line"], ascending=False)
COLe = {}
for k, isin in zip(_le["index"], _le["isin"]):
    if isinstance(isin, str) and isin not in COLe:
        COLe[isin] = k


# historical ISIN -> today's ISIN (Datastream ds2isinchg, s02 isinmap): the panels carry current identifiers only
_IM = pd.read_parquet(DATA / "isin_map.parquet") if (DATA / "isin_map.parquet").exists() else pd.DataFrame(columns=["isin", "isin_now"])
ISIN_NOW = dict(zip(_IM["isin"], _IM["isin_now"]))


class _Map(dict):
    """ISIN -> panel column: the ISIN itself, else today's ISIN of the same Datastream security, else (Canada) the CUSIP."""
    def get(self, isin, default=None):
        v = dict.get(self, isin)
        if v is None and isin in ISIN_NOW:
            v = dict.get(self, ISIN_NOW[isin])
        if v is None and isinstance(isin, str) and isin.startswith("CA"):
            v = dict.get(self, isin[:10])
            if v is None and ISIN_NOW.get(isin, "").startswith("CA"):
                v = dict.get(self, ISIN_NOW[isin][:10])
        return default if v is None else v


COLc, COLe = _Map(COLc), _Map(COLe)


def panel_for(mkt):
    return (Zc, Dc, COLc) if mkt == "CA" else (Ze, De, COLe)


def shares_out(mkt, isins, d):
    Z, D, COL = panel_for(mkt)
    i = D.searchsorted(pd.Timestamp(d), side="right") - 1
    out = {}
    for s in isins:
        c = COL.get(s)
        if c is not None and i >= 0:
            v = Z["SH"][i, c]
            if np.isfinite(v) and v > 0:
                out[s] = v
    return out


def lines(fund, d, mkt):
    x = H.load(fund, d)
    if x is None:
        return None
    x = x[x.ctry == "CA"] if mkt == "CA" else x[x.ctry.isin(EU_C)]
    return x.set_index("isin")


def diff(fund, d0, d1, mkt):
    a, b = lines(fund, d0, mkt), lines(fund, d1, mkt)
    if a is None or b is None or len(a) < 5 or len(b) < 5:
        return None, None
    g = a[["ticker", "name", "shares", "price", "mkt_ccy", "ctry", "sector"]].rename(columns={"shares": "h0", "price": "p0"}).join(
        b[["shares", "price", "ticker", "name", "ctry"]].rename(columns={"shares": "h1", "price": "p1", "ticker": "ticker1",
                                                                       "name": "name1", "ctry": "ctry1"}), how="outer")
    g["ticker"] = g.ticker.fillna(g.ticker1)
    g["name"] = g["name"].fillna(g.name1)
    g["ctry"] = g.ctry.fillna(g.ctry1)
    g[["h0", "h1"]] = g[["h0", "h1"]].fillna(0.0)
    cont = (g.h0 > 0) & (g.h1 > 0)
    c = float(np.median((g.h1 / g.h0)[cont]))
    so = shares_out(mkt, list(a.index), d0)
    own = pd.Series({s: a.loc[s, "shares"] / v for s, v in so.items()})
    own = own[(own > 0) & (own < 1)]
    k0 = float(own.quantile(0.9)) if len(own) >= 5 else np.nan
    g["F0"] = g.h0 / k0
    g["F1"] = g.h1 / (k0 * c)
    g["delta"] = np.where(cont, (g.h1 / g.h0) / c - 1, np.nan)
    st = dict(fund=fund, d0=d0, d1=d1, mkt=mkt, n0=len(a), n1=len(b), c=c, k0=k0, n_so=len(own),
              n_add=int(((g.h0 == 0) & (g.h1 > 0)).sum()), n_del=int(((g.h0 > 0) & (g.h1 == 0)).sum()))
    return g, st


# Reviews whose trade date was a UK bank holiday: MSCI implements UK changes on the next business day at the previous
# close (gimi_size_segments.md §5 holiday rule), so funds book UK lines a day later. For these, Europe uses the
# pre -> T+1 pair and UK lines get no trade-day volume.
UK_HOLIDAY_T = {pd.Timestamp("2020-08-31"), pd.Timestamp("2026-08-31")}


def pick_pair(fund, T, mkt):
    a = H.around(fund, T)
    if mkt == "EU" and pd.Timestamp(T) in UK_HOLIDAY_T and a["pre"] is not None and a["post"] is not None:
        g, st = diff(fund, a["pre"], a["post"], mkt)
        if g is not None:
            st["mode"] = "pre->T+1 (UK holiday)"
            return g, st, st["mode"]
    if a["pre"] is None or a["T"] is None:
        return None, None, "none"
    g, st = diff(fund, a["pre"], a["T"], mkt)
    if g is None:
        return None, None, "none"
    mode = "T"
    if st["n_add"] + st["n_del"] == 0 and a["post"] is not None:
        g2, st2 = diff(fund, a["T"], a["post"], mkt)
        if g2 is not None and st2["n_add"] + st2["n_del"] > 0:
            g, st, mode = g2, st2, "T+1"
    st["mode"] = mode
    return g, st, mode


def panel_fields(mkt, isins, T):
    Z, D, COL = panel_for(mkt)
    if T not in D:
        return pd.DataFrame(index=isins)
    iT = D.get_loc(T)
    V = Z["VC"] if mkt == "CA" else Z["VP"]
    rows = {}
    for s in isins:
        c = COL.get(s)
        if c is None:
            continue
        aj = Z["AJ"][iT - 1, c] if np.isfinite(Z["AJ"][iT - 1, c]) and Z["AJ"][iT - 1, c] > 0 else 1.0
        w = V[iT - 65:iT - 5, c]
        w20 = V[iT - 20:iT, c]
        rows[s] = dict(col=c, adv60=np.nanmedian(w) / aj if np.isfinite(w).sum() >= 20 else np.nan,
                       med20=np.nanmedian(w20) / aj if np.isfinite(w20).sum() >= 10 else np.nan,
                       vol_T=V[iT, c] / aj, price_Tm1=Z["P"][iT - 1, c], shares_out=Z["SH"][iT - 1, c] / aj,
                       fx_eur=(Ze["FX"][iT - 1, c] if mkt != "CA" else np.nan))
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    RV = reviews()
    out, pairs = [], []
    for r in RV.itertuples():
        T = r.T
        for mkt, fstd in (("CA", "CSCA"), ("EU", "IMEA")):
            gs, ss, ms = pick_pair(fstd, T, mkt)
            gc, sc, mc = pick_pair("WSML", T, mkt)
            gw, sw, mw = pick_pair("SWDA", T, mkt)
            for st in (ss, sc, sw):
                if st:
                    pairs.append(st | dict(review=r.review, T=T))
            if gs is None:
                continue
            S = gs.add_suffix("_std")
            if gc is not None:
                S = S.join(gc[["h0", "h1", "F0", "F1", "delta", "ticker", "name", "ctry"]].add_suffix("_sc"), how="outer")
            else:
                for c in ("h0", "h1", "F0", "F1", "delta"):
                    S[c + "_sc"] = np.nan
            if gw is not None:
                S = S.join(gw[["h0", "h1", "delta"]].add_suffix("_wld"), how="left")
            S["ticker"] = S.get("ticker_std").fillna(S.get("ticker_sc")) if "ticker_sc" in S else S.ticker_std
            S["name"] = S.get("name_std").fillna(S.get("name_sc")) if "name_sc" in S else S.name_std
            S["ctry"] = S.get("ctry_std").fillna(S.get("ctry_sc")) if "ctry_sc" in S else S.ctry_std
            for c in ("h0_std", "h1_std", "F0_std", "F1_std"):
                S[c] = S[c].fillna(0.0)
            sc_ok = gc is not None
            for c in ("h0_sc", "h1_sc", "F0_sc", "F1_sc"):
                S[c] = S[c].fillna(0.0) if sc_ok else np.nan
            s0, s1 = S.h0_std > 0, S.h1_std > 0
            c0, c1 = (S.h0_sc > 0), (S.h1_sc > 0)
            ev = np.select(
                [~s0 & s1 & c0, ~s0 & s1, s0 & ~s1 & c1, s0 & ~s1,
                 ~c0 & c1 & ~s0, c0 & ~c1 & ~s1,
                 s0 & s1 & (S.delta_std.abs() > THR), s0 & s1, c0 & c1 & (S.delta_sc.abs() > THR), c0 & c1],
                ["MIG_UP", "ADD_STD", "MIG_DOWN", "DEL_STD", "ADD_SC", "DEL_SC", "W_STD", "U_STD", "W_SC", "U_SC"], "none")
            S["event"] = ev
            S["dF_std"] = S.F1_std - S.F0_std
            S["dF_sc"] = (S.F1_sc - S.F0_sc) if sc_ok else 0.0
            S["mode_std"], S["mode_sc"] = ms, mc
            S["review"], S["T"], S["mkt"] = r.review, T, mkt
            P = panel_fields(mkt, list(S.index), T)
            S = S.join(P)
            if mkt == "EU" and T in UK_HOLIDAY_T:
                S.loc[S.ctry == "GB", "vol_T"] = np.nan
            out.append(S.reset_index().rename(columns={"index": "isin"}))
        print(r.review, T.date(), {m: (len(o), o.event.value_counts().reindex(["ADD_STD", "DEL_STD", "MIG_UP", "MIG_DOWN", "ADD_SC", "DEL_SC"]).fillna(0).astype(int).tolist())
                                   for m, o in [(x.mkt.iloc[0], x) for x in out[-2:] if x.review.iloc[0] == r.review]}, flush=True)
    E = pd.concat(out, ignore_index=True)
    keep = ["review", "T", "mkt", "isin", "ticker", "name", "ctry", "event", "h0_std", "h1_std", "F0_std", "F1_std",
            "delta_std", "dF_std", "h0_sc", "h1_sc", "F0_sc", "F1_sc", "delta_sc", "dF_sc", "h0_wld", "h1_wld",
            "delta_wld", "mode_std", "mode_sc", "col", "adv60", "med20", "vol_T", "price_Tm1", "shares_out", "fx_eur"]
    E = E[[c for c in keep if c in E]]
    E.to_csv(OUT / "events.csv", index=False)
    pd.DataFrame(pairs).to_csv(DATA / "pairs.csv", index=False)
    print(E.groupby(["mkt", "event"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
