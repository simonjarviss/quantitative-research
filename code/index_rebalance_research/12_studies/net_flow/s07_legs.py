"""Step 7: close-dislocation legs for every line with holdings-implied demand (Canada, US Russell, Europe), so that
weight-change flows in existing members can be tested like the adds and deletes.

Legs (../close_dislocation/legs.py arithmetic; sessions relative to T, the close at which index funds trade):
  d0 = T-1 close -> T close; on1 = T close -> T+1 open; id1 = T+1 open -> close; r1 = T close -> T+1 close;
  r1_5 = T close -> T+5 close; r1_20 = T close -> T+20 close (sums of daily abnormal returns).
Control for every line: the equal-weighted mean of the review's UNCHANGED continuing members (|delta| <= THR, no add,
delete or TSX 60 move) in the same market-cap quintile (quintiles of all continuing members at T-1), excluding the line.
Adds and deletes also get these legs (for comparison); the flow test uses the sister study's matched-control legs for
them (events_legs.csv) and these legs for weight changes.
Sign: + = in the direction of net visible demand (sign of D_net).
Returns: CA = ../tsx_composite panel RET; US = CRSP dsf (../russell_june/_cache); EU = ../stoxx600_dax panel RET.
Opens: ../close_dislocation/data opens_ca (Compustat), opens_us (CRSP dlyopen), opens_ds (Datastream primary quote).
Output: data/legs_net.parquet (one row per market x event date x line).
"""
import numpy as np
import pandas as pd
import duckdb
from common import DATA, OUT, CD, TSX, EU, RUS, sessions, legs_mod

CLIP = (-0.5, 1.0)
LEGS = ["pre5", "d0", "on1", "id1", "r1", "r1_5", "r1_20"]
K = np.arange(-5, 21)


def legs_for(Rw, ctrl_cols, c, sub, subc):
    """Rw: (26, n) clipped daily returns k=-5..20; ctrl_cols: control column indices; c: own column."""
    cm = np.nanmean(Rw[:, ctrl_cols], 1) if len(ctrl_cols) else np.full(26, np.nan)
    out = legs_mod.daily_legs(Rw[:, c], cm, np.zeros(26))
    s = {k: v[c] for k, v in sub.items()}
    cms = {k: np.nanmean(v[ctrl_cols]) if len(ctrl_cols) else np.nan for k, v in sub.items()}
    out.update(legs_mod.subday_legs(s, cms, {k: 0.0 for k in sub}))
    return {f"ab_{k}": v[1] for k, v in out.items() if k in LEGS} | {f"raw_{k}": v[0] for k, v in out.items() if k in LEGS}


def quintile(x):
    q = pd.Series(x).rank(pct=True)
    return np.ceil(q * 5).clip(1, 5).values


def build(market, D, RET, OPEN, CLOSE, dates, key_T="T"):
    """D: rows with columns T, col (panel column), event, D_net_sh, mcap. Returns DataFrame of legs."""
    rows = []
    R_cl = np.clip(RET, *CLIP)
    for T, g in D.groupby(key_T):
        if T not in dates:
            continue
        iT = dates.get_loc(T)
        if iT + 1 >= len(dates):
            continue
        ks = np.arange(iT - 5, min(iT + 21, len(dates)))
        g = g[g.col >= 0].copy()
        cols = g.col.values.astype(int)
        Rw = np.full((26, len(g)), np.nan)
        Rw[:len(ks)] = R_cl[ks][:, cols]
        sub = {}
        for key, i in (("on0", iT), ("on1", iT + 1)):
            on, idr = legs_mod.split_session(RET[i, cols], OPEN[i, cols], CLOSE[i, cols])
            sub["on" + key[-1]] = np.clip(on, *CLIP); sub["id" + key[-1]] = np.clip(idr, *CLIP)
        g["q"] = quintile(g.mcap.values)
        pool = (g.event == "unchanged").values
        for j, x in enumerate(g.itertuples()):
            cc = np.where(pool & (g.q.values == x.q))[0]
            cc = cc[cc != j]
            if len(cc) < 5:
                cc = np.where(pool)[0]; cc = cc[cc != j]
            lg = legs_for(Rw, cc, j, sub, None)
            rows.append(dict(idx=x.Index, n_ctrl=len(cc), **lg))
    L = pd.DataFrame(rows).set_index("idx")
    out = D.join(L, how="inner")
    sgn = np.sign(out.D_net_sh).replace(0, np.nan)
    for k in LEGS:
        out[f"s_{k}"] = out[f"ab_{k}"] * sgn
    out["market"] = market
    return out


def ca():
    D = pd.read_csv(OUT / "ca_demand.csv", parse_dates=["T"])
    z = np.load(TSX / "data" / "panel.npz")
    RET, P, SH, AJ = z["RET"], z["P"], z["SH"], z["AJ"]
    AJf = pd.DataFrame(AJ).ffill().bfill().values
    dates = pd.DatetimeIndex(z["dates"])
    o = pd.read_parquet(CD / "data" / "opens_ca.parquet")
    o["datadate"] = pd.to_datetime(o.datadate)
    Lp = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
    lid_ix = pd.Index(Lp.lid)
    o["col"] = lid_ix.get_indexer(o.gvkey + "_" + o.iid)
    o = o[(o.col >= 0) & o.datadate.isin(dates)]
    ok = (o.prcod > 0) & (o.prccd > 0) & (o.prcstd.astype(float) != 4)
    OPEN = np.full(P.shape, np.nan); CLOSE = np.full(P.shape, np.nan)
    pos = dates.get_indexer(o.datadate)
    OPEN[pos[ok.values], o.col[ok].values] = o.prcod[ok].values; CLOSE[pos[ok.values], o.col[ok].values] = o.prccd[ok].values
    D = D[D["T"].isin(dates) & D.event.isin(["add", "delete", "tsx60_add", "tsx60_delete", "weight", "unchanged"])].copy()
    iT = dates.get_indexer(D["T"]); cc = D.col.values.astype(int)
    D["mcap"] = P[iT - 1, cc] * SH[iT - 1, cc] / AJf[iT - 1, cc]
    return build("CA", D, RET, OPEN, CLOSE, dates)


def us():
    D = pd.read_csv(OUT / "us_demand.csv", parse_dates=["T"])
    D = D[D.event.isin(["BUY_add", "SELL_del", "BUY_mig", "SELL_mig", "weight", "unchanged"])].copy()
    con = duckdb.connect()
    cal = sessions("XNYS")
    perm = sorted(D.permno.unique())
    rows = []
    for T in sorted(D["T"].unique()):
        i = cal.get_loc(pd.Timestamp(T))
        rows.append(con.execute(f"""SELECT permno, dlycaldt, ret FROM read_parquet('{RUS}/_cache/dsf/*.parquet')
                    WHERE dlycaldt BETWEEN '{cal[i-6]:%Y-%m-%d}' AND '{cal[i+21]:%Y-%m-%d}'""").df())
    d = pd.concat(rows)
    d = d[d.permno.isin(perm)]
    dates = pd.DatetimeIndex(sorted(d.dlycaldt.unique()))
    pidx = pd.Index(perm)
    RET = np.full((len(dates), len(perm)), np.nan)
    RET[dates.get_indexer(d.dlycaldt), pidx.get_indexer(d.permno)] = d.ret.values
    o = pd.read_parquet(CD / "data" / "opens_us.parquet")
    o = o[o.permno.isin(perm) & o.dlycaldt.isin(dates)]
    OPEN = np.full(RET.shape, np.nan); CLOSE = np.full(RET.shape, np.nan)
    ok = (o.dlyopen > 0) & (o.dlyclose > 0)
    OPEN[dates.get_indexer(o.dlycaldt[ok]), pidx.get_indexer(o.permno[ok])] = o.dlyopen[ok]
    CLOSE[dates.get_indexer(o.dlycaldt[ok]), pidx.get_indexer(o.permno[ok])] = o.dlyclose[ok]
    # dates must be contiguous sessions for the k-window: rebuild on the NYSE calendar restricted to loaded dates
    D["col"] = pidx.get_indexer(D.permno)
    D["mcap"] = D.price_Tm1 * D.shares_out
    D["D_net_sh"] = D.D_net
    out = []
    for T, g in D.groupby("T"):
        i = cal.get_loc(T)
        win = cal[i - 6:i + 22]
        idx = dates.get_indexer(win)
        if (idx < 0).any():
            keep = idx >= 0
            win, idx = win[keep], idx[keep]
        out.append(build("US", g.assign(T=T), RET[idx], OPEN[idx], CLOSE[idx], pd.DatetimeIndex(win)))
    return pd.concat(out)


def eu():
    D = pd.read_csv(OUT / "eu_demand.csv", parse_dates=["T"])
    D = D[D.col >= 0]
    # net across families per line and T; event priority add/delete > weight > unchanged
    pr = {"add": 4, "delete": 3, "weight": 2, "unchanged": 1, "none": 0}
    D["prio"] = D.cls.map(pr)
    agg = D.groupby(["T", "col"]).agg(D_net_sh=("D_vis_sh", "sum"), adv60=("adv60", "first"), vol_T=("vol_T", "first"),
                                      price_Tm1=("price_Tm1", "first"), shares_out=("shares_out", "first"),
                                      fx_eur=("fx_eur", "first"), lid=("lid", "first"), prio=("prio", "max"),
                                      families=("family", lambda s: "+".join(sorted(set(s)))),
                                      review=("review", "first"))
    agg["event"] = agg.prio.map({v: k for k, v in pr.items()})
    for f in ("STOXX", "DAX", "MDAX", "TecDAX", "SDAX", "FTSE100", "FTSE250"):
        x = D[D.family == f].set_index(["T", "col"])
        agg[f"D_vis_{f}"] = x.D_vis_sh.reindex(agg.index).fillna(0.0).values
        agg[f"cls_{f}"] = x.cls.reindex(agg.index).fillna("none").values
    agg = agg.reset_index()
    z = np.load(EU / "data" / "panel.npz")
    RET, P, SH, AJ, FX = z["RET"], z["P"], z["SH"], z["AJ"], z["FX"]
    AJf = pd.DataFrame(AJ).ffill().bfill().values
    dates = pd.DatetimeIndex(z["dates"])
    Lp = pd.read_parquet(EU / "data" / "panel_lines.parquet")
    o = pd.read_parquet(CD / "data" / "opens_ds.parquet")
    o["marketdate"] = pd.to_datetime(o.marketdate)
    ic2col = {int(v): k for k, v in enumerate(Lp.ds_pq) if pd.notna(v)}
    o["col"] = o.infocode.map(ic2col)
    o = o.dropna(subset=["col"])
    o = o[o.marketdate.isin(dates)]
    ok = (o.open_ > 0) & (o.close_ > 0)
    OPEN = np.full(P.shape, np.nan); CLOSE = np.full(P.shape, np.nan)
    OPEN[dates.get_indexer(o.marketdate[ok]), o.col[ok].astype(int)] = o.open_[ok]
    CLOSE[dates.get_indexer(o.marketdate[ok]), o.col[ok].astype(int)] = o.close_[ok]
    agg = agg[agg["T"].isin(dates)].copy()
    iT = dates.get_indexer(agg["T"]); cc = agg.col.values.astype(int)
    agg["mcap"] = P[iT - 1, cc] * SH[iT - 1, cc] / AJf[iT - 1, cc] * FX[iT - 1, cc]
    return build("EU", agg, RET, OPEN, CLOSE, dates)


if __name__ == "__main__":
    parts = []
    for f in (ca, us, eu):
        x = f()
        print(f.__name__, len(x), x.groupby("event").size().to_dict(), flush=True)
        parts.append(x)
    A = pd.concat(parts, ignore_index=True)
    A.to_parquet(DATA / "legs_net.parquet", index=False)
