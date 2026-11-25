"""Step 3: per-event legs on MSCI review closes -> data/legs_msci.parquet (one row per kept event).

Prices (main basis): Datastream (data/ds_px.parquet, s01). Canada = the CAD Toronto (CA-region) quote of the line,
also for interlisted names whose Datastream primary quote is in New York (s01 ca_extra); Europe = the line's primary
quote. Daily total return = ri_t / ri_{t-1} - 1 on the market calendar (TSX
sessions; pan-European sessions), NaN when the quote did not trade that session (see traded()), clipped to
[-50%, +100%] as in close_dislocation. Sub-day pieces from open_ / close_ of the same quote and session:
  id = close / open - 1;  on = (1 + total return) / (1 + id) - 1   (close_dislocation legs.split_session)
Legs (k = sessions relative to T; legs.py of close_dislocation, unchanged): d0 = T-1 close -> T close; d0_on = T-1
close -> T open; d0_id = T open -> T close; on1 = T close -> T+1 open; id1 = T+1 open -> close; r1 = T close -> T+1
close; r1_5, r1_20 = T close -> T+5 / T+20 close (sums of daily abnormal returns; a missing stock day counts 0).
Abnormal = stock minus the equal-weighted mean of its matched-control cell (same quote source, same session; the
controls' own opens for the sub-day pieces). Index-adjusted = stock minus the cap-weighted (T-1 caps) mean of the
S&P/TSX Composite members (Canada) or STOXX Europe 600 members (Europe) on the same Datastream basis.
Matched controls (close_dislocation's cells, re-formed at f = T-20 sessions, about MSCI's price cut-off and before
the announcement): lines priced in the last 10 sessions with >= 60 of the last 126 sessions priced, market cap within
[0.15x, 6x] of the review's median Standard-event cap (the role of the S&P/TSX and STOXX rank thresholds there),
excluding every line with any MSCI index event at the review (Standard or Small Cap add/delete/migration).
  Canada: Canadian, GICS-classified, non-Venture TSX lines, price >= C$1; cell = size decile x 6-month momentum tercile
          (>= 5 controls), else size decile (s04_ca.py cells).
  Europe: one home line per operating company in a DM Europe country; chain country x size decile x momentum ->
          country x size quintile x momentum -> bloc x decile x momentum -> bloc x decile -> all x decile x
          momentum -> all x decile (>= 5 controls each; s05_eu.py STOXX chain).
Cross-check basis: the same legs, controls and cells on the sister panels' Compustat close-to-close total returns
(cs_* columns; close-to-close legs only), i.e. close_dislocation's own return source.
Stale-open check (added 2026-10-15; extra columns only, no existing leg changes): for d0_on (open at T) and on1 (open
at T+1), stale = the quote's open equals its previous session's close. ctrl_stale_<leg> = share of the event's
controls with an overnight piece that are stale; ctrl_n_<leg> = that count; abx_<leg> = the leg against the controls
without a stale open; ix_stale_<leg> = cap-weighted share of index members that are stale; ixx_<leg> = the leg
against the index without them; stale_<leg> = the event's own open is stale.
"""
import numpy as np
import pandas as pd

from common import (CLIP, DATA, DM_EUROPE, EU, KS, MPI, OUT, TSX, ALL_INDEX_EVENTS, _load, ca_calendar, eu_calendar,
                    legs_mod)

daily_legs, split_session, subday_legs, flatten = (legs_mod.daily_legs, legs_mod.split_session, legs_mod.subday_legs,
                                                   legs_mod.flatten)
ec = _load("eu_common_sd", EU / "common.py")
F_LAG = 20


def traded(vo, cl, op):
    """A quote traded in the session if Datastream shows positive volume; where the quote reports no volume at all
    on that day, a positive close with an open counts (Datastream pads non-trading days with the last close and no
    volume and no open)."""
    return (vo > 0) | (np.isnan(vo) & (cl > 0) & (op > 0))


class Market:
    def __init__(self, mkt):
        self.mkt = mkt
        if mkt == "CA":
            z = np.load(TSX / "data" / "panel.npz")
            self.cal = ca_calendar()
            self.P, self.RET, self.SH, self.AJ = z["P"], z["RET"], z["SH"], z["AJ"]
            self.V, self.Vfb = z["VC"], z["VT"]          # consolidated; TSX-only fallback (s04_ca.py)
            FX = 1.0
            self.L = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
            self.MEM = np.load(TSX / "data" / "member_matrix.npy")
            self.ixcs = pd.read_parquet(TSX / "data" / "index_proxy.parquet").proxy_tr.fillna(0).values
            L = self.L
            self.base = (((L["loc"] == "CAN") | (L.fic == "CAN")).values & L.gsubind.notna().values & (L.exchg != 9).values)
            self.cty = np.array(["CAN"] * len(L))
            cands = []
            for c, s in enumerate(L.ds_ca_ics.astype(str)):
                for x in s.split(","):
                    x = x.strip()
                    if x and x.replace(".0", "").isdigit():
                        cands.append((c, int(float(x))))
            # interlisted names with a US primary quote: their CAD Toronto quote(s), mapped in s01 ca_extra
            xm = pd.read_parquet(DATA / "ca_extra_ics.parquet")
            cands += list(zip(xm.col.astype(int), xm.infocode.astype(int)))
        else:
            z = np.load(EU / "data" / "panel.npz")
            self.cal = eu_calendar()
            self.P, self.RET, self.SH, self.AJ = z["P"], z["RET"], z["SH"], z["AJ"]
            self.V, self.Vfb = z["VP"], None             # primary venue
            FX = z["FX"]
            self.VAL = z["VAL"]
            self.L = pd.read_parquet(EU / "data" / "panel_lines.parquet")
            self.MEM = np.load(EU / "data" / "member_stoxx.npy")
            self.ixcs = pd.read_parquet(EU / "data" / "index_proxy.parquet").stoxx_proxy_tr_local.fillna(0).values
            L = self.L
            self.cty = L.home_cty.astype(str).values
            self.base = (L.gsector.notna().values & (L.tpci == "0").values & L.home_line.values
                         & np.isin(self.cty, DM_EUROPE))
            cands = [(c, int(v)) for c, v in enumerate(L.ds_pq.values) if np.isfinite(v)]
        assert len(z["dates"]) == len(self.cal) and (pd.DatetimeIndex(z["dates"]) == self.cal).all()
        self.nd, self.nl = self.P.shape
        AJf = pd.DataFrame(self.AJ).ffill().values
        self.MCAP = self.P * self.SH / AJf * FX
        self.R_cs = np.clip(self.RET, *CLIP)
        self.TRI = np.nancumprod(np.nan_to_num(self.RET) + 1, axis=0)
        self.gv = self.L.gvkey.values
        self.cands = pd.DataFrame(cands, columns=["col", "ic"])
        self._ds()

    def _ds(self):
        x = pd.read_parquet(DATA / "ds_px.parquet")
        x = x[x.infocode.isin(set(self.cands.ic))]
        x = x[x.marketdate.isin(self.cal)]
        self.ics = np.array(sorted(x.infocode.unique()))
        icix = pd.Index(self.ics)
        pos = np.array(sorted(set(self.cal.get_indexer(x.marketdate))))
        self.pos = pos
        self.prow = pd.Series(np.arange(len(pos)), index=pos)
        r = self.prow.loc[self.cal.get_indexer(x.marketdate)].values
        c = icix.get_indexer(x.infocode)
        shp = (len(pos), len(self.ics))
        self.RI, self.OP, self.CL, self.VO, self.BID, self.ASK = (np.full(shp, np.nan) for _ in range(6))
        self.EXC = np.full(shp, -1, dtype=np.int32)
        for arr, col in [(self.RI, "ri"), (self.OP, "open_"), (self.CL, "close_"), (self.VO, "volume"),
                         (self.BID, "bid"), (self.ASK, "ask")]:
            v = x[col].values.astype(float)
            arr[r, c] = np.where(v > 0, v, np.nan) if col != "volume" else v
        self.EXC[r, c] = x.exchintcode.fillna(-1).astype(int).values
        self.TR = traded(self.VO, self.CL, self.OP)
        # daily DS total return at row r = ri[r] / last ri on or before the previous calendar session - 1.
        # Datastream has no row when a quote's own market is shut (e.g. a UK bank holiday on a pan-European session),
        # so the previous ri is carried forward up to 5 sessions within the same window block.
        prev = self.prow.reindex(pos - 1).values
        block = np.cumsum(np.r_[1, np.diff(pos) > 1])
        RIff = pd.DataFrame(self.RI).groupby(block).ffill(limit=5).values
        self.DRET = np.full(shp, np.nan)
        okp = ~np.isnan(prev)
        pr = prev[okp].astype(int)
        self.DRET[okp] = self.RI[okp] / RIff[pr] - 1
        self.DRET[~self.TR] = np.nan
        self.DRET = np.clip(self.DRET, *CLIP)
        self.cands["icx"] = icix.get_indexer(self.cands.ic)
        self.cands = self.cands[self.cands.icx >= 0]

    def rows(self, i0, i1):
        """compact rows for calendar positions i0..i1 (inclusive); -1 where not pulled"""
        return self.prow.reindex(np.arange(i0, i1 + 1)).fillna(-1).astype(int).values

    def pick(self, iT):
        """line -> DS infocode column for the window T-6..T+20: the candidate quote with most traded sessions,
        ties to the one that traded at T."""
        rr = self.rows(iT - 6, iT + 20)
        rr = rr[rr >= 0]
        cnt = self.TR[rr][:, self.cands.icx.values].sum(0)
        rT = self.prow.get(iT, -1)
        atT = self.TR[rT, self.cands.icx.values] if rT >= 0 else np.zeros(len(self.cands), bool)
        c = self.cands.assign(n=cnt, t=atT.astype(int)).sort_values(["col", "n", "t"], ascending=[True, False, False])
        c = c[c.n > 0].drop_duplicates("col")
        icx = np.full(self.nl, -1)
        icx[c.col.values] = c.icx.values
        return icx

    def gather(self, A, rr, icx):
        """A[rows rr, ic columns of each line] -> (len(rr), nl), NaN where missing"""
        out = np.full((len(rr), self.nl), np.nan)
        okc = icx >= 0
        for j, r in enumerate(rr):
            if r >= 0:
                out[j, okc] = A[r, icx[okc]]
        return out


def one_line_per_company(M, mask, f):
    adtv = np.nan_to_num(np.nanmean(M.VAL[max(0, f - 62):f + 1], 0), nan=-1)
    idx = np.where(mask)[0]
    df = pd.DataFrame({"g": M.gv[idx], "c": idx, "a": adtv[idx]}).sort_values(["g", "a"], ascending=[True, False])
    m = np.zeros(M.nl, bool)
    m[df.drop_duplicates("g").c.values] = True
    return m


def chain_keys(M, f, univ):
    mc = M.MCAP[f]
    mc = np.where(np.isnan(mc), pd.DataFrame(M.MCAP[max(0, f - 9):f + 1]).ffill().values[-1], mc)
    mom = M.TRI[f] / M.TRI[max(0, f - 126)] - 1
    sb = np.nanquantile(mc[univ], np.linspace(0.1, 0.9, 9))
    mb = np.nanquantile(mom[univ], [1 / 3, 2 / 3])
    sdec = np.searchsorted(sb, np.nan_to_num(mc, nan=0.0))
    squi = sdec // 2
    mter = np.searchsorted(mb, np.nan_to_num(mom, nan=0.0))
    if M.mkt == "CA":
        return [("sz_mom", (sdec * 3 + mter).astype(str)), ("sz", np.char.add("d", sdec.astype(str)))]
    keyset = M.cty.astype(str)
    blocs = np.array([ec.bloc(c) for c in keyset])
    return [("cty_dec_mom", np.char.add(np.char.add(keyset, "|"), (sdec * 3 + mter).astype(str))),
            ("cty_qui_mom", np.char.add(np.char.add(keyset, "|q"), (squi * 3 + mter).astype(str))),
            ("bloc_dec_mom", np.char.add(np.char.add(blocs.astype(str), "|"), (sdec * 3 + mter).astype(str))),
            ("bloc_dec", np.char.add(np.char.add(blocs.astype(str), "|d"), sdec.astype(str))),
            ("all_dec_mom", (sdec * 3 + mter).astype(str)),
            ("all_dec", np.char.add("d", sdec.astype(str)))]


def run(mkt, EV, DEM):
    M = Market(mkt)
    print(mkt, "DS quotes", len(M.ics), "rows", len(M.pos), flush=True)
    E = EV[(EV.mkt == mkt) & (EV["drop"] == "")]
    D = DEM[DEM.mkt == mkt]
    out = []
    for review, e in E.groupby("review"):
        T = e["T"].iloc[0]
        iT = M.cal.get_loc(T)
        f = iT - F_LAG
        icx = M.pick(iT)
        rr = M.rows(iT - 5, iT + 20)
        Rds = M.gather(M.DRET, rr, icx)
        # sub-day pieces on T (0) and T+1 (1), DS open/close of the same quote
        sub = {}
        stale = {}      # (added 2026-10-15) open equal to the previous session's close, every line (controls, members)
        for key, i in [("0", iT), ("1", iT + 1)]:
            r1 = M.rows(i, i)
            ret, op, cl = (M.gather(A, r1, icx)[0] for A in (M.DRET, M.OP, M.CL))
            tr = M.gather(M.TR.astype(float), r1, icx)[0] > 0
            op = np.where(tr, op, np.nan)
            on, idr = split_session(ret, op, cl)
            sub["on" + key] = np.clip(on, *CLIP)
            sub["id" + key] = np.clip(idr, *CLIP)
            pcl = M.gather(M.CL, M.rows(i - 1, i - 1), icx)[0]
            stale[key] = np.isfinite(op) & np.isfinite(pcl) & (op == pcl)
        # index (cap-weighted members, T-1 caps, DS basis)
        w = np.where(M.MEM[iT - 1], np.nan_to_num(M.MCAP[iT - 1]), 0.0)
        xw = np.array([np.sum(w[m] * v[m]) / np.sum(w[m]) if (m := np.isfinite(v) & (w > 0)).any() else np.nan for v in Rds])
        ixs = {k: (np.sum(w[m] * v[m]) / np.sum(w[m]) if (m := np.isfinite(v) & (w > 0)).any() else np.nan)
               for k, v in sub.items()}
        # stale-open check (added 2026-10-15): cap-weighted share of index members with a stale open, and the index
        # overnight piece without them
        ix_st, ixs_x = {}, {}
        for key in ("0", "1"):
            v = sub["on" + key]
            m = np.isfinite(v) & (w > 0)
            ix_st[key] = np.sum(w[m & stale[key]]) / np.sum(w[m]) if m.any() else np.nan
            mx = m & ~stale[key]
            ixs_x[key] = np.sum(w[mx] * v[mx]) / np.sum(w[mx]) if mx.any() else np.nan
        # Compustat cross-check returns (sister panel)
        ks = np.arange(iT - 5, min(iT + 21, M.nd))
        Rcs = np.full((len(KS), M.nl), np.nan)
        Rcs[:len(ks)] = M.R_cs[ks]
        xcs = np.full(len(KS), np.nan)
        xcs[:len(ks)] = M.ixcs[ks]
        # controls
        excl = np.zeros(M.nl, bool)
        ecols = D[(D.review == review) & D.event.isin(ALL_INDEX_EVENTS)].col.dropna().astype(int).values
        excl[ecols] = True
        scols = D[(D.review == review) & D.event.isin(["ADD_STD", "DEL_STD", "MIG_UP", "MIG_DOWN"])].col.dropna().astype(int).values
        Lref = np.nanmedian(M.MCAP[f, scols])
        live = (~np.isnan(M.P[max(0, f - 9):f + 1])).any(0)
        ok_hist = (~np.isnan(M.P[max(0, f - 126):f + 1])).sum(0) >= 60
        mc0 = M.MCAP[f]
        if mkt == "CA":
            base = live & M.base & (M.P[f] >= 1)
        else:
            base = one_line_per_company(M, live & M.base, f)
        univ = base & (mc0 >= 0.15 * Lref) & (mc0 <= 6 * Lref) & ok_hist
        ctrl = univ & ~excl
        chain = chain_keys(M, f, univ)
        # spreads at the T close (DS bid/ask of the quote)
        rT = M.rows(iT, iT)
        bid, ask = (M.gather(A, rT, icx)[0] for A in (M.BID, M.ASK))
        spr = np.where((bid > 0) & (ask > bid), (ask - bid) / ((ask + bid) / 2), np.nan)
        rT1 = M.rows(iT + 1, iT + 1)
        bid1, ask1 = (M.gather(A, rT1, icx)[0] for A in (M.BID, M.ASK))
        spr1 = np.where((bid1 > 0) & (ask1 > bid1), (ask1 - bid1) / ((ask1 + bid1) / 2), np.nan)
        clT = M.gather(M.CL, rT, icx)[0]
        exT = M.gather(M.EXC.astype(float), rT, icx)[0]
        opT1 = M.gather(M.OP, rT1, icx)[0]
        r20 = M.rows(iT + 20, iT + 20)
        cl20 = M.gather(M.CL, r20, icx)[0] if r20[0] >= 0 else np.full(M.nl, np.nan)
        AJl = pd.DataFrame(M.AJ[max(0, iT - 30):iT]).ffill().values[-1]
        for r in e.itertuples():
            c = int(r.col)
            # ADV60 = median volume T-65..T-6 (>= 20 sessions), split-adjusted to T-1; Canada: consolidated, else the
            # TSX-only volume (interlisted names whose Datastream CA quote is not in the sister panel), as s04_ca.py
            aj = AJl[c] if np.isfinite(AJl[c]) and AJl[c] > 0 else 1.0
            adv_used, adv_basis = np.nan, ""
            for V, nm_ in [(M.V, "CA consolidated" if mkt == "CA" else "primary venue"), (M.Vfb, "TSX only")]:
                if V is None:
                    continue
                w = V[iT - 65:iT - 5, c]
                if np.isfinite(w).sum() >= 20:
                    adv_used, adv_basis = np.nanmedian(w) / aj, nm_
                    break
            cm, lvl = None, "none"
            for nm_, keys in chain:
                mbr = ctrl & (keys == keys[c])
                mbr[c] = False
                if mbr.sum() >= 5:
                    cm, lvl = mbr, nm_
                    break
            row = dict(market=mkt, review=review, event_date=T, isin=r.isin, ticker=r.ticker, name=r.name, ctry=r.ctry,
                       uk=bool(r.uk), event=r.event, etype=r.etype, side=int(r.side), group=r.group, col=c,
                       infocode=int(M.ics[icx[c]]) if icx[c] >= 0 else np.nan, exch_T=exT[c], ctrl_level=lvl,
                       n_ctrl=int(cm.sum()) if cm is not None else 0, in_univ=bool(univ[c]),
                       mcap_f=M.MCAP[f, c], mcap_ref=Lref, price_Tm1=r.price_Tm1, D_vis=r.D_vis, dvis_pct_so=r.dvis_pct_so,
                       dvis_adv=r.dvis_adv, adv60=r.adv60, adv60_used=adv_used, adv_basis=adv_basis,
                       dvis_adv_used=r.D_vis / adv_used if adv_used > 0 else np.nan, med20=r.med20, vol_T=r.vol_T, fx_eur=r.fx_eur,
                       spread_T=spr[c], spread_T1=spr1[c],
                       ds_close_T=clT[c], cs_close_T=M.P[iT, c],
                       open_T1_eqprev=bool(np.isfinite(opT1[c]) and np.isfinite(clT[c]) and opT1[c] == clT[c]),
                       ds_close_T20_ok=bool(np.isfinite(cl20[c])),
                       d0_ret_is_zero=bool(np.isfinite(Rds[5, c]) and Rds[5, c] == 0.0))
            if cm is None:
                out.append(row)
                continue
            legs = daily_legs(Rds[:, c], np.nanmean(Rds[:, cm], 1), xw)
            s = {k: v[c] for k, v in sub.items()}
            cms = {k: np.nanmean(v[cm]) for k, v in sub.items()}
            legs.update(subday_legs(s, cms, ixs))
            row.update(flatten(legs))
            # stale-open check (added 2026-10-15): share of the event's controls (with an overnight piece) whose open
            # equals the previous close; the overnight legs against controls / index members without stale opens
            for key, leg in (("0", "d0_on"), ("1", "on1")):
                v = sub["on" + key]
                okc = cm & np.isfinite(v)
                xs = okc & ~stale[key]
                sv = v[c]
                row[f"ctrl_stale_{leg}"] = float(stale[key][okc].mean()) if okc.any() else np.nan
                row[f"ctrl_n_{leg}"] = int(okc.sum())
                row[f"abx_{leg}"] = sv - v[xs].mean() if (xs.any() and np.isfinite(sv)) else np.nan
                row[f"ix_stale_{leg}"] = ix_st[key]
                row[f"ixx_{leg}"] = sv - ixs_x[key] if (np.isfinite(ixs_x[key]) and np.isfinite(sv)) else np.nan
                row[f"stale_{leg}"] = bool(stale[key][c])
            lc = daily_legs(Rcs[:, c], np.nanmean(Rcs[:, cm], 1), xcs)
            row.update({f"cs_{k}": v for k, v in flatten(lc).items()})
            out.append(row)
        print(mkt, review, len(e), "events;", int(ctrl.sum()), "controls in universe", flush=True)
    return pd.DataFrame(out)


def main():
    EV = pd.read_csv(OUT / "events_msci.csv", parse_dates=["T"])
    EV["drop"] = EV["drop"].fillna("")
    DEM = pd.read_csv(MPI / "demand.csv", usecols=["review", "mkt", "event", "col"])
    parts = [run(m, EV, DEM) for m in ("CA", "EU")]
    X = pd.concat(parts, ignore_index=True)
    X.to_parquet(DATA / "legs_msci.parquet", index=False)
    print(X.groupby(["market", "group"]).agg(n=("col", "size"), with_ctrl=("n_ctrl", lambda v: (v > 0).sum()),
                                              d0_ok=("ab_d0", lambda v: v.notna().sum()),
                                              on1_ok=("ab_on1", lambda v: v.notna().sum())))
    print(X.ctrl_level.value_counts())


if __name__ == "__main__":
    main()
