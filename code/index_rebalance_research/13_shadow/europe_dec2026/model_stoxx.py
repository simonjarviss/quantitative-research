"""STOXX Europe 600 December 2026 review: shadow ranking, adds/deletes, probabilities (paper only).

Rules [stoxx_europe_600.md §3, §10(a); STOXX-MG §7.2.2, p.67]: rank all eligible lines (most liquid line per company,
3-month ADTV > EUR 1m) by free-float market cap in EUR at the ranking cut-off C (Mon 30 Nov 2026 close); ranks
1-550 are selected; the remaining 50 places go to current members ranked 551-750 in rank order; then the largest
remaining. No fast entry/exit. Free float and shares freeze at F = Mon 23 Nov [STOXX-CG §8.2, p.24].

How the shadow ranking is built (the 12_studies/stoxx600_dax/05_predict_stoxx.py rule, fed with live public data):
  1. The latest public selection list (slpublic_sxxp, published on the first trading day of the month) gives STOXX's
     own Rank (FINAL) at the previous month-end cut-off C0, for every eligible line; it already embeds STOXX's free-float
     factors, line choice, country assignment, TMI membership and the ADTV screen (lines failing it carry no rank).
  2. Members' free-float market cap at C0 = EXSA index shares (holdings / k) x close(C0) x FX(C0) (the "anchors").
  3. A monotone curve log FFMC = g(rank) is fitted through the anchors (isotonic, linear between blocks); every ranked
     line gets FFMC(C0) = exp(g(official rank)), so the C0 ordering is STOXX's own.
  4. FFMC is rolled forward to the as-of close with each line's own price and FX return since C0 (shares and free
     float held at their C0 values; the 23 Nov factor update is not observable and sits in sigma_data).
  5. The selection rule is applied "as if the cut-off were today" -> predicted adds and deletes.
  6. Probabilities: margin = log distance to the line that flips the call (add line = between ranks 550/551; delete
     line = between the last member kept in the 551-750 buffer and the first one dropped, or rank 750 if the buffer
     is not crowded), p = Phi(margin / sqrt(sigma_rel^2 * h + sigma_data^2)), sigma_rel = 60-day daily volatility of
     the line's EUR return relative to the equal-weighted ranking band, h = trading days to C. A joint Monte Carlo of
     the same model (p_mc) includes buffer crowding. predict.py reports P(add) = margin probability and
     P(delete) = p_mc: the delete line moves with crowding of the 551-750 buffer, which only the joint simulation sees
     (June 2026 replay, scores/validate_june2026.md).
Membership now = EXSA holdings (lines on the list as members but not in EXSA were deleted since the list, e.g. M&A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import lib
import universe as U


def list_cutoff(list_date: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    first = pd.Timestamp(list_date.year, list_date.month, 1)
    s = sessions[sessions < first]
    return s[-1]


def match_extra(u: pd.DataFrame, unmatched: pd.DataFrame) -> dict:
    """EXSA lines not matched to a list member -> list non-member line (added since the list)."""
    out = {}
    for j, q in unmatched.iterrows():
        cand = u[(~u.member) & (u.suffix == q.suffix)]
        if cand.empty:
            continue
        sc = cand.apply(lambda r: U.sim(r.Instrument_Name, q["name"]) +
                        (0.5 if (U.RIC_ALIAS.get(r.RIC) or U.ric_symbol(r.RIC)) == q.symbol else 0), axis=1)
        if sc.max() >= 0.6:
            out[sc.idxmax()] = q
    return out


def run(sl: pd.DataFrame, exsa: pd.DataFrame, P: pd.DataFrame, symmap: dict, asof: pd.Timestamp,
        cutoff: pd.Timestamp, index_cap_eur: float | None = None, n_mc: int = C.N_MC, seed: int = 7) -> dict:
    asof, cutoff = pd.Timestamp(asof), pd.Timestamp(cutoff)
    W, ccy = lib.price_panel(P, "close")
    FX = lib.fx_to(P, "EUR")
    sessions = W.index[W.notna().sum(axis=1) > 50]
    list_date = pd.Timestamp(sl.list_date.iloc[0])
    C0 = list_cutoff(list_date, sessions)
    t = sessions[sessions <= asof][-1]
    u, unmatched = U.stoxx_universe(sl, exsa)
    unmatched = unmatched.assign(symbol=[U.exsa_symbol(a, b) for a, b in zip(unmatched.ticker, unmatched.exchange)],
                                 suffix=unmatched.exchange.map(U.EXSA_SUFFIX))
    extra = match_extra(u, unmatched)
    ex = exsa.set_index("isin")
    u["member_list"] = u.member
    u["member_now"] = u.exsa_isin.notna()
    for i, q in extra.items():
        u.loc[i, ["exsa_isin", "exsa_ticker", "member_now"]] = [q["isin"], q.symbol, True]
    u["deleted_since"] = u.member_list & ~u.member_now
    # symbols: EXSA ticker where matched, else the resolved map
    u["symbol"] = np.where(u.exsa_ticker.notna(), u.exsa_ticker, u.RIC.map(symmap))
    alt = u.RIC.map(symmap)
    have = set(W.columns)
    u["symbol"] = [s if s in have else (a if a in have else s) for s, a in zip(u.symbol, alt)]
    # EXSA members with no list line at all
    miss = exsa[~exsa["isin"].isin(set(u.exsa_isin.dropna()))]
    # prices
    u["ccy_px"] = u.symbol.map(ccy).fillna(u.Currency)
    p0 = lib.asof_value(W, C0).reindex(u.symbol).values
    pt = lib.asof_value(W, t).reindex(u.symbol).values
    fx0s, fxts = lib.asof_value(FX, C0), lib.asof_value(FX, t)
    fx0 = u.ccy_px.map(fx0s).values.astype(float)
    fxt = u.ccy_px.map(fxts).values.astype(float)
    u["px_C0"], u["px_t"], u["fx_C0"], u["fx_t"] = p0, pt, fx0, fxt
    u["ret_eur"] = (pt * fxt) / (p0 * fx0)
    # anchors: index shares F = H / k
    aum = float(exsa.mv.sum())
    k = aum / index_cap_eur if index_cap_eur else np.nan
    u["H"] = u.exsa_isin.map(ex.shares)
    u["F_index"] = u.H / k if np.isfinite(k) else u.H
    u["ffmc_anchor_C0"] = u.F_index * u.px_C0 * u.fx_C0
    anc = u.member_list & u.member_now & u.ffmc_anchor_C0.gt(0) & u["rank"].notna()
    g = lib.rank_curve(u.loc[anc, "rank"].values.astype(float), np.log(u.loc[anc, "ffmc_anchor_C0"].values))
    rho = pd.Series(u.loc[anc, "ffmc_anchor_C0"].values).rank().corr(pd.Series(-u.loc[anc, "rank"].values).rank())
    u["logF_C0"] = g(u["rank"].values.astype(float))
    u.loc[u["rank"].isna(), "logF_C0"] = np.log(u.loc[u["rank"].isna(), "ffmc_anchor_C0"])
    band = u["rank"].between(400, 900) & np.isfinite(u.ret_eur)
    med_ret = float(np.nanmedian(u.loc[band, "ret_eur"])) if band.any() else 1.0
    u["flags"] = ""
    noprice = ~np.isfinite(u.ret_eur)
    u.loc[noprice, "flags"] += "no_price;"
    st = u.symbol.map(lib.stale(W, t, P=P)).fillna(False).astype(bool)
    u.loc[st & ~noprice, "flags"] += "stale_price;"
    u["logF_t"] = u.logF_C0 + np.log(np.where(noprice, med_ret, u.ret_eur))
    u["F_implied"] = np.exp(u.logF_C0) / (u.px_C0 * u.fx_C0)
    # add EXSA members without a list line (ranked on holdings FFMC at t)
    if len(miss):
        rows = []
        for r in miss.itertuples():
            s = U.exsa_symbol(r.ticker, r.exchange)
            c = ccy.get(s, r.mkt_ccy)
            pt_ = lib.asof_value(W, t).get(s, np.nan)
            fxt_ = fxts.get(c, np.nan)
            F = r.shares / k if np.isfinite(k) else r.shares
            rows.append(dict(RIC="ISIN:" + r.isin, key="ISIN:" + r.isin, Instrument_Name=r.name, Country=r.isin[:2],
                             Currency=r.mkt_ccy, member_now=True, member_list=False, deleted_since=False,
                             symbol=s, exsa_isin=r.isin, F_index=F, px_t=pt_, fx_t=fxt_,
                             logF_t=np.log(F * pt_ * fxt_), flags="not_on_list;"))
        u = pd.concat([u, pd.DataFrame(rows)], ignore_index=True)
    # eligible at t: ranked lines (not deleted since the list) + current members
    elig = (u["rank"].notna() | u.member_now) & ~u.deleted_since & np.isfinite(u.logF_t)
    E = u[elig].copy().reset_index(drop=True)
    sel, info = select(E.logF_t.values, E.member_now.values.astype(bool))
    E["rank_t"] = info["rank"]
    E["selected"] = sel
    E["pred"] = np.select([sel & ~E.member_now, ~sel & E.member_now], ["add", "delete"], "none")
    # lines
    v = np.sort(E.logF_t.values)[::-1]
    line_add = (v[C.STOXX["upper"] - 1] + v[C.STOXX["upper"]]) / 2
    line_del = info["line_del"]
    E["line"] = np.where(E.member_now, line_del, line_add)
    E["line_name"] = np.where(E.member_now, info["line_del_name"], "rank 550/551")
    E["margin_log"] = np.where(E.member_now, E.logF_t - line_del, E.logF_t - line_add)   # + = above the line
    # volatility
    cols_band = E.loc[E.rank_t.between(350, 950), "symbol"].dropna().unique().tolist()
    Weur = W[[c for c in W.columns if c in set(E.symbol.dropna())]].copy()
    for s in Weur.columns:
        c = ccy.get(s, "EUR")
        if c in FX:
            Weur[s] = Weur[s] * FX[c].reindex(Weur.index).ffill()
    sig = lib.rel_sigma(Weur, t, cols_band)
    E["sigma_rel"] = E.symbol.map(sig).fillna(sig.median())
    h = C.trading_days_between(t, cutoff, "EU")
    E["h_days"] = h
    E["sigma_data"] = np.where(E["flags"].str.contains("no_price|not_on_list|stale_price"), C.SIGMA_DATA["flagged"],
                               np.where(E.member_now, C.SIGMA_DATA["member"], C.SIGMA_DATA["nonmember"]))
    E["sigma_tot"] = lib.sigma_total(E.sigma_rel, h, E.sigma_data)
    p_above = lib.p_cross(E.margin_log, E.sigma_tot)            # P(ends above its line)
    E["p_change"] = np.where(E.member_now, 1 - p_above, p_above)  # members: P(delete); non-members: P(add)
    E["p_margin"] = np.where(E.pred != "none", E.p_change, 1 - E.p_change)   # P(the call is right)
    E["margin_pct"] = np.exp(np.abs(E.margin_log)) - 1
    # Monte Carlo cross-check (buffer crowding included)
    E["p_mc_change"] = mc(E.logF_t.values, E.sigma_tot.values, E.member_now.values.astype(bool), n_mc, seed)
    E["ffmc_t_eur"] = np.exp(E.logF_t)
    E["F_new_est"] = np.where(E.member_now, E.F_index, E.F_implied)
    meta = dict(list_date=str(list_date.date()), C0=str(C0.date()), t=str(t.date()), h_days=h,
                anchors=int(anc.sum()), anchor_rank_spearman=float(rho), k=k, aum_eur=aum,
                index_cap_eur=index_cap_eur, line_add_eur=float(np.exp(line_add)), line_del_eur=float(np.exp(line_del)),
                buffer_members=int(info["n_buffer"]), buffer_slots=int(info["K"]),
                members_now=int(E.member_now.sum()), deleted_since=u.loc[u.deleted_since, "Instrument_Name"].tolist(),
                added_since=[q["name"] for q in extra.values()], no_price=int(noprice[elig].sum()),
                exsa_max_weight=float(exsa.weight.max()), cap=C.STOXX["cap"], median_ret_band=med_ret)
    return dict(E=E, U=u, meta=meta)


def select(logF: np.ndarray, member: np.ndarray):
    n = len(logF)
    order = np.argsort(-logF)
    rank = np.empty(n, int)
    rank[order] = np.arange(1, n + 1)
    up, lo, N = C.STOXX["upper"], C.STOXX["lower"], C.STOXX["n"]
    sel = rank <= up
    K = N - int(sel.sum())
    buf = np.where(member & (rank > up) & (rank <= lo))[0]
    buf = buf[np.argsort(rank[buf])]
    keep = buf[:max(K, 0)]
    sel[keep] = True
    if len(buf) > K:
        a, b = buf[K - 1], buf[K]
        line_del = (logF[a] + logF[b]) / 2
        line_name = f"buffer crowding: member #{K}/{K + 1} of 551-750"
    else:
        v = np.sort(logF)[::-1]
        line_del = (v[lo - 1] + v[lo]) / 2 if n > lo else v[-1]
        line_name = "rank 750/751"
    rem = N - int(sel.sum())
    if rem > 0:
        rest = np.where(~sel)[0]
        sel[rest[np.argsort(rank[rest])][:rem]] = True
    return sel, dict(rank=rank, line_del=line_del, line_del_name=line_name, n_buffer=len(buf), K=K)


def mc(logF, sig, member, n_mc, seed):
    rng = np.random.default_rng(seed)
    n = len(logF)
    cnt = np.zeros(n)
    up, lo, N = C.STOXX["upper"], C.STOXX["lower"], C.STOXX["n"]
    for b in range(0, n_mc, 500):
        m = min(500, n_mc - b)
        X = logF[None, :] + sig[None, :] * rng.standard_normal((m, n))
        order = np.argsort(-X, axis=1)
        rank = np.empty_like(order)
        rows = np.arange(m)[:, None]
        rank[rows, order] = np.arange(1, n + 1)[None, :]
        sel = rank <= up
        memo = member[order]                                      # member flag in rank order
        inbuf = memo & (np.arange(1, n + 1)[None, :] > up) & (np.arange(1, n + 1)[None, :] <= lo)
        cum = np.cumsum(inbuf, axis=1)
        keep_sorted = inbuf & (cum <= (N - up))
        filled = up + keep_sorted.sum(axis=1)
        selsorted = (np.arange(1, n + 1)[None, :] <= up) | keep_sorted
        # fill (rare): largest remaining
        short = N - filled
        if (short > 0).any():
            for r in np.where(short > 0)[0]:
                rest = np.where(~selsorted[r])[0][:short[r]]
                selsorted[r, rest] = True
        sel = np.zeros_like(selsorted)
        sel[rows, order] = selsorted
        cnt += (sel != member[None, :]).sum(axis=0)
    return cnt / n_mc
