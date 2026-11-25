"""FTSE 100 / FTSE 250 December 2026 quarterly review: full-cap ranking, moves, probabilities (paper only).

Rules [ftse_uk_100_250_allshare.md §3.2, §10.1-10.8; FTSE-UK-GR §8.3.4, p.19; §6.6, p.15; §8.1.1, p.18]:
companies ranked by FULL market cap (price x shares in issue, before investability weighting) on the FTSE UK
Monitored List at the close of Tue 1 Dec 2026. FTSE 100: non-member in at rank <= 90, member out at rank >= 111;
FTSE 250: non-FTSE 350 in at <= 325, member out at >= 376; constant counts (extra moves: lowest-ranked members out /
highest-ranked non-members in); FTSE 100 deletions normally go to the FTSE 250 [§8.3.4].

Inputs (free):
  membership   FTSE 100 = ISF holdings; FTSE 250 = Wikipedia table (checked against MIDD, a sampling fund);
               SmallCap candidates = R2 Datastream All-Share list 23 Jun 2026 minus today's FTSE 350, minus names with
               a FTSE UK constituent-deletion notice since.
  shares       Yahoo sharesOutstanding (not FTSE's Monitored List count, which is updated quarterly from the
               30 Oct shares cut-off); R2 Datastream numshrs (Jun 2026) as a fallback and a >15% disagreement flag.
  prices       Yahoo daily close (LSE official close proxy), USD/EUR lines converted to GBP.
Probabilities: margin to the rule line (ranks 90/91, 110/111, 325/326, 375/376), sigma_rel x sqrt(h) with h = LSE
trading days to 1 Dec, plus sigma_data for shares; joint Monte Carlo for the constant-count moves.
Not modelled: nationality decisions, the June liquidity test results (failures are excluded until next June: we
cannot see them, all current All-Share members are assumed to pass), new issues since 4 Aug, free-float < 10%.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config as C
import lib
import universe as U


def deleted_names(notices: pd.DataFrame, since: str = "2026-06-23") -> list[str]:
    if notices is None or notices.empty:
        return []
    n = notices.copy()
    n["AnnouncedDate"] = pd.to_datetime(n.AnnouncedDate, errors="coerce")
    n = n[(n.AnnouncedDate >= pd.Timestamp(since)) & n.Title.str.contains(r"Constituent Deletion|Scheme of Reconstruction|Merger with", case=False, na=False)]
    out = []
    for t in n.Title:
        m = re.match(r"^(.*?)\s*\((?:UK|Netherlands|Canada)\)", t)
        if m:
            out.append(m.group(1).strip())
    return out


def ftse_select(cap: pd.Series, cur100: set, cur250: set) -> tuple[set, set]:
    order = cap.sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(order) + 1), index=order.index)
    I = [g for g in rank.index if g not in cur100 and rank[g] <= C.FTSE["in100"]]
    D = [g for g in rank.index[::-1] if g in cur100 and rank[g] >= C.FTSE["out100"]]
    if len(I) > len(D):
        D += [g for g in rank.index[::-1] if g in cur100 and g not in D][:len(I) - len(D)]
    elif len(D) > len(I):
        I += [g for g in rank.index if g not in cur100 and g not in I][:len(D) - len(I)]
    new100 = (cur100 - set(D)) | set(I)
    prov = (cur250 - set(I)) | set(D)
    I2 = [g for g in rank.index if g not in new100 and g not in prov and rank[g] <= C.FTSE["in250"]]
    D2 = [g for g in rank.index[::-1] if g in prov and g not in D and rank[g] >= C.FTSE["out250"]]
    new250 = (prov - set(D2)) | set(I2)
    if len(new250) > C.FTSE["n250"]:
        extra = [g for g in rank.index[::-1] if g in new250 and g in prov][:len(new250) - C.FTSE["n250"]]
        new250 -= set(extra)
    elif len(new250) < C.FTSE["n250"]:
        extra = [g for g in rank.index if g not in new250 and g not in new100][:C.FTSE["n250"] - len(new250)]
        new250 |= set(extra)
    return new100, new250


def run(fu: pd.DataFrame, P: pd.DataFrame, symmap: dict, info: pd.DataFrame, notices: pd.DataFrame,
        asof: pd.Timestamp, cutoff: pd.Timestamp, index_caps: dict, H: dict, n_mc: int = 2000, seed: int = 5,
        members_override: dict | None = None, shares_override: pd.Series | None = None) -> dict:
    asof, cutoff = pd.Timestamp(asof), pd.Timestamp(cutoff)
    W, ccy = lib.price_panel(P, "close")
    FXg = lib.fx_to(P, "GBP")
    sessions = W.index[W.notna().sum(axis=1) > 50]
    t = sessions[sessions <= asof][-1]
    u = fu.copy()
    if members_override:
        u = members_override["universe"].copy()
    u["symbol"] = u.symbol.where(u.symbol.notna(), u.key.map(symmap))
    alt = u.key.map(symmap)
    have = set(W.columns)
    u["symbol"] = [s if isinstance(s, str) and s in have else (a if isinstance(a, str) and a in have else s)
                   for s, a in zip(u.symbol, alt)]
    # the same company twice: a SmallCap candidate from the R2 June list whose Yahoo line is already a FTSE 350 member
    # (Datastream keeps old ISINs, e.g. IHG GB00B1WQCS47, and abbreviated names, e.g. HARBOURVEST GLOBAL, that the
    # ISIN/name match in universe.ftse_universe misses)
    sym350 = set(u.loc[u.cur != "SmallCap", "symbol"].dropna())
    dup = (u.cur == "SmallCap") & u.symbol.isin(sym350)
    dup_names = u.loc[dup, "name"].tolist()
    u = u[~dup].copy()
    dels = deleted_names(notices)
    u["deleted_notice"] = [any(U.sim(n, d) >= 0.8 for d in dels) for n in u["name"]]
    u = u[~(u.deleted_notice & (u.cur == "SmallCap"))].copy()
    # shares in issue
    inf = info.set_index("symbol") if len(info) else pd.DataFrame()
    def col(c):
        return pd.to_numeric(u.symbol.map(inf[c]), errors="coerce") if c in inf else pd.Series(np.nan, index=u.index)
    so, imp, fl = col("sharesOutstanding"), col("impliedSharesOutstanding"), col("floatShares")
    r2s = pd.to_numeric(u.get("r2_numshrs_jun"), errors="coerce") if "r2_numshrs_jun" in u else pd.Series(np.nan, index=u.index)
    u["shares_yf"] = so
    # R2 Datastream numshrs is already in units (e.g. Vesuvius 248,171,717 on 30 Jun 2026); a former "x 1e3" here
    # made every Yahoo-less name 1000x too large (fixed 2026-10-13)
    u["shares"] = so.where(so > 0, imp.where(imp > 0, r2s))
    u["shares_src"] = np.where(so > 0, "yahoo sharesOutstanding", np.where(imp > 0, "yahoo implied",
                                                                          np.where(r2s > 0, "R2 numshrs Jun", "none")))
    if shares_override is not None:
        so2 = u.key.map(shares_override)
        u["shares"] = so2.where(so2 > 0, u.shares)
        u["shares_src"] = np.where(so2 > 0, "override", u.shares_src)
    u["float_ratio_yf"] = (fl / so).clip(upper=1.0)
    u["flags"] = ""
    # prices in GBP
    pc = u.symbol.map(ccy).fillna("GBP")
    px = u.symbol.map(lib.asof_value(W, t))
    fx = pc.map(lib.asof_value(FXg, t)).values.astype(float)
    u["ccy_px"] = pc
    u["px_t"] = px
    u["px_gbp"] = px * fx
    u["cap_gbp"] = u.px_gbp * u.shares
    # shares sanity: ISF-implied investability factor for FTSE 100 names; MIDD for FTSE 250
    k100 = float(H["ISF"].mv.sum()) / index_caps["FTSE100"] if index_caps.get("FTSE100") else np.nan
    k250 = float(H["MIDD"].mv.sum()) / index_caps["FTSE250"] if index_caps.get("FTSE250") else np.nan
    u["F_index"] = np.where(u.cur == "FTSE100", u.hold_shares / k100, np.where(u.cur == "FTSE250", u.hold_shares / k250, np.nan))
    u["iwf_implied"] = u.F_index / u.shares
    # fund-scale correction: fully investable names should sit at iwf 1.0; MIDD (a sampling fund, ~233 of 250 names)
    # puts them at ~1.07-1.11 because k = AUM / index cap understates its holding ratio in the names it holds.
    # Rescale F_index per index by the median iwf_implied of the 0.9-1.3 cluster (2026-10-13).
    k_adj = {}
    for c in ("FTSE100", "FTSE250"):
        x = u.loc[(u.cur == c) & (u.shares_src == "yahoo sharesOutstanding"), "iwf_implied"]
        core = x[(x > 0.9) & (x < 1.3)]
        k_adj[c] = float(core.median()) if len(core) >= 10 else 1.0
        u.loc[u.cur == c, "F_index"] = u.loc[u.cur == c, "F_index"] / k_adj[c]
    u["iwf_implied"] = u.F_index / u.shares
    bad = (u.iwf_implied > 1.15) | (u.iwf_implied < 0.15)
    u.loc[bad & u.iwf_implied.notna(), "flags"] += "shares_vs_fund;"
    # full shares can never be below the fund-implied index shares: where Yahoo's count is >15% lower (e.g. NCC Group)
    # or missing (e.g. Croda, Vesuvius), use the index shares (investability weight 1 assumed). Fixed 2026-10-13.
    low = u.F_index.notna() & ~(u.shares * 1.15 >= u.F_index)
    u.loc[low, "flags"] += "shares_from_fund;"
    u.loc[low, "shares_src"] = "fund index shares (Yahoo >15% lower or missing)"
    u.loc[low, "shares"] = u.loc[low, "F_index"]
    u["cap_gbp"] = u.px_gbp * u.shares
    # SmallCap candidates with no Yahoo bar on either of the last two sessions: delisted or suspended (e.g. Senior,
    # last bar 8 Oct 2026 with zero volume) -> not candidates
    last_bar = W[W.index <= t].notna()[::-1].idxmax() if len(W) else pd.Series(dtype="datetime64[ns]")
    recent = sessions[sessions <= t][-2]
    gone = (u.cur == "SmallCap") & (u.symbol.map(last_bar) < recent)
    gone_names = u.loc[gone, "name"].tolist()
    u.loc[gone, "cap_gbp"] = np.nan
    u.loc[gone, "flags"] += "no_recent_price;"
    u.loc[~np.isfinite(u.cap_gbp), "flags"] += "no_cap;"
    u.loc[u.symbol.map(lib.stale(W, t, P=P)).fillna(False).astype(bool), "flags"] += "stale_price;"
    u.loc[u.key.isin(C.FTSE_KNOWN), "flags"] += "known_issue;"
    u["F_est"] = u.F_index.where(u.F_index > 0, u.shares * u.float_ratio_yf.fillna(0.85))
    E = u[np.isfinite(u.cap_gbp) & (u.cap_gbp > 0)].copy().set_index("key")
    cur100 = set(E.index[E.cur == "FTSE100"])
    cur250 = set(E.index[E.cur == "FTSE250"])
    logc = np.log(E.cap_gbp)
    new100, new250 = ftse_select(E.cap_gbp, cur100, cur250)
    order = E.cap_gbp.sort_values(ascending=False)
    E["rank_t"] = pd.Series(np.arange(1, len(order) + 1), index=order.index)
    E["pred"] = np.where(E.index.isin(new100), "FTSE100", np.where(E.index.isin(new250), "FTSE250", "SmallCap"))
    v = np.log(order.values)

    def line(r):
        return (v[r - 1] + v[r]) / 2
    lines = {"in100": line(90), "out100": line(110), "in250": line(325), "out250": line(375)}
    # volatility
    Wg = W[[c for c in W.columns if c in set(E.symbol.dropna())]].copy()
    for s in Wg.columns:
        c = ccy.get(s, "GBP")
        if c in FXg and c != "GBP":
            Wg[s] = Wg[s] * FXg[c].reindex(Wg.index).ffill()
    sig = lib.rel_sigma(Wg, t, E.loc[E.rank_t <= 450, "symbol"].dropna().tolist())
    E["sigma_rel"] = E.symbol.map(sig).fillna(sig.median())
    h = C.trading_days_between(t, cutoff, "XLON")
    E["h_days"] = h
    E["sigma_data"] = np.where(E["flags"].str.contains("shares_vs_fund|stale_price") | (E.shares_src != "yahoo sharesOutstanding"),
                               0.10, 0.03)
    E["sigma_tot"] = lib.sigma_total(E.sigma_rel, h, E.sigma_data)
    rows = []
    for g, r in E.iterrows():
        lc = np.log(r.cap_gbp)
        cand = []
        if r.cur == "FTSE100":
            cand.append(("FTSE100->FTSE250", "out100", -1))
        else:
            cand.append((f"{r.cur}->FTSE100", "in100", +1))
        if r.cur == "FTSE250":
            cand.append(("FTSE250->SmallCap", "out250", -1))
        elif r.cur == "SmallCap":
            cand.append(("SmallCap->FTSE250", "in250", +1))
        for lab, ln, sgn in cand:
            m = lc - lines[ln]
            pa = float(lib.p_cross(m, r.sigma_tot))
            p = pa if sgn > 0 else 1 - pa
            if p >= 0.01:
                rows.append(dict(key=g, move=lab, line=ln, margin_log=m, p_margin=p, primary=True))
    MV = pd.DataFrame(rows)
    det = [(g, f"{E.loc[g, 'cur']}->{E.loc[g, 'pred']}") for g in E.index if E.loc[g, "pred"] != E.loc[g, "cur"]]
    rng = np.random.default_rng(seed)
    cnt = {}
    sd = E.sigma_tot.values
    for _ in range(n_mc):
        x = pd.Series(np.exp(logc.values + sd * rng.standard_normal(len(E))), index=E.index)
        a, b = ftse_select(x, cur100, cur250)
        for g in (cur100 - a) | (a - cur100) | (cur250 - b) | (b - cur250):
            new = "FTSE100" if g in a else "FTSE250" if g in b else "SmallCap"
            k = (g, f"{E.loc[g, 'cur']}->{new}")
            if E.loc[g, "cur"] != new:
                cnt[k] = cnt.get(k, 0) + 1
    MC = pd.DataFrame([dict(key=g, move_simple=m, p_mc=c / n_mc) for (g, m), c in cnt.items()])
    meta = dict(t=str(t.date()), h_days=h, n_ranked=int(len(E)), counts={"FTSE100": len(cur100), "FTSE250": len(cur250),
                "SmallCap_candidates": int((E.cur == "SmallCap").sum())},
                lines_gbp={k: float(np.exp(x)) for k, x in lines.items()}, k100=k100, k250=k250,
                deleted_by_notice=u.loc[u.deleted_notice, "name"].tolist() if "deleted_notice" in u else [],
                dup_smallcap_dropped=dup_names, smallcap_no_recent_price=gone_names, fund_scale_adj=k_adj,
                shares_from_fund=u.loc[u["flags"].str.contains("shares_from_fund"), "name"].tolist(),
                no_cap=u.loc[~np.isfinite(u.cap_gbp), "name"].tolist(),
                shares_flagged=int(u["flags"].str.contains("shares_vs_fund").sum()))
    return dict(E=E.reset_index(), U=u, MV=MV, MC=MC, det=det, lines=lines, meta=meta)
