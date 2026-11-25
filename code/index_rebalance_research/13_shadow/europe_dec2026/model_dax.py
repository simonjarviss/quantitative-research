"""DAX / MDAX / SDAX / TecDAX December 2026 review (Fast Exit / Fast Entry only): shadow selection list, moves,
probabilities (paper only).

Rules [dax_family.md §3, §10(a); DAX-MG §7.1.2, p.34-39]: one selection list ranks every FSE Regulated Market
company by free-float market cap of its largest share class, priced at the 20-trading-day adjusted Xetra VWAP at the
cut-off (Mon 30 Nov 2026, 17:30 CET). December applies only Fast Exit and Fast Entry [DAX-MG §7.1.2 "Overview of
rules", p.37]:  DAX exit > 60 / entry <= 33 (+ positive EBITDA in the two latest fiscal years for newcomers);
MDAX 110 / 83; SDAX 180 / 153; TecDAX (own technology ranking) 45 / 25. Indices are reviewed DAX -> MDAX -> SDAX;
a fast exit is replaced by the best-ranked non-member (DAX: EBITDA and rank <= 53); a fast entry removes the
lowest-ranked member; counts are restored from relegation candidates first (study 06_predict_dax.py select()).

Shadow ranking (the backtest rule logic, fed with the public selection list):
  1. slpublic_mkdx / slpublic_tdxk (published 22:00 CET on the 3rd trading day) give STOXX's own rank at the previous
     month-end cut-off C0, membership (DAXK/MKDX/SDXK, TDXK) and eligibility comments (unranked = not eligible).
  2. Anchors: index shares from iShares EXS1/EXS3/EXS2 (holdings / k) x 20-day VWAP proxy at C0 (mean of Yahoo daily
     typical price (H+L+C)/3 on Xetra). Names without a fund: Yahoo floatShares x VWAP, scaled to the anchors
     (median ratio), as weaker points.
  3. A monotone curve log FFMC = g(rank) through those points (isotonic) gives every ranked company its C0 value, so
     the C0 order is STOXX's own; it is rolled to the as-of date with each name's VWAP20 ratio.
  4. Rules applied as if the cut-off were today -> predicted moves (deterministic).
  5. p_margin: margin to the rule line that flips the move (rank 60/61, 33/34, 110/111, 83/84, 180/181, 153/154;
     TecDAX 45/46, 25/26 on tech ranks), sigma = sqrt(sigma_rel^2 * h_eff + sigma_data^2) where h_eff accounts for
     the 20-day VWAP averaging (lib.h_eff_vwap). p_mc: joint Monte Carlo of the full rule cascade (needed for the
     forced moves: the member pushed out by a fast entry, the replacement for a fast exit, count restoration).
EBITDA: not observable for free at this horizon; assumed positive for DAX candidates (flagged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import lib

LV = {"DAX": 3, "MDAX": 2, "SDAX": 1}
LV_NAME = {3: "DAX", 2: "MDAX", 1: "SDAX", 0: "none"}


def list_cutoff(list_date: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    first = pd.Timestamp(list_date.year, list_date.month, 1)
    return sessions[sessions < first][-1]


# ------------------------------------------------------------------------------------- rules (from 06_predict_dax.py)
def select(rk: dict, members: set, n: int, thr: dict, ok_entry, superior: set) -> set:
    """One index, Fast Exit then Fast Entry (December). rk: company -> rank (eligible only)."""
    cur = {g for g in members if g in rk and g not in superior}

    def nonmem():
        return [g for g in sorted(rk, key=rk.get) if g not in cur and g not in superior]
    while len(cur) < n:                                   # restore count (relegation candidates first by rank)
        cands = [g for g in nonmem() if ok_entry(g) and rk[g] <= thr["rx"]] or nonmem()
        if not cands:
            break
        cur.add(cands[0])
    while len(cur) > n:
        cur.remove(max(cur, key=rk.get))
    for g in sorted([g for g in cur if rk[g] > thr["fx"]], key=rk.get, reverse=True):     # 1 fast exit
        cands = [h for h in nonmem() if ok_entry(h) and rk[h] <= thr["rx"]] or nonmem()
        if cands:
            cur.discard(g)
            cur.add(cands[0])
    for g in [h for h in nonmem() if rk[h] <= thr["fe"] and ok_entry(h)]:                 # 2 fast entry
        low = max(cur, key=rk.get)
        if rk[low] > rk[g]:
            cur.discard(low)
            cur.add(g)
    return cur


def apply_rules(logF: pd.Series, lvl: dict, tec: set, tech: set, ebitda_ok=lambda g: True) -> tuple[dict, set]:
    order = logF.sort_values(ascending=False).index
    rank = {g: i + 1 for i, g in enumerate(order)}
    new, superior = {}, set()
    for nm, lv in [("DAX", 3), ("MDAX", 2), ("SDAX", 1)]:
        mem = {g for g, l in lvl.items() if l == lv}
        ok = (lambda g: lvl.get(g) == 3 or ebitda_ok(g)) if nm == "DAX" else (lambda g: True)
        new[nm] = select(rank, mem, C.DAX_N[nm], C.DAX_THR[nm], ok, superior)
        superior |= new[nm]
    tl = [g for g in order if g in tech]
    rkT = {g: i + 1 for i, g in enumerate(tl)}
    newT = select(rkT, tec, C.DAX_N["TecDAX"], C.DAX_THR["TecDAX"], lambda g: True, set())
    lvl_new = {}
    for nm, lv in [("SDAX", 1), ("MDAX", 2), ("DAX", 3)]:
        for g in new[nm]:
            lvl_new[g] = lv
    return lvl_new, newT


# ------------------------------------------------------------------------------------- main
def run(main: pd.DataFrame, tec: pd.DataFrame, H: dict, P: pd.DataFrame, symmap: dict, info: pd.DataFrame,
        asof: pd.Timestamp, cutoff: pd.Timestamp, index_caps: dict, n_mc: int = 2000, seed: int = 11,
        members_override: dict | None = None, adjustments: pd.DataFrame | None = None) -> dict:
    import universe as U
    asof, cutoff = pd.Timestamp(asof), pd.Timestamp(cutoff)
    W, ccy = lib.price_panel(P, "close")
    sessions = W.index[W.notna().sum(axis=1) > 50]
    list_date = pd.Timestamp(main.list_date.iloc[0])
    C0 = list_cutoff(list_date, sessions)
    t = sessions[sessions <= asof][-1]
    u = U.dax_universe(main, tec, H)
    adj_notes = []
    if adjustments is not None and len(adjustments):
        # between-review changes announced after the list (stoxx.com/feed/), effective on or before the cut-off
        lvl_of = {"DAX": 3, "MDAX": 2, "SDAX": 1}
        for a in adjustments.to_dict("records"):
            if pd.Timestamp(a["effective"]) > cutoff or pd.Timestamp(a["effective"]) <= list_date:
                continue
            m = u.ISIN == a["isin"]
            if a["index"] in lvl_of:
                u.loc[m, "lvl"] = lvl_of[a["index"]] if a["side"] == "add" else 0
            elif a["index"] == "TECDAX":
                u.loc[m, "tec"] = a["side"] == "add"
            if int(a["exclude_from_ranking"]):
                u.loc[m, "rank"] = np.nan
            adj_notes.append(f"{a['index']} {a['side']} {a['name']} (eff. {a['effective']})")
    if members_override:                     # calibration: membership at another date
        u["lvl"] = u.ISIN.map(members_override["lvl"]).fillna(0).astype(int)
        u["tec"] = u.ISIN.isin(members_override["tec"])
    u["symbol"] = u.symbol_hold.where(u.symbol_hold.notna(), u.ISIN.map(symmap))
    alt = u.ISIN.map(symmap)
    have = set(W.columns)
    u["symbol"] = [s if s in have else (a if a in have else s) for s, a in zip(u.symbol, alt)]
    v0 = lib.vwap20(P, C0)
    vt = lib.vwap20(P, t)
    u["vwap_C0"] = u.symbol.map(v0)
    u["vwap_t"] = u.symbol.map(vt)
    u["close_t"] = u.symbol.map(lib.asof_value(W, t))
    # fund k: AUM / index cap (index caps from R2 MV x level ratio); index shares F = H / k
    ks = {}
    for etf, idx in [("EXS1", "DAX"), ("EXS3", "MDAX"), ("EXS2", "TECDAX")]:
        M = index_caps.get(idx)
        ks[etf] = float(H[etf].mv.sum()) / M if M else np.nan
    # TecDAX caps SAP/Infineon (and maybe others): prefer EXS1 / EXS3 index shares, EXS2 only for TecDAX-only names
    F = u.h_EXS1 / ks["EXS1"]
    F = F.where(F.notna(), u.h_EXS3 / ks["EXS3"])
    F = F.where(F.notna(), u.h_EXS2 / ks["EXS2"])
    u["F_index"] = F
    # cross-check fund scales on overlapping, uncapped names (EXS3 & EXS2)
    ov = u[u.h_EXS3.notna() & u.h_EXS2.notna()]
    scale_23 = float(np.median((ov.h_EXS2 / ks["EXS2"]) / (ov.h_EXS3 / ks["EXS3"]))) if len(ov) else np.nan
    u["ffmc_anchor_C0"] = u.F_index * u.vwap_C0
    # Yahoo float estimates (weaker points) for names without a fund
    inf = info.set_index("symbol") if len(info) else pd.DataFrame()
    fs = u.symbol.map(inf.floatShares) if "floatShares" in inf else pd.Series(np.nan, index=u.index)
    so = u.symbol.map(inf.sharesOutstanding) if "sharesOutstanding" in inf else pd.Series(np.nan, index=u.index)
    fs = pd.to_numeric(fs, errors="coerce")
    so = pd.to_numeric(so, errors="coerce")
    u["yf_float_shares"] = fs.where(fs > 0, so)
    u["ffmc_yf_C0"] = u.yf_float_shares * u.vwap_C0
    both = u.ffmc_anchor_C0.gt(0) & u.ffmc_yf_C0.gt(0)
    s_yf = float(np.median(u.loc[both, "ffmc_anchor_C0"] / u.loc[both, "ffmc_yf_C0"])) if both.sum() >= 5 else 1.0
    est = u.ffmc_anchor_C0.where(u.ffmc_anchor_C0 > 0, s_yf * u.ffmc_yf_C0)
    wts = np.where(u.ffmc_anchor_C0 > 0, 1.0, 0.3)
    ranked = u["rank"].notna() & est.gt(0)
    g = lib.rank_curve(u.loc[ranked, "rank"].values.astype(float), np.log(est[ranked].values), wts[ranked.values])
    anc = u.ffmc_anchor_C0.gt(0) & u["rank"].notna()
    rho = u.loc[anc, "ffmc_anchor_C0"].rank().corr((-u.loc[anc, "rank"]).rank())
    u["logF_C0"] = np.where(u["rank"].notna(), g(u["rank"].fillna(0).values.astype(float)), np.nan)
    ret = u.vwap_t / u.vwap_C0
    med = float(np.nanmedian(ret[u["rank"].between(30, 210)])) if np.isfinite(ret[u["rank"].between(30, 210)]).any() else 1.0
    u["flags"] = ""
    u.loc[~np.isfinite(ret), "flags"] += "no_price;"
    u.loc[u.symbol.map(lib.stale(W, t, P=P)).fillna(False).astype(bool), "flags"] += "stale_price;"
    u.loc[~u.ffmc_anchor_C0.gt(0) & u["rank"].notna(), "flags"] += "no_fund_anchor;"
    u["logF_t"] = u.logF_C0 + np.log(ret.where(np.isfinite(ret), med))
    u["F_implied"] = np.exp(u.logF_C0) / u.vwap_C0
    u["F_est"] = u.F_index.where(u.F_index.notna(), u.F_implied)
    # eligible companies = ranked on the list (members unranked: basic-criteria breach -> removed at the review)
    E = u[u["rank"].notna() & np.isfinite(u.logF_t)].copy().set_index("ISIN")
    lvl = {g: l for g, l in u.set_index("ISIN").lvl.items() if l > 0}
    tecset = set(u.loc[u.tec, "ISIN"])
    tech = set(u.loc[u.tech_listed & (u.rank_tec.notna() | u.tec), "ISIN"])
    unranked_members = u.loc[(u.lvl > 0) & u["rank"].isna(), "Instrument_Name"].tolist()
    lvl_new, tec_new = apply_rules(E.logF_t, lvl, tecset, tech)
    order = E.logF_t.sort_values(ascending=False)
    E["rank_t"] = pd.Series(np.arange(1, len(order) + 1), index=order.index)
    tl = [g for g in order.index if g in tech]
    E["rank_tec_t"] = pd.Series(np.arange(1, len(tl) + 1), index=tl)
    E["lvl_cur"] = E.index.map(lambda g: lvl.get(g, 0))
    E["lvl_pred"] = E.index.map(lambda g: lvl_new.get(g, 0))
    E["tec_cur"] = E.index.isin(tecset)
    E["tec_pred"] = E.index.isin(tec_new)
    # volatility with VWAP lock-in
    Wde = W[[c for c in W.columns if c in set(E.symbol.dropna())]]
    sig = lib.rel_sigma(Wde, t, E.loc[E.rank_t <= 220, "symbol"].dropna().tolist())
    E["sigma_rel"] = E.symbol.map(sig).fillna(sig.median())
    h = C.trading_days_between(t, cutoff, "XETR")
    heff = lib.h_eff_vwap(h)
    E["h_days"], E["h_eff"] = h, heff
    E["sigma_data"] = np.where(E["flags"].str.contains("no_price|stale_price"), C.SIGMA_DATA["flagged"],
                               np.where(E["flags"].str.contains("no_fund_anchor"), C.SIGMA_DATA["nonmember"],
                                        C.SIGMA_DATA["member"]))
    E["sigma_tot"] = lib.sigma_total(E.sigma_rel, heff, E.sigma_data)
    # rule lines at t (log FFMC between rank r and r+1)
    v = order.values
    vT = E.loc[tl, "logF_t"].values

    def line(r, arr=v):
        return (arr[r - 1] + arr[r]) / 2 if len(arr) > r else arr[-1]
    lines = {"DAX_fx": line(60), "DAX_fe": line(33), "MDAX_fx": line(110), "MDAX_fe": line(83),
             "SDAX_fx": line(180), "SDAX_fe": line(153), "TecDAX_fx": line(45, vT), "TecDAX_fe": line(25, vT)}
    # primary moves with margin-based probabilities
    rows = []
    for g, r in E.iterrows():
        l0 = r.lvl_cur
        cand = []
        if l0 == 3:
            cand.append(("DAX->MDAX (fast exit)", "DAX_fx", -1))
        else:
            cand.append((f"{LV_NAME[l0]}->DAX (fast entry)", "DAX_fe", +1))
        if l0 == 2:
            cand.append(("MDAX->SDAX (fast exit)", "MDAX_fx", -1))
        elif l0 < 2:
            cand.append((f"{LV_NAME[l0]}->MDAX (fast entry)", "MDAX_fe", +1))
        if l0 == 1:
            cand.append(("SDAX->none (fast exit)", "SDAX_fx", -1))
        elif l0 == 0:
            cand.append(("none->SDAX (fast entry)", "SDAX_fe", +1))
        if g in tech:
            cand.append(("TecDAX out (fast exit)", "TecDAX_fx", -1) if r.tec_cur else ("TecDAX in (fast entry)", "TecDAX_fe", +1))
        for lab, ln, sgn in cand:
            m = r.logF_t - lines[ln]                                 # + = above the line
            p_above = float(lib.p_cross(m, r.sigma_tot))
            p = p_above if sgn > 0 else 1 - p_above
            if p >= 0.01:
                rows.append(dict(ISIN=g, move=lab, line=ln, margin_log=m, p_margin=p, primary=True))
    MV = pd.DataFrame(rows)
    # deterministic predicted moves at t (incl. forced / cascade moves)
    det = []
    for g, r in E.iterrows():
        if r.lvl_pred != r.lvl_cur:
            det.append((g, f"{LV_NAME[r.lvl_cur]}->{LV_NAME[r.lvl_pred]}"))
        if r.tec_pred != r.tec_cur:
            det.append((g, "TecDAX in" if r.tec_pred else "TecDAX out"))
    for g in set(lvl) - set(E.index):                     # unranked members drop out
        det.append((g, f"{LV_NAME[lvl[g]]}->none (unranked)"))
    # Monte Carlo of the full cascade
    mc_counts = {}
    rng = np.random.default_rng(seed)
    base = E.logF_t
    sd = E.sigma_tot
    for _ in range(n_mc):
        x = base + sd * rng.standard_normal(len(base))
        ln_, tn_ = apply_rules(x, lvl, tecset, tech)
        for g in E.index:
            a, b = lvl.get(g, 0), ln_.get(g, 0)
            if a != b:
                k = (g, f"{LV_NAME[a]}->{LV_NAME[b]}")
                mc_counts[k] = mc_counts.get(k, 0) + 1
            ta, tb = g in tecset, g in tn_
            if ta != tb:
                k = (g, "TecDAX in" if tb else "TecDAX out")
                mc_counts[k] = mc_counts.get(k, 0) + 1
    MC = pd.DataFrame([dict(ISIN=g, move_simple=m, p_mc=c / n_mc) for (g, m), c in mc_counts.items()])
    meta = dict(list_date=str(list_date.date()), C0=str(C0.date()), t=str(t.date()), h_days=h, h_eff=heff,
                adjustments_applied=adj_notes,
                anchors=int(anc.sum()), anchor_rank_spearman=float(rho), yf_scale=s_yf, k=ks,
                fund_scale_EXS2_vs_EXS3=scale_23, lines_eur={k: float(np.exp(v)) for k, v in lines.items()},
                unranked_members=unranked_members, n_ranked=int(len(E)),
                counts_list={k: int((u.lvl == v).sum()) for k, v in LV.items()} | {"TecDAX": int(u.tec.sum())},
                counts_funds={"EXS1": len(H["EXS1"]), "EXS3": len(H["EXS3"]), "EXS2": len(H["EXS2"])},
                max_weight={"DAX": float(H["EXS1"].weight.max()), "MDAX": float(H["EXS3"].weight.max()),
                            "TecDAX": float(H["EXS2"].weight.max())})
    # SDAX weights from estimated FFMC (no free fund file)
    sd_m = E[E.lvl_cur == 1]
    if len(sd_m):
        w = np.exp(sd_m.logF_t) / np.exp(sd_m.logF_t).sum()
        meta["max_weight"]["SDAX_est"] = float(w.max())
    return dict(E=E.reset_index(), U=u, MV=MV, MC=MC, det=det, lines=lines, meta=meta)
