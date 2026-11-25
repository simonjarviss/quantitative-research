"""The MSCI size-segment model: cutoff reconstruction, the Standard <-> Small Cap / IMI assignment rules, and
probabilities. Rules from 11_methodology/specs/msci/ (gimi = gimi_size_segments.md):

  buffers on FULL company cap: Standard member stays while cap >= 2/3 C; Small Cap member moves up at >= 1.5 C;
      a company outside the IMI enters Standard at >= C                                   [gimi §3.2 step 6; §3.1.5.1]
  final float requirements (security float cap): new / migrating >= 0.5 C, existing >= 2/3 of that; FIF < 0.15
      needs 1.8x; a Standard member failing in the lower buffer goes to Small Cap, failing above it is deleted
                                                                                   [gimi §2.3 table; §3.1.6.2]
  liquidity (DM): new 12m ATVR >= 20%, 3m ATVR >= 20%, 3m FoT >= 90%; existing 12m > 13.3%, 3m >= 5%, FoT >= 80%
                                                                                         [gimi §2.2.5; §3.1.2.4]
  EPI: additions with FIF < 0.75 breaching any excess-return window are not added           [gimi §2.3.6.3]
  seasoning 3 months [gimi §2.2.7]; cutoffs bounded to 0.5-1.15 x GMSR [gimi §2.3.2]

C per market (Canada; DM Europe as ONE market, eu §0) is not published. It is reconstructed from the previous
review's actual changes at that review's price-cutoff prices (gimi §9 "Market Size-Segment Cutoffs" row;
ca §3), then moved to the as-of date by the price change of the Standard members around the cutoff (tag I).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C


# --------------------------------------------------------------------------------------------- bracket
def bracket(Tc: pd.DataFrame, std_pre: set, std_post: set, sc_post: set, gmsr: float) -> dict:
    """Bounds on the market cutoff C at a past review from its changes.

    Tc: company table at that review's price cutoff (full_cap_usd, fif, seg as of the cutoff, isins).
    std_pre / std_post / sc_post: ISIN sets of Standard members before / after, Small Cap after.
    """
    T = Tc.copy()
    isets = T.isins.fillna("").str.split(";").apply(set)
    T["pre_std"] = [bool(s & std_pre) for s in isets]
    T["post_std"] = [bool(s & std_post) for s in isets]
    T["post_sc"] = [bool(s & sc_post) for s in isets]
    extra = T["flags"].fillna("").str.extract(r"unlisted_or_other_classes\+(\d+)%")[0].astype(float).fillna(0)
    # companies whose cap rests on a large Datastream company total (dual-listed groups, LP units, CDIs) or on no
    # mapping at all are left out of the bounds (e.g. Carnival plc held as UK Small Cap with the Carnival group total)
    T["clean"] = (~T["flags"].fillna("").str.contains("unmapped|no_full_cap|full_cap_from_DS") & T.full_cap_usd.notna()
                  & (extra < 30))
    big_float = T.fif.fillna(0) >= 0.6            # float test cannot have bound -> a size decision
    down = T[T.pre_std & ~T.post_std & T.post_sc & T.clean]
    out_ = T[T.pre_std & ~T.post_std & ~T.post_sc]
    kept = T[T.pre_std & T.post_std & T.clean]
    up = T[~T.pre_std & T.post_std & (T.seg == "SC") & T.clean]
    new = T[~T.pre_std & T.post_std & (T.seg == "NON") & T.clean]
    sc_stay = T[(T.seg == "SC") & T.post_sc & ~T.post_std & T.clean & big_float]
    L = {"1.5 x largest size-driven down-migration": 1.5 * down[big_float.loc[down.index]].full_cap_usd.max()
         if len(down[big_float.loc[down.index]]) else np.nan,
         "largest Small Cap not moved up / 1.5": sc_stay.full_cap_usd.max() / 1.5 if len(sc_stay) else np.nan}
    U = {"1.5 x smallest retained Standard member": 1.5 * kept.full_cap_usd.min() if len(kept) else np.nan,
         "smallest direct new add": new.full_cap_usd.min() if len(new) else np.nan}
    U_up = up.full_cap_usd.min() / 1.5 if len(up) else np.nan
    lo = np.nanmax(list(L.values())) if any(np.isfinite(list(L.values()))) else 0.5 * gmsr
    hi = np.nanmin(list(U.values())) if any(np.isfinite(list(U.values()))) else 1.15 * gmsr
    note = []
    if np.isfinite(U_up):
        if U_up >= lo:
            hi = min(hi, U_up)
        else:
            note.append(f"up-migration(s) below 1.5 x the lower bound ({', '.join(up.name)}): read as NOC fill "
                        "(gimi §3.2 step 6, priority 5), not used as an upper bound")
    consistent = lo <= hi
    if not consistent:
        note.append("bounds cross: cutoff set to their geometric mean, wider sigma")
    c = float(np.sqrt(lo * hi))
    c_b = float(np.clip(c, C.GMSR_RANGE[0] * gmsr, C.GMSR_RANGE[1] * gmsr))
    if c_b != c:
        note.append(f"bounded to the GMSR range ({C.GMSR_RANGE[0] * gmsr / 1e9:.2f}-{C.GMSR_RANGE[1] * gmsr / 1e9:.2f}bn)")
    return dict(C=c_b, lo=float(lo), hi=float(hi), consistent=bool(consistent), L=L, U=U, U_up=U_up,
                n_down=len(down), n_out=len(out_), n_up=len(up), n_new=len(new),
                down=[f"{r.name} {r.full_cap_usd / 1e9:.2f}bn FIF {r.fif:.2f}" for r in down.itertuples()],
                out=[f"{r.name} {r.full_cap_usd / 1e9 if pd.notna(r.full_cap_usd) else np.nan:.2f}bn FIF {r.fif:.2f}" for r in out_.itertuples()],
                up=[f"{r.name} {r.full_cap_usd / 1e9:.2f}bn" for r in up.itertuples()],
                new=[f"{r.name} {r.full_cap_usd / 1e9:.2f}bn" for r in new.itertuples()],
                smallest_kept=[f"{r.name} {r.full_cap_usd / 1e9:.2f}bn" for r in kept.nsmallest(3, "full_cap_usd").itertuples()],
                largest_sc_stay=[f"{r.name} {r.full_cap_usd / 1e9:.2f}bn" for r in sc_stay.nlargest(3, "full_cap_usd").itertuples()],
                note="; ".join(note))


def scale_factor(Ta: pd.DataFrame, Tc: pd.DataFrame, c_hat: float) -> tuple[float, int]:
    """Median USD price change (cutoff date -> as-of) of Standard members with full cap in [0.5, 2.5] x C."""
    a = Ta[Ta.seg == "STD"].set_index("main_isin").main_px_usd
    c = Tc[(Tc.seg == "STD") & Tc.full_cap_usd.between(0.5 * c_hat, 2.5 * c_hat)].set_index("main_isin").main_px_usd
    j = pd.concat([a.rename("a"), c.rename("c")], axis=1, join="inner").dropna()
    r = np.log(j.a / j.c)
    return float(np.exp(r.median())), int(len(r))


# --------------------------------------------------------------------------------------------- rules
def assign(T: pd.DataFrame, c: float, sigma_c: float, h: int, mkt: str) -> pd.DataFrame:
    """Predicted segment and probabilities for every company. T needs: seg, full_cap_usd, float_usd, fif, atvr12,
    atvr3, fot3, sig_rel, seasoned, epi_breach, known_excl (non-members), flags."""
    T = T.copy()
    sd = np.where(T["flags"].fillna("").str.contains("unlisted|unmapped|multi_class|full_cap_from_DS|shares_"),
                  C.SIGMA_DATA_FLAG, C.SIGMA_DATA)
    sr = T.sig_rel.fillna(T.sig_rel.median() if T.sig_rel.notna().any() else 0.015).clip(0.005, 0.08)
    T["sigma"] = np.sqrt(sr ** 2 * max(h, 0) + sigma_c ** 2 + sd ** 2)
    cap, flt, fif = T.full_cap_usd, T.float_usd, T.fif.fillna(1.0)
    lowfif = np.where(fif < C.FIF_MIN, C.LOW_FIF_MULT, 1.0)
    # ----- size edges (log margins, + = the change happens)
    T["edge"] = np.select([T.seg == "STD", T.seg == "SC", T.seg == "NON"],
                          [C.BUF_LO * c, C.BUF_HI * c, c], np.nan)
    m_size = np.where(T.seg == "STD", np.log(T.edge / cap), np.log(cap / T.edge))
    # ----- float edges: existing Standard >= 2/3 x 0.5 C; new / up >= 0.5 C (x1.8 if FIF < 0.15)
    fl_thr = np.where(T.seg == "STD", C.FLOAT_EXIST * c, C.FLOAT_NEW * c) * lowfif
    m_float = np.log(flt / fl_thr)                                       # + = passes
    T["m_size"], T["m_float"], T["float_thr"] = m_size, m_float, fl_thr
    sig_f = np.sqrt(T.sigma ** 2 + 0.10 ** 2)                            # FIF proxy error on top (I)
    p_size = norm.cdf(m_size / T.sigma)
    p_float_pass = norm.cdf(m_float / sig_f)
    # ----- liquidity (deterministic pass/fail on R2 data; unknown -> pass with a flag)
    new_ok = (T.atvr12.fillna(1) >= C.ATVR_NEW_12M) & (T.atvr3.fillna(1) >= C.ATVR_NEW_3M) & (T.fot3.fillna(1) >= C.FOT_NEW)
    ex_ok = (T.atvr12.fillna(1) > C.ATVR_EX_12M) & (T.atvr3.fillna(1) >= C.ATVR_EX_3M) & (T.fot3.fillna(1) >= C.FOT_EX)
    T["liq_ok"] = np.where(T.seg == "NON", new_ok, ex_ok)
    epi_block = T.epi_breach.fillna(False).astype(bool) & (fif < C.EPI_FIF_EXEMPT)
    # ----- probabilities of a Standard change
    p = np.zeros(len(T))
    std = (T.seg == "STD").values
    # Standard member leaves if size falls below 2/3 C (-> Small Cap) or the float test fails (-> Small Cap in the
    # buffer, deleted above it) or existing liquidity fails
    # existing-member float test: a fail deletes only with probability FLOAT_EXIST_BITE (Aug 2026 Europe: of the
    # Standard members below C/3 on fund-implied float, only some were removed; see README) (I)
    bite = np.where(T.get("float_exempt", pd.Series(False, index=T.index)).fillna(False), 0.1, C.FLOAT_EXIST_BITE)
    p_float_keep = 1 - (1 - p_float_pass) * bite
    liq_keep = np.where(T.liq_ok, 1.0, np.where(cap >= c, 0.95, 0.5))     # large members failing = our venue data (I)
    p_stay = (1 - p_size) * p_float_keep * liq_keep
    p[std] = 1 - p_stay[std]
    sc = (T.seg == "SC").values
    p[sc] = (p_size * p_float_pass * np.where(T.liq_ok, 1.0, 0.1) * np.where(epi_block, 0.05, 1.0))[sc]
    non = (T.seg == "NON").values
    p_non = p_size * p_float_pass * np.where(T.liq_ok, 1.0, 0.1) * np.where(epi_block, 0.05, 1.0) \
        * np.where(T.seasoned.fillna(True), 1.0, 0.05) * np.where(T.known_excl.fillna(False), C.KNOWN_EXCL_FACTOR, 1.0)
    p[non] = p_non[non]
    if "nonelig" in T:
        p[T.nonelig.fillna("").astype(str).str.len().values > 0] = 0.0
    T["p_change"] = np.nan_to_num(np.clip(p, 0, 1), nan=0.0)
    # ----- category of the change
    # a Standard member failing the float test inside the lower buffer [2/3 C, C) moves to Small Cap; one failing
    # above C is deleted from the GIMI (gimi §3.1.6.2; Aug 2026: Bollore, Latour, CSG deleted at caps > C) 
    float_dom = (1 - p_float_keep) > p_size
    T["category"] = np.select(
        [std & float_dom & (cap >= c), std, sc, non],
        ["DELETE_FLOAT", "MIG_STD_TO_SC", "MIG_SC_TO_STD", "ADD_STD"], "none")
    T["risk"] = np.where(std, np.where(float_dom, "float test", "size"), "size")
    T.loc[std & ~T.liq_ok.values, "category"] = "DELETE_LIQ"
    T["why"] = [_why(r, c) for r in T.itertuples()]
    return T


def _why(r, c) -> str:
    bits = [f"cap {r.full_cap_usd / 1e9:.2f}bn vs edge {r.edge / 1e9:.2f}bn ({np.exp(r.m_size) - 1:+.0%})"
            if np.isfinite(r.m_size) else "no cap"]
    if np.isfinite(r.m_float):
        bits.append(f"float {r.float_usd / 1e9:.2f}bn vs {r.float_thr / 1e9:.2f}bn")
    if not r.liq_ok:
        bits.append(f"liquidity fails (ATVR12 {r.atvr12:.0%}, 3m {r.atvr3:.0%}, FoT {r.fot3:.0%})")
    if getattr(r, "epi_breach", False) and (r.fif if np.isfinite(r.fif) else 1) < C.EPI_FIF_EXEMPT:
        bits.append("EPI breach")
    if getattr(r, "known_excl", False):
        bits.append("known excluded (qualified on size at Aug 2026 and not added)")
    if getattr(r, "seasoned", True) is False:
        bits.append("listed < 3 months before the review")
    return "; ".join(bits)


def noc_fill(T: pd.DataFrame, c: float) -> pd.DataFrame:
    """Priority 5 of gimi §3.2 step 6 (largest Small Cap names in the upper buffer [C, 1.5 C) fill the Standard NOC).
    MSCI's NOC adjustment (§3.1.4) is not reproducible from public data: in Aug 2026 Europe lost 9 Standard names
    to size/float and filled 1 (Diploma), Canada lost 2 and filled none. Rule used (judgment, I): when expected
    deletions exceed expected additions by >= 3, the largest upper-buffer Small Cap name that passes the float test
    gets p = 0.25, the second 0.10."""
    T = T.copy()
    T["noc_fill"] = False
    dele = float(T.loc[T.seg == "STD", "p_change"].sum())
    add = float(T.loc[T.seg != "STD", "p_change"].sum())
    gap = dele - add
    if gap >= 3:
        cand = T[(T.seg == "SC") & T.full_cap_usd.between(c, C.BUF_HI * c) & (T.m_float > 0) & T.liq_ok].sort_values(
            "full_cap_usd", ascending=False)
        for (i, r), q in zip(cand.iterrows(), (0.25, 0.10)):
            if q > T.loc[i, "p_change"]:
                T.loc[i, "p_change"] = q
                T.loc[i, "noc_fill"] = True
                T.loc[i, "why"] += f"; NOC fill candidate (expected deletions - additions = {gap:.1f})"
    T.attrs["noc_target"], T.attrs["exp_std"], T.attrs["gap"] = int((T.seg == "STD").sum()), \
        float((T.seg == "STD").sum() - dele + add), gap
    return T
