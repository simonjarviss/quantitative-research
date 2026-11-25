"""Calibration replay: run this free-data pipeline on the June 2026 reviews (STOXX Europe 600, DAX family, FTSE UK)
and score it against the actual changes (paper only).

Two replays per family, mirroring the live runs:
  A  cut-off run   STOXX/DAX: the May selection list (Rank FINAL at the 30 Apr cut-off) rolled forward with actual
                   prices to the 29 May cut-off (h = 0, the 30 Nov analogue). FTSE: full caps at the 2 Jun cut-off.
  B  7-week run    STOXX/DAX: the April list (31 Mar cut-off) rolled to the 13 Apr close, probabilities to 29 May
                   (the 12 Oct -> 30 Nov analogue). FTSE: 14 Apr close, probabilities to 2 Jun.
  oracle           STOXX/DAX: the rule applied directly to the official June list ranks (cut-off 29 May): what is
                   left is STOXX's own float/share updates, corporate events and discretion.
Pre-review membership and index shares: iShares holdings dated 18 Jun 2026 (16_flow_engine cache, read-only; copied to
raw/calib_2026-06/), the DAX lists of 2 Jun (SDAX), R2 Datastream FTSE lists of 2 Jun. Actual changes: EXSA/EXS*
holdings 18 -> 22 Jun, DAX lists 2 Jun -> 1 Jul, R2 FTSE lists 19 -> 22 Jun.

Look-ahead and other biases (stated in the output):
  - symbols: today's Yahoo symbol map, so names delisted since June have no prices and drop out (survivorship);
  - STOXX/DAX anchors use index shares from 18 Jun holdings (after the 22 May free-float freeze, i.e. ~3 weeks of
    share updates the 29 May ranking could not see) - small;
  - FTSE shares: Yahoo's CURRENT shares outstanding (look-ahead of buybacks/issuance since June) where Yahoo has the
    name, else R2 Datastream numshrs at 2 Jun (point in time); a fully point-in-time variant uses Datastream market cap;
  - FTSE June is the annual review (liquidity and voting tests; 0.15%/0.10% SmallCap thresholds): extra deletions the
    quarterly model cannot see;
  - DAX: EBITDA assumed positive; the June review is Fast-only, like December.
Usage: python3 validate_june2026.py --fetch-date 2026-10-12
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as Dm
import fetch as F
import lib
import model_dax as MD
import model_ftse as MF
import model_stoxx as MS
import universe as U

CAL = C.RAW / "calib_2026-06"


def june_holdings() -> dict:
    """iShares holdings 18 and 22 Jun 2026 from the engine cache (read-only), copied to raw/calib_2026-06/."""
    src = C.ENGINE / "data" / "holdings"
    H = F._holdings_module(CAL / "holdings")
    out = {}
    for etf in ["EXSA", "EXS1", "EXS3", "EXS2", "ISF", "MIDD"]:
        for d in ["20260618", "20260622"]:
            s = src / etf / f"{etf}_asof{d}.json"
            dst = CAL / "holdings" / etf / s.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() and s.exists():
                shutil.copyfile(s, dst)
            out[(etf, d)] = F.parse_holdings(H, etf, dst)
    return out


def prf(pred: set, act: set) -> dict:
    hit = len(pred & act)
    return dict(n_pred=len(pred), n_actual=len(act), n_hit=hit, precision=hit / len(pred) if pred else np.nan,
                recall=hit / len(act) if act else np.nan)


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y.astype(float)) ** 2).mean()) if len(p) else np.nan


# ------------------------------------------------------------------------------------------ STOXX
def stoxx(D, HJ, caps_jun):
    """Truth = EXSA holdings 18 Jun (pre-review) -> 22 Jun (post-review), keyed by ISIN. The 1 Jun 2026 public list
    (review month) has an empty "Index Membership" column, so list-based truth (1 Jun vs 1 Jul) is not available; the
    May -> July list difference is shown as a cross-check (it also contains between-review M&A changes).
    Predicted adds (list RICs) are mapped to ISINs through the 1 Jul list matched to the 22 Jun holdings."""
    files, P = D["files"], D["prices"]
    sym = dict(zip(D["lines"].key, D["lines"].symbol))
    pre, post = HJ[("EXSA", "20260618")], HJ[("EXSA", "20260622")]
    may, jul = F.read_list(files[("sxxp", "20260501")]), F.read_list(files[("sxxp", "20260701")])
    act_add, act_del = set(post["isin"]) - set(pre["isin"]), set(pre["isin"]) - set(post["isin"])
    names = dict(zip(pre["isin"], pre["name"])) | dict(zip(post["isin"], post["name"]))
    uj, _ = U.stoxx_universe(jul, post)
    ric2isin = dict(zip(uj.loc[uj.exsa_isin.notna(), "RIC"], uj.loc[uj.exsa_isin.notna(), "exsa_isin"]))
    mem = lambda x: set(x.loc[x.member_code.isin(["Large", "Mid", "Small"]), "RIC"])
    lnames = dict(zip(may.RIC, may.Instrument_Name)) | dict(zip(jul.RIC, jul.Instrument_Name))
    list_add = sorted(lnames[k] for k in mem(jul) - mem(may))
    list_del = sorted(lnames[k] for k in mem(may) - mem(jul))
    res = {}
    for lab, ld, asof in [("A cut-off (May list -> 29 May)", "20260501", "2026-05-29"),
                          ("B 7 weeks (Apr list -> 13 Apr)", "20260401", "2026-04-13"),
                          ("oracle (June list ranks)", "20260601", "2026-05-29")]:
        sl = F.read_list(files[("sxxp", ld)])
        if not sl.member_code.isin(["Large", "Mid", "Small"]).any():     # 1 Jun list: membership blank
            sl = Dm.fill_members(sl, may)
        r = MS.run(sl, pre, P, sym, pd.Timestamp(asof), pd.Timestamp("2026-05-29"), caps_jun.get("STOXX600"), n_mc=2000)
        E = r["E"].copy()
        E["ik"] = [i if isinstance(i, str) else ric2isin.get(k, "RIC:" + str(k)) for i, k in zip(E.exsa_isin, E.key)]
        nm_e = dict(zip(E.ik, E.Instrument_Name))
        nmf = lambda k: names.get(k) or nm_e.get(k, k)
        pa, pdl = set(E.loc[E.pred == "add", "ik"]), set(E.loc[E.pred == "delete", "ik"])
        nm, mm = E[~E.member_now], E[E.member_now]
        res[lab] = dict(add=prf(pa, act_add), delete=prf(pdl, act_del),
                        brier_add=brier(nm.p_change, nm.ik.isin(act_add)),
                        brier_del=brier(mm.p_change, mm.ik.isin(act_del)),
                        brier_add_mc=brier(nm.p_mc_change, nm.ik.isin(act_add)),
                        brier_del_mc=brier(mm.p_mc_change, mm.ik.isin(act_del)),
                        sum_p_add=float(nm.p_change.sum()), sum_p_del=float(mm.p_change.sum()),
                        sum_p_add_mc=float(nm.p_mc_change.sum()), sum_p_del_mc=float(mm.p_mc_change.sum()),
                        p_actual_del=float(mm.loc[mm.ik.isin(act_del), "p_change"].sum()),
                        p_actual_del_mc=float(mm.loc[mm.ik.isin(act_del), "p_mc_change"].sum()),
                        p_actual_add=float(nm.loc[nm.ik.isin(act_add), "p_change"].sum()),
                        p_actual_add_mc=float(nm.loc[nm.ik.isin(act_add), "p_mc_change"].sum()),
                        actual_adds_not_ranked=len(act_add - set(E.ik)), meta=r["meta"],
                        false_adds=[nmf(k) for k in sorted(pa - act_add)],
                        missed_adds=[nmf(k) for k in sorted(act_add - pa)],
                        false_dels=[nmf(k) for k in sorted(pdl - act_del)],
                        missed_dels=[nmf(k) for k in sorted(act_del - pdl)])
    return res, dict(adds=sorted(names[k] for k in act_add), deletes=sorted(names[k] for k in act_del),
                     list_adds=list_add, list_dels=list_del)


# ------------------------------------------------------------------------------------------ DAX
def dax(D, HJ, caps_jun):
    files, P = D["files"], D["prices"]
    sym = dict(zip(D["lines"].key, D["lines"].symbol))
    pre_m = F.read_list(files[("mkdx", "20260602")])
    pre_t = F.read_list(files[("tdxk", "20260602")])
    post_m = F.read_list(files[("mkdx", "20260701")])
    post_t = F.read_list(files[("tdxk", "20260701")])
    lv = {"DAXK": 3, "MKDX": 2, "SDXK": 1}
    lvl_pre = {g: lv[c] for g, c in zip(pre_m.ISIN, pre_m.member_code) if c in lv}
    lvl_post = {g: lv[c] for g, c in zip(post_m.ISIN, post_m.member_code) if c in lv}
    tec_pre = set(pre_t.loc[pre_t.member_code == "TDXK", "ISIN"])
    tec_post = set(post_t.loc[post_t.member_code == "TDXK", "ISIN"])
    act = set()
    for g in set(lvl_pre) | set(lvl_post):
        a, b = lvl_pre.get(g, 0), lvl_post.get(g, 0)
        if a != b:
            act.add((g, f"{MD.LV_NAME[a]}->{MD.LV_NAME[b]}"))
    for g in tec_pre ^ tec_post:
        act.add((g, "TecDAX in" if g in tec_post else "TecDAX out"))
    H = {"EXS1": HJ[("EXS1", "20260618")], "EXS3": HJ[("EXS3", "20260618")], "EXS2": HJ[("EXS2", "20260618")]}
    res = {}
    for lab, ld, asof in [("A cut-off (May list -> 29 May)", "20260505", "2026-05-29"),
                          ("B 7 weeks (Apr list -> 13 Apr)", "20260401", "2026-04-13"),
                          ("oracle (June list ranks)", "20260602", "2026-05-29")]:
        main = F.read_list(files[("mkdx", ld)])
        tec = F.read_list(files[("tdxk", ld)])
        r = MD.run(main, tec, H, P, sym, D["info"], pd.Timestamp(asof), pd.Timestamp("2026-05-29"), caps_jun,
                   n_mc=600, members_override=dict(lvl=lvl_pre, tec=tec_pre))
        det = {(g, m.replace(" (unranked)", "")) for g, m in r["det"]}
        up = {x for x in det if MD_up(x[1])}
        dn = det - up
        aup = {x for x in act if MD_up(x[1])}
        adn = act - aup
        mc = {(x.ISIN, x.move_simple): x.p_mc for x in r["MC"].itertuples()} if len(r["MC"]) else {}
        cand = set(mc) | act
        p = pd.Series({k: mc.get(k, 0.0) for k in cand})
        y = pd.Series({k: k in act for k in cand})
        res[lab] = dict(up=prf(up, aup), down=prf(dn, adn), all=prf(det, act), brier_mc=brier(p, y),
                        sum_p_mc=float(p.sum()), meta=r["meta"],
                        false=[f"{name_of(r, g)} {m}" for g, m in det - act],
                        missed=[f"{name_of(r, g, pre_m)} {m}" for g, m in act - det])
    return res, sorted(f"{name_of(None, g, pre_m)} {m}" for g, m in act)


def MD_up(m: str) -> bool:
    if m.startswith("TecDAX"):
        return m.endswith("in")
    a, b = m.split("->")
    o = {"none": 0, "SDAX": 1, "MDAX": 2, "DAX": 3}
    return o[b] > o[a]


def name_of(r, g, lst=None):
    if r is not None:
        x = r["U"].loc[r["U"].ISIN == g, "Instrument_Name"]
        if len(x):
            return x.iloc[0]
    if lst is not None:
        x = lst.loc[lst.ISIN == g, "Instrument_Name"]
        if len(x):
            return x.iloc[0]
    return g


# ------------------------------------------------------------------------------------------ FTSE
def ftse(D, HJ, caps_jun):
    r2 = D["r2"]
    L, S = r2["lists"], r2["shares"]
    lines = D["lines"]
    sym = dict(zip(lines.key, lines.symbol))
    fu_now = D["ftse_u"]

    def members(d):
        d = pd.Timestamp(d)
        x = L[(L.startdate <= d) & (L.enddate >= d)]
        return {c: set(x.loc[x.list == c, "infocode"]) for c in (3670, 3671, 3455)}
    pre, post = members("2026-06-19"), members("2026-06-22")
    m0 = members("2026-06-02")
    d0 = pd.Timestamp("2026-06-02")
    x = L[(L.startdate <= d0) & (L.enddate >= d0) & (L.list == 3455)].drop_duplicates("infocode").copy()
    s = S[S.marketdate == S[S.marketdate <= d0].marketdate.max()].set_index("infocode")
    x["r2_mktcap"] = x.infocode.map(s.mktcap)
    x["r2_numshrs_jun"] = x.infocode.map(s.numshrs)
    x["cur"] = np.where(x.infocode.isin(m0[3670]), "FTSE100", np.where(x.infocode.isin(m0[3671]), "FTSE250", "SmallCap"))
    # symbol: by ISIN key (data.prepare searched June names), else by name against today's universe
    nm = dict(zip([U.norm_name(n) for n in fu_now["name"]], fu_now.symbol))
    def find(r):
        s_ = sym.get(r.isin)
        if isinstance(s_, str):
            return s_
        k = U.norm_name(r.dsqtname)
        best = max(nm, key=lambda q: U.difflib.SequenceMatcher(None, k, q).ratio()) if nm else None
        return nm[best] if best and U.difflib.SequenceMatcher(None, k, best).ratio() > 0.85 else None
    x["symbol"] = [find(r) for r in x.itertuples()]
    uni = pd.DataFrame(dict(key=x.infocode.astype(str), isin=x["isin"], name=x.dsqtname, ticker=None, symbol=x.symbol,
                            cur=x.cur, src="R2 2026-06-02", ccy=x.isocurrcode, hold_shares=np.nan, hold_price=np.nan,
                            hold_weight=np.nan, r2_mktcap_jun=x.r2_mktcap, r2_numshrs_jun=x.r2_numshrs_jun))
    uni = uni[uni.r2_mktcap_jun.fillna(0) > 100e6]
    act = set()
    for g in pre[3455] | post[3455]:
        a = "FTSE100" if g in pre[3670] else "FTSE250" if g in pre[3671] else "SmallCap"
        b = "FTSE100" if g in post[3670] else "FTSE250" if g in post[3671] else "SmallCap"
        if a != b and (a != "SmallCap" or b != "SmallCap"):
            act.add((str(g), f"{a}->{b}"))
    H = {"ISF": HJ[("ISF", "20260618")], "MIDD": HJ[("MIDD", "20260618")]}
    res = {}
    ds_cap = pd.Series(uni.r2_mktcap_jun.values, index=uni.key.values)
    for lab, asof, mode in [("A cut-off (2 Jun close)", "2026-06-02", "pipeline"),
                            ("A' cut-off, point-in-time Datastream caps", "2026-06-02", "pit"),
                            ("B 7 weeks (14 Apr close)", "2026-04-14", "pipeline")]:
        so = None
        if mode == "pit":
            # shares such that Yahoo price x shares = Datastream market cap on 2 Jun
            W, _ = lib.price_panel(D["prices"])
            FXg = lib.fx_to(D["prices"], "GBP")
            px = uni.symbol.map(lib.asof_value(W, pd.Timestamp(asof)))
            so = pd.Series((ds_cap.values / px.values), index=uni.key.values)
        r = MF.run(uni, D["prices"], sym, D["info"], pd.DataFrame(), pd.Timestamp(asof), pd.Timestamp("2026-06-02"),
                   caps_jun, H, n_mc=600, members_override=dict(universe=uni), shares_override=so)
        det = set(r["det"])
        E = r["E"]
        visible = set(E.key)
        act_v = {a for a in act if a[0] in visible}
        mc = {(x.key, x.move_simple): x.p_mc for x in r["MC"].itertuples()} if len(r["MC"]) else {}
        cand = set(mc) | act_v
        p = pd.Series({k: mc.get(k, 0.0) for k in cand})
        y = pd.Series({k: k in act_v for k in cand})
        nm_of = dict(zip(E.key, E["name"]))
        res[lab] = dict(all=prf(det, act), visible=prf(det, act_v), brier_mc=brier(p, y), sum_p_mc=float(p.sum()),
                        n_ranked=int(len(E)), meta=r["meta"],
                        false=[f"{nm_of.get(g, g)} {m}" for g, m in det - act],
                        missed=[f"{nm_of.get(g, x.set_index(x.infocode.astype(str)).dsqtname.get(g, g))} {m}" for g, m in act - det])
    nmap = dict(zip(x.infocode.astype(str), x.dsqtname))
    return res, sorted(f"{nmap.get(g, g)} {m}" for g, m in act)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-date", required=True)
    a = ap.parse_args()
    run_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    D = Dm.prepare(a.fetch_date, pd.Timestamp(a.fetch_date), offline=True)
    HJ = june_holdings()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(C.ENGINE))
    import pi as EP
    caps_jun = {}
    for idx in ["STOXX600", "DAX", "MDAX", "TECDAX", "FTSE100", "FTSE250"]:
        mv, d = EP.index_mv(idx, pd.Timestamp("2026-06-18"))
        caps_jun[idx] = mv
    out = {}
    out["stoxx600"], act_s = stoxx(D, HJ, caps_jun)
    out["dax"], act_d = dax(D, HJ, caps_jun)
    out["ftse_uk"], act_f = ftse(D, HJ, caps_jun)
    md = render(out, act_s, act_d, act_f)
    p = C.SCORES / "validate_june2026.md"
    lib.archive_old(p)
    p.write_text(md)
    pj = C.SCORES / "validate_june2026.json"
    lib.archive_old(pj)
    pj.write_text(json.dumps(out, indent=1, default=str))
    print(md)
    lib.append_log([f"\n## Calibration replay {run_utc:%Y-%m-%d %H:%M:%S} UTC — June 2026 reviews (validate_june2026.py)", "",
                    f"- Command: `python3 validate_june2026.py --fetch-date {a.fetch_date}`. Report `scores/validate_june2026.md`.",
                    "", "| File | SHA-256 |", "|---|---|",
                    f"| `scores/{p.name}` | `{lib.sha(p)}` |", f"| `scores/{pj.name}` | `{lib.sha(pj)}` |"])


def fmt(d):
    return f"{d['n_hit']}/{d['n_pred']} predicted, {d['n_actual']} actual: precision {d['precision']:.0%}, recall {d['recall']:.0%}" \
        if d["n_pred"] or d["n_actual"] else "none predicted, none actual"


def render(out, act_s, act_d, act_f) -> str:
    L = ["# June 2026 calibration replay of the free-data pipeline (not a live prediction)", "",
         f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `validate_june2026.py`. Biases: see the "
         "script docstring (today's symbol map = survivorship; 18 Jun index shares; FTSE current Yahoo shares; FTSE June "
         "is the annual review with liquidity and voting tests).", "",
         "## STOXX Europe 600 (June 2026 review: actual changes = EXSA holdings 18 -> 22 Jun, by ISIN)", "",
         f"Actual adds ({len(act_s['adds'])}): {', '.join(act_s['adds'])}.", "",
         f"Actual deletes ({len(act_s['deletes'])}): {', '.join(act_s['deletes'])}.", "",
         f"Cross-check, membership on the 1 May vs 1 Jul selection lists (includes between-review M&A changes; the 1 Jun "
         f"list has no membership column): adds {', '.join(act_s['list_adds'])}; deletes {', '.join(act_s['list_dels'])}.", ""]
    for lab, r in out["stoxx600"].items():
        L += [f"**{lab}**", "", f"- Adds: {fmt(r['add'])}. Deletes: {fmt(r['delete'])}.",
              f"- Brier (adds / deletes over all ranked lines): {r['brier_add']:.4f} / {r['brier_del']:.4f}; "
              f"sum of P(add) {r['sum_p_add']:.1f}, sum of P(delete) {r['sum_p_del']:.1f}; actual adds not on the list's "
              f"ranked lines: {r['actual_adds_not_ranked']}.",
              f"- Joint Monte Carlo (p_mc, includes buffer crowding): Brier adds / deletes {r['brier_add_mc']:.4f} / "
              f"{r['brier_del_mc']:.4f}; sum of p_mc(add) {r['sum_p_add_mc']:.1f}, p_mc(delete) {r['sum_p_del_mc']:.1f}. "
              f"Probability mass on the names that actually changed: adds {r['p_actual_add']:.1f} (margin) / "
              f"{r['p_actual_add_mc']:.1f} (MC); deletes {r['p_actual_del']:.1f} / {r['p_actual_del_mc']:.1f}.",
              f"- False adds: {', '.join(r['false_adds']) or 'none'}. Missed adds: {', '.join(r['missed_adds']) or 'none'}.",
              f"- False deletes: {', '.join(r['false_dels']) or 'none'}. Missed deletes: {', '.join(r['missed_dels']) or 'none'}.", ""]
    L += ["## DAX family (June 2026 review, Fast rules only: actual moves from the 2 Jun and 1 Jul selection lists)", "",
          f"Actual moves ({len(act_d)}): {'; '.join(act_d)}.", ""]
    for lab, r in out["dax"].items():
        L += [f"**{lab}**", "", f"- Promotions: {fmt(r['up'])}. Demotions: {fmt(r['down'])}. All moves: {fmt(r['all'])}.",
              f"- Monte Carlo Brier {r['brier_mc']:.4f}; sum of p_mc {r['sum_p_mc']:.1f}.",
              f"- False: {'; '.join(r['false']) or 'none'}. Missed: {'; '.join(r['missed']) or 'none'}.", ""]
    L += ["## FTSE 100 / 250 (June 2026 annual review: actual moves from R2 Datastream lists 19 -> 22 Jun)", "",
          f"Actual moves ({len(act_f)}): {'; '.join(act_f)}.", ""]
    for lab, r in out["ftse_uk"].items():
        L += [f"**{lab}**", "", f"- All moves: {fmt(r['all'])}; among names we could price: {fmt(r['visible'])}.",
              f"- Monte Carlo Brier {r['brier_mc']:.4f}; sum of p_mc {r['sum_p_mc']:.1f}; companies ranked {r['n_ranked']}.",
              f"- False: {'; '.join(r['false']) or 'none'}. Missed: {'; '.join(r['missed']) or 'none'}.", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
