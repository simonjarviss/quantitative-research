"""Validation on the Fri 19 Jun 2026 close (inside R2 coverage): build inputs, run the engine, compare with volume.

    cd 16_flow_engine
    python3 validate_2026_06_19.py inputs        # holdings diffs -> inputs/2026-06-19/*.csv, ISIN list for volume
    (cd ../.. && PYTHONPATH=. python3 index_rebalance_research/16_flow_engine/r2_pull.py vol \
        index_rebalance_research/16_flow_engine/validation/isins_2026-06-19.csv 2026-04-01 2026-06-24 jun2026)
    python3 validate_2026_06_19.py run           # engine + fit -> validation/2026-06-19_*.csv/.md

Families that traded this close and are built here (event_calendar.md; NYSE was closed for Juneteenth, so the US
quarterly traded Thu 18 Jun and shares no close with Canada or Europe):
  S&P/TSX Composite (XIC) and S&P/TSX 60 (XIU); STOXX Europe 600 (EXSA); FTSE 100 (ISF) and FTSE 250 (MIDD,
  sampled); DAX (EXS1), MDAX (EXS3), TecDAX (EXS2); SDAX only the legs implied by MDAX<->SDAX moves (I).
Not built (no pi or no free holdings): CAC/SBF 120, AEX, BEL, PSI, ISEQ, FTSE MIB, IBEX, SMI, FTSE All-World,
MSCI (no June event). The inputs are 'final' (what the trackers actually did at the close: iShares holdings dated T
already show the rebalance) - equivalent to having the providers' pro-forma files, which a desk would have.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import engine as E
import holdings as H
import inputs as I
import marketdata as M
from common import R2DIR, SHADOW_TSX, STUDIES, VALID
from pi import index_mv

T = "2026-06-19"
D0 = "2026-06-18"
FUNDS = [("XIC", "SPTSX_COMP"), ("XIU", "SPTSX_60"), ("EXSA", "STOXX600"), ("ISF", "FTSE100"),
         ("MIDD", "FTSE250"), ("EXS1", "DAX"), ("EXS3", "MDAX"), ("EXS2", "TECDAX")]
MIN_REL = {"MIDD": 0.05}            # sampled fund: only large changes are index changes
DEFAULT_MIN_REL = 0.0025


def build_inputs():
    notes = []
    built = {}
    for etf, idx in FUNDS:
        if etf == "XIU":
            continue
        mv, mvd = index_mv(idx, D0)
        x = I.from_holdings_diff(etf, D0, T, idx, T, index_mv=mv, min_rel=MIN_REL.get(etf, DEFAULT_MIN_REL))
        built[idx] = x
        I.write(x, T, idx)
        notes.append(f"{idx}: {etf} diff, index MV {mv / 1e9:,.0f}bn ({mvd.date()}), rows {len(x)} "
                     f"({x.change_type.value_counts().to_dict()})")
    # XIU: k from the XIC-implied index shares of the same names on D0 (TSX 60 members use the same S&P float)
    xic = H.parse("XIC", D0)
    xiu = H.parse("XIU", D0)
    mv, _ = index_mv("SPTSX_COMP", D0)
    kxic = xic.mv.sum() / mv
    j = xiu.merge(xic[["isin", "shares"]], on="isin", suffixes=("", "_xic"))
    k_xiu = float(np.median(j.shares / (j.shares_xic / kxic)))
    mv60 = xiu.mv.sum() / k_xiu
    x = I.from_holdings_diff("XIU", D0, T, "SPTSX_60", T, index_mv=mv60, min_rel=DEFAULT_MIN_REL)
    I.write(x, T, "SPTSX_60")
    notes.append(f"SPTSX_60: XIU diff, index MV implied from XIC overlap {mv60 / 1e9:,.0f}bn, rows {len(x)}")
    # SDAX legs implied by MDAX moves (I): MDAX deletes that are not DAX adds -> SDAX adds; MDAX adds that were not
    # in DAX -> SDAX deletes (TecDAX-only or new names would be wrong; flagged in notes)
    mdx = built["MDAX"]
    dax = built["DAX"]
    rows = []
    for r in mdx.itertuples():
        if r.change_type == "delete" and r.isin not in set(dax["isin"]):
            rows.append(dict(change_type="add", isin=r.isin, ticker=r.ticker, country=r.country, name=r.name,
                             currency=r.currency, old_index_shares=0.0, new_index_shares=r.old_index_shares,
                             price=r.price, prob=1.0, status="final",
                             source="inferred: MDAX delete -> SDAX add (DAX-family float = MDAX float)",
                             source_ts=r.source_ts, notes="SDAX leg inferred (I); no SDAX holdings file"))
        if r.change_type == "add" and r.isin not in set(dax["isin"]):
            rows.append(dict(change_type="delete", isin=r.isin, ticker=r.ticker, country=r.country, name=r.name,
                             currency=r.currency, old_index_shares=r.new_index_shares, new_index_shares=0.0,
                             price=r.price, prob=1.0, status="final",
                             source="inferred: MDAX add (not from DAX) -> SDAX delete", source_ts=r.source_ts,
                             notes="SDAX leg inferred (I); a TecDAX-only or new name would not be an SDAX delete"))
    if rows:
        I.write(pd.DataFrame(rows), T, "SDAX")
        notes.append(f"SDAX: {len(rows)} legs inferred from MDAX moves")
    # ISIN list for the volume pull: every holding of every fund on D0 or T (changes + no-change controls)
    hs = pd.concat([H.parse(etf, d)[["isin", "country"]] for etf, _ in FUNDS for d in (D0, T)])
    hs = hs.dropna(subset=["isin"]).drop_duplicates("isin").sort_values("isin")
    hs.to_csv(VALID / f"isins_{T}.csv", index=False)
    isins = set(hs["isin"])
    (VALID / f"{T}_build_notes.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))
    print(len(isins), "ISINs for the volume pull")


def r2_membership_check() -> pd.DataFrame:
    """Compare holdings-diff adds/deletes with R2 membership (Compustat; Datastream daily FTSE lists)."""
    rows = I.load(T)
    m = pd.read_parquet(R2DIR / "members_jun2026.parquet")
    m["fromdate"] = pd.to_datetime(m.fromdate)
    m["thru"] = pd.to_datetime(m.thru)
    r2_add = m[m.fromdate == "2026-06-22"].groupby("index")["isin"].apply(set).to_dict()
    r2_del = m[(m.thru >= "2026-06-18") & (m.thru <= "2026-06-21")].groupby("index")["isin"].apply(set).to_dict()
    f = pd.read_parquet(R2DIR / "ftse_lists_jun2026.parquet")
    f["startdate"] = pd.to_datetime(f.startdate)
    f["enddate"] = pd.to_datetime(f.enddate)
    for lst, idx in ((3670, "FTSE100"), (3671, "FTSE250")):
        g = f[f.list == lst]
        before = set(g[(g.startdate <= "2026-06-19") & (g.enddate >= "2026-06-19")]["isin"])
        after = set(g[(g.startdate <= "2026-06-22") & (g.enddate >= "2026-06-22")]["isin"])
        r2_add[idx], r2_del[idx] = after - before, before - after
    # names, to match where Datastream carries an older ISIN for the same company (FTSE lists)
    import re
    nm = dict(zip(f["isin"], f.dsqtname))
    nm.update(dict(zip(m["isin"], m.conm)))
    tok = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper().split()[0]) if isinstance(s, str) and s.split() else ""  # noqa
    out = []
    for idx in sorted(set(rows["index"])):
        x = rows[rows["index"] == idx]
        if idx not in r2_add and idx not in r2_del:
            continue
        o = dict(index=idx)
        for side, r2set in (("add", r2_add.get(idx, set())), ("delete", r2_del.get(idx, set()))):
            h = x[x.change_type == side]
            hi = set(h["isin"])
            hn = {tok(v) for v in h["name"]}
            both_isin = hi & r2set
            r2_rest = r2set - both_isin
            by_name = {i for i in r2_rest if tok(nm.get(i)) in hn}
            o[f"holdings_{side}s"], o[f"r2_{side}s"] = len(hi), len(r2set)
            o[f"both_{side}s_isin"], o[f"both_{side}s_isin_or_name"] = len(both_isin), len(both_isin) + len(by_name)
            o[f"r2_only_{side}s"] = ", ".join(f"{i} {nm.get(i, '')}" for i in sorted(r2_rest - by_name))[:160]
        out.append(o)
    return pd.DataFrame(out)


def sanity_float_check() -> pd.DataFrame:
    """Known adds: XIC/EXSA-implied index shares F1 vs an independent float estimate, and D_vis = pi_vis x F1."""
    rows = I.load(T)
    out = []
    # S&P/TSX June adds vs the June backfill shadow run (S&P float proxy = MiG O/S x IWF proxy)
    sp = pd.read_csv(SHADOW_TSX / "runs/2026-06/2026-05-22_R/screen_all.csv").drop_duplicates("symbol")
    fl = dict(zip(sp.symbol, sp.float_sh))
    so = dict(zip(sp.symbol, sp.so))
    for r in rows[(rows["index"] == "SPTSX_COMP") & (rows.change_type == "add")].itertuples():
        out.append(dict(index="SPTSX_COMP", ticker=r.ticker, name=r.name, F_index=r.new_index_shares,
                        F_independent=fl.get(r.ticker, np.nan), shares_out=so.get(r.ticker, np.nan),
                        independent_source="TSX shadow 2026-05-22 run: MiG O/S x IWF proxy (DS/WS/Yahoo float)"))
    # STOXX 600 adds vs Compustat shares x Datastream/Worldscope float (EU sister-study cache)
    d = STUDIES / "stoxx600_dax" / "data"
    L = pd.read_parquet(d / "panel_lines.parquet")
    z = np.load(d / "panel.npz")
    dates = pd.DatetimeIndex(z["dates"])
    i = dates.searchsorted(pd.Timestamp(D0))
    SH, AJ = z["SH"][i], z["AJ"][i]
    fm = pd.read_parquet(d / "float_monthly.parquet")
    fm = fm[fm.month == "2026-06-01"].set_index("line")
    for r in rows[(rows["index"] == "STOXX600") & (rows.change_type == "add")].itertuples():
        c = L.index[(L["isin"] == r.isin)]
        if len(c) == 0:
            continue
        c = c[0]
        sh = float(SH[c] / AJ[c]) if np.isfinite(SH[c]) else np.nan
        ff = fm.ff_ds.get(c, np.nan)
        if not np.isfinite(ff):
            ff = fm.ff_ws.get(c, np.nan)
        out.append(dict(index="STOXX600", ticker=r.ticker, name=r.name, F_index=r.new_index_shares,
                        F_independent=sh * ff if np.isfinite(ff) else np.nan, shares_out=sh,
                        independent_source="Compustat cshoc x Datastream/Worldscope free float (Jun 2026)"))
    s = pd.DataFrame(out)
    s["ratio"] = s.F_index / s.F_independent
    return s


def fit(sheet: pd.DataFrame, md: pd.DataFrame, contrib: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Predicted |net demand| vs realised abnormal trade-day volume."""
    from scipy import stats
    s = sheet.copy()
    # background lift on this close: median V_T/ADV of index members with no change (controls), by country
    allk = set(md.key)
    chg = set(s.key)
    ctrl = md[~md.key.isin(chg) & np.isfinite(md.adv_sh) & np.isfinite(md.v_T) & (md.adv_sh > 0)].copy()
    ctrl["ratio"] = ctrl.v_T / ctrl.adv_sh
    lift = ctrl.groupby("country").ratio.median()
    lift_all = float(ctrl.ratio.median())
    # size-dependent lift: big names trade more on a 3rd-Friday close even with no index change (expiry,
    # benchmark-close trading), so the background is fitted on controls: log(V_T/ADV) = a_region + b log(ADV in EUR)
    ctrl["eur_adv"] = np.log(ctrl.adv_sh * ctrl.price * ctrl.eur_per_ccy)
    ctrl["reg"] = np.where(ctrl.country == "CA", "CA", "EU")
    ctrl = ctrl[np.isfinite(ctrl.eur_adv)]
    coef = {}
    for g, x in ctrl.groupby("reg"):
        b = np.polyfit(x.eur_adv, np.log(x.ratio), 1)
        coef[g] = b
    res_ = {g: (round(float(b[0]), 3), round(float(b[1]), 2)) for g, b in coef.items()}
    s["lift"] = s.country.map(lift).fillna(lift_all)
    s = s[np.isfinite(s.adv_sh) & np.isfinite(s.v_T) & (s.adv_sh > 0) & (s.net_sh != 0)].copy()
    s["excess"] = s.v_T - s.adv_sh                     # plain definition: V_T - ADV
    s["abnormal"] = s.v_T - s.lift * s.adv_sh          # net of the median background lift of the country (main)
    reg = np.where(s.country == "CA", "CA", "EU")
    la = np.log(s.adv_sh * s.price * s.eur_per_ccy)
    s["lift_size"] = [np.exp(np.polyval(coef[g], v)) if g in coef and np.isfinite(v) else np.nan
                      for g, v in zip(reg, la)]
    s["abnormal_size"] = s.v_T - s.lift_size.fillna(s.lift) * s.adv_sh   # robustness: size-dependent background
    s["absD"] = s.net_sh.abs()
    s["absD_vis"] = s.net_sh_visible.abs()
    s["m_T"] = s.v_T / s.adv_sh
    s["grp"] = np.where(s.change_types.str.contains("add|delete|migration"), "add/delete", "share/float/cap")
    res = {"n_changed": len(s), "n_controls": len(ctrl), "lift_all": lift_all,
           "lift_by_country": lift.round(2).to_dict(), "lift_size_fit": res_}

    def block(x, label):
        o = dict(sample=label, n=len(x))
        if len(x) < 5:
            return o
        o["spearman_absD_excess"] = stats.spearmanr(x.absD, x.excess).statistic
        o["spearman_absD_abnormal"] = stats.spearmanr(x.absD, x.abnormal).statistic
        o["spearman_absD_abnormal_size"] = stats.spearmanr(x.absD, x.abnormal_size).statistic
        o["spearman_absD_over_adv_mT"] = stats.spearmanr(x.absD / x.adv_sh, x.m_T).statistic
        pos = x[x.abnormal > 0]
        if len(pos) >= 5:
            b = np.polyfit(np.log(pos.absD), np.log(pos.abnormal), 1)
            pred = np.polyval(b, np.log(pos.absD))
            ss = ((np.log(pos.abnormal) - pred) ** 2).sum()
            st = ((np.log(pos.abnormal) - np.log(pos.abnormal).mean()) ** 2).sum()
            o.update(n_pos=len(pos), share_pos=len(pos) / len(x), slope=b[0], intercept=b[1], r2_loglog=1 - ss / st)
            o["median_abnormal_over_D"] = float(np.median(pos.abnormal / pos.absD))
            o["median_abnormal_over_Dvis"] = float(np.median(pos.abnormal / pos.absD_vis))
        # volume-model form (volume_forecasting.md §5.1): log m_T on log(1 + |D|/ADV)
        b2 = np.polyfit(np.log1p(x.absD / x.adv_sh), np.log(x.m_T), 1)
        pr = np.polyval(b2, np.log1p(x.absD / x.adv_sh))
        ss = ((np.log(x.m_T) - pr) ** 2).sum()
        st = ((np.log(x.m_T) - np.log(x.m_T).mean()) ** 2).sum()
        o.update(r2_logmT_on_log1px=1 - ss / st, slope_logmT=b2[0])
        return o

    blocks = [block(s, "all changed names")]
    for g, x in s.groupby("grp"):
        blocks.append(block(x, g))
    for idx, x in s.groupby("dominant_index"):
        blocks.append(block(x, f"dominant {idx}"))
    s["region"] = np.where(s.country == "CA", "CA", "EU")
    for g, x in s.groupby("region"):
        blocks.append(block(x, f"region {g}"))
    for g in ["add/delete"]:
        for reg, x in s[s.grp == g].groupby("region"):
            blocks.append(block(x, f"{g}, {reg}"))
    t = pd.DataFrame(blocks)
    # misses: residual of the pooled log-log fit on positive abnormal volume (and the names with abnormal <= 0)
    pos = s[s.abnormal > 0].copy()
    b = np.polyfit(np.log(pos.absD), np.log(pos.abnormal), 1)
    pos["pred_abnormal"] = np.exp(np.polyval(b, np.log(pos.absD)))
    pos["log_resid"] = np.log(pos.abnormal) - np.log(pos.pred_abnormal)
    s = s.merge(pos[["key", "pred_abnormal", "log_resid"]], on="key", how="left")
    return t, res, s


def run():
    rows = I.load(T)
    lines = rows.assign(key=[E.key_of(r) for r in rows.itertuples()])
    lines = lines.drop_duplicates(["key", "country"])[["key", "isin", "country", "currency", "price"]]
    # controls: every holding of the funds on T (no-change names give the background lift of the close)
    hs = pd.concat([H.parse(e, T) for e, _ in FUNDS]).drop_duplicates("isin")
    ctrl = hs[~hs["isin"].isin(set(lines["isin"]))]
    cl = pd.DataFrame(dict(key=ctrl["isin"], isin=ctrl["isin"], country=ctrl.country,
                           currency=ctrl.mkt_ccy.replace({"GBX": "GBP"}), price=ctrl.price))
    allines = pd.concat([lines, cl], ignore_index=True).rename(columns={"currency": "ccy"})
    md = M.md_r2("jun2026", T, allines)
    md.to_csv(VALID / f"{T}_marketdata.csv", index=False)
    notes = [f"Validation build: final inputs from iShares holdings diffs {D0} -> {T}; volume from R2 Datastream.",
             "Volume basis: Canada consolidated (Datastream exch 295); Europe primary + lit MTFs."]
    r = E.run(T, "best", md=md, label="final", extra_notes=notes)
    sheet = r["sheet"]
    t, res, s = fit(sheet, md, r["contrib"])
    t.to_csv(VALID / f"{T}_fit.csv", index=False, float_format="%.4g")
    s.to_csv(VALID / f"{T}_names.csv", index=False, float_format="%.6g")
    mc = r2_membership_check()
    mc.to_csv(VALID / f"{T}_membership_check.csv", index=False)
    sc = sanity_float_check()
    pis = r["pis"].set_index("index")
    sc["pi_visible"] = sc["index"].map(pis.pi_visible)
    sc["D_vis_sh"] = sc.pi_visible * sc.F_index
    sc.to_csv(VALID / f"{T}_sanity_adds.csv", index=False, float_format="%.6g")
    write_md(t, res, s, mc, sc, r)
    print(t.round(3).to_string(index=False))


def write_md(t, res, s, mc, sc, r):
    f = lambda v, d=2: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"  # noqa
    L = [f"# Validation: net-flow engine on the {T} close", "",
         f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `validate_2026_06_19.py`. Inputs are the "
         "trackers' actual index-share changes (iShares holdings dated T vs T-1; status final). Volume: Canada "
         "consolidated (Datastream 295), Europe primary + lit MTFs (Datastream). ADV = median of T-25..T-6, "
         "29 May (MSCI) removed. `abnormal` = V_T - lift x ADV, lift = median V_T/ADV of the "
         "index members of the same country with no change on this close (the close's background); `excess` = V_T - ADV.", "",
         f"Changed names with volume: {res['n_changed']}; no-change controls: {res['n_controls']}; background "
         f"lift (median V_T/ADV of controls) {res['lift_all']:.2f}x; by country {res['lift_by_country']}. Size-dependent "
         f"background fitted on the controls, log(V_T/ADV) = a + b log(EUR ADV), (b, a) by region: {res['lift_size_fit']}; "
         "`abnormal` uses the country median; `abnormal_size` the size-dependent fit (robustness).", "",
         "## Fit", "",
         "| sample | n | Spearman(|D|, excess) | Spearman(|D|, abnormal) | Spearman(|D|, abnormal_size) | Spearman(|D|/ADV, V_T/ADV) | share abnormal>0 | "
         "R² log abnormal on log |D| | slope | R² log m_T on log(1+|D|/ADV) | median abnormal / D | median abnormal / D_vis |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in t.itertuples():
        g = x._asdict()
        L.append(f"| {x.sample} | {x.n} | {f(g.get('spearman_absD_excess'))} | {f(g.get('spearman_absD_abnormal'))} | "
                 f"{f(g.get('spearman_absD_abnormal_size'))} | "
                 f"{f(g.get('spearman_absD_over_adv_mT'))} | {f(g.get('share_pos'))} | {f(g.get('r2_loglog'))} | "
                 f"{f(g.get('slope'))} | {f(g.get('r2_logmT_on_log1px'))} | {f(g.get('median_abnormal_over_D'))} | "
                 f"{f(g.get('median_abnormal_over_Dvis'), 1)} |")
    L += ["", "D = central demand (R x visible pi x dF); D_vis = visible trackers only. `median abnormal / D_vis` is the "
          "event-implied R on this close (flow_sizing §4.4 priors: 4.2 TSX, 5.1 STOXX).", ""]
    L += ["## Biggest misses (pooled log-log fit; and names whose volume fell short of the background lift)", ""]
    cols = ["ticker", "name", "country", "change_types", "net_sh", "adv_sh", "v_T", "m_T", "abnormal", "pred_abnormal",
            "log_resid"]
    over = s.dropna(subset=["log_resid"]).sort_values("log_resid", ascending=False).head(10)
    under = s.dropna(subset=["log_resid"]).sort_values("log_resid").head(10)
    none_ = s[s.abnormal <= 0].sort_values("absD", ascending=False).head(10)
    for title, x in (("Volume far above prediction", over), ("Volume far below prediction", under),
                     ("Largest predicted demand with no abnormal volume", none_)):
        L += [f"**{title}**", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r_ in x.itertuples():
            d = r_._asdict()
            L.append("| " + " | ".join(
                (f"{d[c]:,.0f}" if c in ("net_sh", "adv_sh", "v_T", "abnormal", "pred_abnormal") and np.isfinite(d[c])
                 else f"{d[c]:.2f}" if c in ("m_T", "log_resid") and isinstance(d[c], float) and np.isfinite(d[c])
                 else str(d[c])[:40]) for c in cols) + " |")
        L.append("")
    L += ["## Membership check: holdings-diff adds/deletes vs R2", "", mc.to_markdown(index=False), ""]
    L += ["## Sanity check: known adds, index shares vs an independent float estimate", "",
          "D_vis = visible pi x F_index is what the visible trackers had to buy; F_index comes from the tracker itself "
          "(holdings / k).", "", sc.round(3).to_markdown(index=False), ""]
    (VALID / f"{T}_validation.md").write_text("\n".join(L))


if __name__ == "__main__":
    {"inputs": build_inputs, "run": run}[sys.argv[1]]()
