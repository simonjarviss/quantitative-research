"""Live shadow prediction of an S&P/TSX Composite quarterly review (paper-only research; no orders, no accounts).

Usage
  python3 predict.py --asof 2026-10-09 --review 2026-12            # today's H-10w run for Dec 2026
  python3 predict.py --asof 2026-11-20 --review 2026-12            # reference-date run (label R)
  python3 predict.py --asof 2026-12-18 --review 2027-03            # March 2027 H3 run
  python3 predict.py --asof 2026-08-21 --review 2026-09 --backfill # pseudo point-in-time re-run (memory input)
Options: --label NAME (default: R / H3 / H-<weeks>w), --offline (cached raw files only), --no-log,
         --force (overwrite an existing run folder; otherwise refused so logged SHA-256 values stay valid).
Same --asof re-uses cache/yf_*_<asof>.parquet (reproducible); delete those files to refetch.

Model = the backtest's screen (../../12_studies/tsx_composite/04_predict.py, current-era rules) on live data,
plus the backtest's overlays (05_filters.py) as a second variant:
  rules : literal screen "if the review were held at the screen date" (H3/R style).
  main  : rules minus pending M&A targets (SDC to 2026-06-26 + hand list ma_pending.csv) minus committee
          memory (non-members our R-screen passed at >= 1.25x the size threshold at one of the last two
          announced reviews but S&P did not add; members our R-screen failed but S&P kept).
Screen date d = min(asof, R). At R the size VWAP uses the 10 TSX sessions before R (SPTSX-M p.5); before R it
uses the last 10 sessions to d (backtest H3 convention). Turnover = 12 months to d, Canada (TSX volume x the
name's consolidated/TSX ratio from R2) + US exchange volume, over float shares.

Outputs: runs/<review>/<asof>_<label>/{predictions.csv, screen_all.csv, run_meta.json}; appends shadow_log.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import live_common as lc

warnings.simplefilter("ignore", FutureWarning)
R = lc.RULES


def default_label(asof: pd.Timestamp, rd: dict) -> str:
    if asof >= rd["R"]:
        return "R"
    if abs((asof - rd["H3"]).days) <= 5:
        return "H3"
    return f"H-{int(round((rd['T'] - asof).days / 7))}w"


def known_members(xic: pd.DataFrame, asof: pd.Timestamp, changes: pd.DataFrame) -> tuple[set, list]:
    """XIC constituents plus quarterly-review changes announced on or before asof but not yet effective."""
    mem = set(xic.symbol)
    applied = []
    for rv, g in changes.groupby("review"):
        rd = lc.review_dates(rv)
        ann = pd.Timestamp(g.announced.iloc[0]) if "announced" in g and g.announced.iloc[0] else rd["A"]
        if ann <= asof < rd["eff"]:
            for r in g.itertuples():
                (mem.add if r.side == "add" else mem.discard)(r.symbol)
                applied.append(f"{rv}:{r.side}:{r.symbol}")
    return mem, applied


def estimate_k(xic: pd.DataFrame, mig: pd.DataFrame) -> tuple[float, int]:
    """XIC shares held / index float shares. Single-class members: ratio = k x IWF; IWF = 1 for most names, so k
    is the top cluster of the ratio distribution (median of ratios >= 0.9 x the 90th percentile)."""
    x = xic[~xic.symbol.str.contains(r"\.(?:A|B|X|SV)$")].copy()
    x["root"] = x.symbol.str.replace(r"\.UN$", "", regex=True)
    x = x.merge(mig[["root", "mig_os"]], on="root", how="inner")
    r = (x.xic_shares / x.mig_os).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[(r > 0) & (r < 0.05)]
    top = r[r >= 0.9 * r.quantile(0.9)]
    return float(top.median()), int(len(top))


def committee_memory(review: str, asof: pd.Timestamp, changes: pd.DataFrame) -> tuple[set, set, list]:
    """Reject/protect sets from the R-screens of the last two reviews announced on or before asof.

    Source per review: this pipeline's own R run (runs/<review>/*_R, incl. backfills) if present, else the
    backtest's R-screen (predictions_final.csv, reviews to 2026-06). Same rule as 05_filters.py."""
    out_rej, out_prot, notes = set(), set(), []
    past = [rv for rv in lc.prev_reviews(review, 4) if lc.review_dates(rv)["A"] <= asof][:2]
    bt = None
    for rv in past:
        own = sorted((lc.RUNS / rv).glob("*_R/screen_all.csv"))
        if rv <= "2026-06" and not own:                      # live pipeline's own R-screen preferred
            if bt is None:
                bt = pd.read_csv(lc.BT / "predictions_final.csv", usecols=[
                    "review", "horizon", "tic", "w", "pred_add_ma", "pred_del_ma", "act_add", "act_del"])
            s = bt[(bt.review == rv) & (bt.horizon == "R")].copy()
            s["symbol"] = s.tic.map(lc.norm_tic)
            src = "backtest predictions_final.csv (R2)"
        else:
            runs = own
            if not runs:
                notes.append(f"memory: no R-screen for {rv} (run predict.py --asof <R> --review {rv} --backfill)")
                continue
            s = pd.read_csv(runs[-1], dtype={"symbol": str})
            s["pred_add_ma"] = s.pred_add_rules & ~s.ma_pending
            s["pred_del_ma"] = s.pred_del_rules & ~s.ma_pending
            ch = changes[changes.review == rv]
            if ch.empty:
                notes.append(f"memory: no announced changes recorded for {rv} in review_changes.csv")
                continue
            s["act_add"] = s.symbol.isin(ch[ch.side == "add"].symbol)
            s["act_del"] = s.symbol.isin(ch[ch.side == "delete"].symbol)
            src = f"own R run {runs[-1].parent.name}"
        rej = s[s.pred_add_ma & ~s.act_add & (s.w >= 1.25 * R["size_entry"])].symbol
        prot = s[s.pred_del_ma & ~s.act_del].symbol
        out_rej |= set(rej)
        out_prot |= set(prot)
        notes.append(f"memory {rv} ({src}): reject {sorted(rej)} protect {sorted(prot)}")
    return out_rej, out_prot, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True)
    ap.add_argument("--review", required=True, help="YYYY-MM, e.g. 2026-12 or 2027-03")
    ap.add_argument("--label")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="historical re-run; logged as BACKFILL, not shadow")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run folder (breaks logged hashes)")
    a = ap.parse_args()
    t0 = dt.datetime.now(dt.timezone.utc)
    asof = pd.Timestamp(a.asof)
    review = a.review
    rd = lc.review_dates(review)
    label = a.label or default_label(asof, rd)
    if (lc.RUNS / review / f"{asof:%Y-%m-%d}_{label}" / "predictions.csv").exists() and not a.force:
        raise SystemExit(f"runs/{review}/{asof:%Y-%m-%d}_{label} exists and is logged with its SHA-256; "
                         f"use --label <new> for a re-run (or --force to overwrite)")
    cal = lc.sessions()
    d = min(lc.last_on_or_before(cal, asof), rd["R"])                     # screen date
    notes = []
    # intervening review: a literal screen predicts the earlier review, not this one
    for rv in lc.prev_reviews(review, 2):
        if lc.review_dates(rv)["A"] > asof:
            notes.append(f"WARNING: review {rv} is announced after asof; a literal screen predicts {rv} first")

    # ---------------------------------------------------------------- raw inputs (date-stamped, hashed)
    p_xic = lc.fetch_xic(asof, a.offline)
    p_mig = lc.fetch_mig(a.offline)
    p_dir = lc.fetch_tsxdir(a.offline)
    xic_date, xic = lc.parse_xic(p_xic)
    mig_date, mig = lc.parse_mig(p_mig)
    tdir = lc.parse_tsxdir(p_dir)
    changes = lc.load_review_changes()
    members, applied = known_members(xic, asof, changes)
    if applied:
        notes.append("announced, not yet effective, applied to membership: " + ", ".join(applied))
    k, nk = estimate_k(xic, mig)
    M_est = xic.xic_mv.sum() / k

    # ---------------------------------------------------------------- universe (TSX equity lines)
    tdir = tdir.assign(root=tdir.symbol.str.split(".").str[0])               # MiG root = base ticker of the line
    u = tdir.merge(mig, on="root", how="left")
    u = u[u.symbol.str.match(lc.CLASS_OK) & u.operating.fillna(False)]
    u = u[~u.symbol.str.endswith(".U")]
    nlines = u.groupby("root").symbol.transform("count")
    u = u.assign(n_class_lines=nlines)
    cut = 0.25 * R["size_entry"] * M_est                                     # coarse pre-filter (issuer cap)
    keep = (u.mig_mcap >= cut) | u.symbol.isin(members)
    u = u[keep].copy()
    miss = sorted(members - set(u.symbol))
    if miss:                                                                 # members not matched to MiG/dir
        extra = pd.DataFrame(dict(symbol=miss, root=[s.split(".")[0] for s in miss]))
        extra = extra.merge(mig, on="root", how="left")
        extra["n_class_lines"] = 1
        u = pd.concat([u, extra], ignore_index=True)
        notes.append(f"members not in TSX directory/MiG operating set (added back): {miss}")
    u = u.drop_duplicates("symbol").reset_index(drop=True)
    u["member"] = u.symbol.isin(members)
    u["yf"] = u.symbol.map(lc.yf_symbol)

    # ---------------------------------------------------------------- R2 links (floats, ratio, US ticker, domicile)
    L = lc.r2_links()
    L = L[L.has_px & (L.last_px >= "2025-06-01")].sort_values("last_px").drop_duplicates("symbol", keep="last")
    u = u.merge(L[["symbol", "line", "conm", "fic", "loc", "cons_ratio", "us_tic_r2", "cusip"]], on="symbol", how="left")
    ff = lc.r2_float_at(asof)
    u = u.merge(ff, on="line", how="left")
    # US ticker: Compustat US exchange line, else MiG interlisting flag with the TSX root as a guess
    u["us_ticker"] = u.us_tic_r2
    guess = u.us_ticker.isna() & u.us_listed_mig.fillna(False)
    u.loc[guess, "us_ticker"] = u.loc[guess, "root"]
    u["us_src"] = np.where(u.us_tic_r2.notna(), "R2", np.where(guess, "MiG-guess", ""))
    ov = u.symbol.map(lc.US_TICKER_OVERRIDES)
    u.loc[ov.notna(), "us_ticker"] = ov[ov.notna()]
    u.loc[ov.notna(), "us_src"] = "override"
    # multi-class issuers: US volume goes to one line only (the most traded TSX line; set after download)

    # ---------------------------------------------------------------- yfinance
    tag = f"{asof:%Y%m%d}"
    start = (d - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    end = (asof + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    syms = list(u.yf) + list(u.us_ticker.dropna().unique()) + ["XIC.TO"]
    H = lc.yf_history(syms, start, end, tag, a.offline)
    C, AC, V = H["Close"], H["Adj Close"], H["Volume"]
    sess = cal[(cal >= C.index.min()) & (cal <= asof)]
    C, AC, V = C.reindex(sess), AC.reindex(sess), V.reindex(sess)
    di = sess.get_loc(d)

    info = lc.yf_info(list(u.yf), tag, a.offline)
    u = u.merge(info.rename(columns={"sharesOutstanding": "yf_so", "floatShares": "yf_float", "country": "yf_country"})
                [["yf", "yf_so", "yf_float", "yf_country"]], on="yf", how="left")

    def col(frame, s):
        return frame[s].values.astype(float) if s in frame else np.full(len(frame), np.nan)

    n = len(u)
    vw10, vw3, live, tsx12, us12, us63, close_d, close_asof, adj_asof, sig, first_px = (np.full(n, np.nan) for _ in range(11))
    w_end = di - 1 if d == rd["R"] else di                                   # VWAP window end
    d0 = sess.searchsorted(d - pd.Timedelta(days=365), side="right")
    mk = AC["XIC.TO"].values.astype(float)
    mret = np.log(mk[1:] / mk[:-1])
    for i, r in enumerate(u.itertuples()):
        c, v = col(C, r.yf), col(V, r.yf)
        if np.all(np.isnan(c)):
            continue
        sl = slice(w_end - R["vwap_days"] + 1, w_end + 1)
        cv, vv = c[sl], np.nan_to_num(v[sl])
        vw10[i] = np.nansum(cv * vv) / vv.sum() if vv.sum() > 0 else np.nanmean(cv)
        s3 = slice(max(0, di - 62), di + 1)
        c3, v3 = c[s3], np.nan_to_num(v[s3])
        vw3[i] = np.nansum(c3 * v3) / v3.sum() if v3.sum() > 0 else np.nanmean(c3)
        live[i] = float((~np.isnan(c[max(0, di - 9):di + 1])).any() and np.nansum(v[max(0, di - 9):di + 1]) > 0)
        lst = r.listing_date if isinstance(r.listing_date, pd.Timestamp) and not pd.isna(r.listing_date) else None
        a0 = max(d0, sess.searchsorted(lst)) if lst is not None else d0      # SPTSX-M p.5 (E4)
        tsx12[i] = np.nansum(v[a0:di + 1])
        close_d[i] = c[di] if not np.isnan(c[di]) else np.nan
        close_asof[i] = c[-1]
        adj_asof[i] = col(AC, r.yf)[-1]
        fp = np.where(~np.isnan(c))[0]
        first_px[i] = fp[0] if len(fp) else np.nan
        ac = col(AC, r.yf)[max(0, di - 120):di + 1]
        rr = np.log(ac[1:] / ac[:-1]) - mret[max(0, di - 120):di]
        rr = rr[~np.isnan(rr)]
        sig[i] = np.clip(np.std(rr), 0.005, 0.10) if len(rr) >= 60 else 0.03
        if isinstance(r.us_ticker, str) and r.us_ticker in V:
            uv = col(V, r.us_ticker)
            us12[i] = np.nansum(uv[d0:di + 1])
            us63[i] = np.nansum(uv[max(0, di - 62):di + 1])
    u["vwap10"], u["vwap3m"], u["live"] = vw10, vw3, live == 1
    u["tsx_vol12"], u["close_screen"], u["close_asof"], u["adjclose_asof"], u["sigma"] = tsx12, close_d, close_asof, adj_asof, sig
    # US volume to the most traded TSX line of each issuer only
    u["us_vol12"], u["us_vol63"] = us12, us63
    top = u.sort_values("tsx_vol12", ascending=False).drop_duplicates("root").symbol
    u.loc[~u.symbol.isin(top), ["us_vol12", "us_vol63"]] = np.nan
    u["dual"] = (u.us_vol63.fillna(0) > 0)
    med_ratio = float(np.nanmedian(L.cons_ratio))
    u["ratio_src"] = np.where(u.cons_ratio.notna(), "R2", "median")
    u["cons_ratio"] = u.cons_ratio.fillna(med_ratio).clip(1.0, 4.0)
    u["ca_vol12"] = u.tsx_vol12 * u.cons_ratio

    # ---------------------------------------------------------------- shares and float
    # members: S&P float shares backed out of XIC (shares held / k)
    u = u.merge(xic[["symbol", "xic_shares", "xic_weight_pct"]], on="symbol", how="left")
    single = u.n_class_lines.fillna(1) <= 1
    u["so"] = np.where(single & u.mig_os.notna(), u.mig_os, u.yf_so)
    u["so_src"] = np.where(single & u.mig_os.notna(), "MiG", np.where(u.yf_so.notna(), "YF", ""))
    u.loc[u.so.isna(), "so"] = u.mig_os
    u["so_yf_vs_mig"] = u.yf_so / u.mig_os
    yf_ff = (u.yf_float / u.yf_so).where((u.yf_float / u.yf_so).between(0.05, 1.0))
    iwf = np.where(u.ff_ds.notna(), u.ff_ds, np.where(u.ff_ws.notna(), u.ff_ws, np.where(yf_ff.notna(), yf_ff, 1.0)))
    src = np.where(u.ff_ds.notna(), "DS", np.where(u.ff_ws.notna(), "WS", np.where(yf_ff.notna(), "YF", "default100")))
    iwf = np.round(iwf.astype(float), 2)
    iwf = np.where(iwf >= 0.96, 1.0, iwf)                                    # S&P-style rounding
    u["iwf"], u["iwf_src"] = iwf, src
    u["float_sh"] = u.so * u.iwf
    m_x = u.member & u.xic_shares.notna()
    u.loc[m_x, "float_sh"] = u.loc[m_x, "xic_shares"] / k
    u.loc[m_x, "iwf_src"] = "XIC"
    u.loc[m_x, "iwf"] = (u.loc[m_x, "float_sh"] / u.loc[m_x, "so"]).round(2)
    u["qmv"] = u.float_sh * u.vwap10
    nopx = u.member & u.qmv.isna() & u.xic_shares.notna()                    # e.g. delisted since (backfills)
    u.loc[nopx, "qmv"] = u.loc[nopx, "float_sh"] * u.loc[nopx, "symbol"].map(dict(zip(xic.symbol, xic.xic_price)))
    u.loc[nopx, "live"] = True
    u["qmv_src"] = np.where(nopx, "XIC_price", "vwap10")
    u["mcap"] = u.so * u.close_asof

    # ---------------------------------------------------------------- denominator and size
    mem = u.member & u.live & (u.qmv > 0)
    M = u.loc[mem, "qmv"].sum()
    wmax = u.loc[mem, "qmv"].max() / M
    capped = 0
    if wmax > R["cap_switch"]:                                               # App. G reading (spec §10 step 3)
        q = np.sort(u.loc[mem, "qmv"].values)[::-1]
        for kc in range(1, 6):
            Mc = q[kc:].sum() / (1 - R["cap_switch"] * kc)
            if q[kc] <= R["cap_switch"] * Mc:
                M, capped = Mc, kc
                break
    M_close = (u.loc[mem, "float_sh"] * u.loc[mem, "close_screen"]).sum()
    u["w"] = np.where(u.member, u.qmv / M, u.qmv / (M + u.qmv))

    # ---------------------------------------------------------------- eligibility and tests
    u["turn"] = (u.ca_vol12 + u.us_vol12.fillna(0)) / u.float_sh
    u["turn_ca"] = u.ca_vol12 / u.float_sh
    first_date = pd.Series(sess[np.nan_to_num(first_px, nan=len(sess) - 1).astype(int)], index=u.index)
    u["listing_src"] = np.where(u.listing_date.notna(), "MiG", "first_yf_price")
    u["listing_date"] = u.listing_date.fillna(first_date)
    u["seasoned"] = u.listing_date <= rd["seasoning_cutoff"]
    ban_rv = lc.prev_reviews(review, R["ban_reviews"])
    banned = set(changes[(changes.review.isin(ban_rv)) & (changes.side == "delete")].symbol)
    u["banned"] = u.symbol.isin(banned)
    u["dom_ok"] = u.hq_region.eq("Canada") | u.fic.eq("CAN") | u["loc"].eq("CAN")
    u["dom_unverified"] = ~u.hq_region.eq("Canada") & u.fic.isna()
    u.loc[u.dom_unverified, "dom_ok"] = True                                # foreign HQ, incorporation unknown: keep, flag
    u["elig"] = u.dom_ok & u.live
    price_in = (u.vwap3m >= R["price_min"]) & (u.vwap10 >= R["price_min"])
    turn_in = (u.turn >= R["turn_entry"]) & (~u.dual | (u.turn_ca >= R["ca_only_entry"]))
    turn_st = (u.turn >= R["turn_stay"]) & (~u.dual | (u.turn_ca >= R["ca_only_stay"]))
    u["nonsize_in"] = u.elig & ~u.member & u.seasoned & ~u.banned & price_in & turn_in
    u["pred_add_rules"] = u.nonsize_in & (u.w >= R["size_entry"])
    u["pred_del_rules"] = u.member & u.vwap10.notna() & ~((u.vwap3m >= R["price_min"]) & turn_st & (u.w >= R["size_stay"]))

    # margins (fraction above (+) / below (-) each threshold) and binding rule
    nm = ~u.member
    u["thr_size"] = np.where(nm, R["size_entry"], R["size_stay"])
    u["m_size"] = u.w / u.thr_size - 1
    u["m_turn"] = u.turn / np.where(nm, R["turn_entry"], R["turn_stay"]) - 1
    u["m_turn_ca"] = np.where(u.dual, u.turn_ca / np.where(nm, R["ca_only_entry"], R["ca_only_stay"]) - 1, np.nan)
    u["m_price"] = np.where(nm, np.fmin(u.vwap10, u.vwap3m), u.vwap3m) / R["price_min"] - 1
    mcols = ["m_size", "m_turn", "m_turn_ca", "m_price"]
    mm = u[mcols].astype(float)
    u["binding_rule"] = mm.fillna(np.inf).idxmin(axis=1).str.replace("m_", "", regex=False)
    u["margin"] = mm.min(axis=1)
    fails = []
    for r in u.itertuples():
        f = [c[2:] for c in mcols if not np.isnan(getattr(r, c)) and getattr(r, c) < 0]
        if not r.member:
            f += [x for x, ok in [("seasoning", r.seasoned), ("reentry_ban", not r.banned),
                                  ("domicile", r.dom_ok), ("not_trading", r.live)] if not ok]
        fails.append(",".join(f))
    u["failing_rules"] = fails

    # P(size test passes at R): driftless random walk of log relative size, trailing 120-session vol
    t_left = max(0, cal.searchsorted(rd["W10_end"]) - 4 - cal.get_loc(d))   # sessions to the VWAP-window middle
    x0 = np.log(u.w.clip(lower=1e-9) / u.thr_size)
    if t_left > 0:
        u["p_size_pass_R"] = norm.cdf(x0 / (u.sigma * np.sqrt(t_left)))
    else:
        u["p_size_pass_R"] = (x0 >= 0).astype(float)

    # ---------------------------------------------------------------- overlays (main variant)
    sdc = lc.sdc_pending(asof)
    c6 = u.cusip.astype(str).str[:6]
    ma, ma_cancel = lc.load_ma(asof)
    sdc_hit = c6.isin(set(sdc.master_cusip)) & ~u.symbol.isin(ma_cancel)
    hand_hit = u.symbol.isin(set(ma.symbol))
    u["ma_pending"] = sdc_hit | hand_hit
    u["ma_source"] = np.where(hand_hit, "ma_pending.csv", np.where(sdc_hit, "SDC(R2, to 2026-06-26)", ""))
    rej, prot, mnotes = committee_memory(review, asof, changes)
    notes += mnotes
    u["mem_reject"] = u.symbol.isin(rej) & ~u.member
    u["mem_protect"] = u.symbol.isin(prot) & u.member
    u["pred_add_main"] = u.pred_add_rules & ~u.ma_pending & ~u.mem_reject
    u["pred_del_main"] = u.pred_del_rules & ~u.ma_pending & ~u.mem_protect

    # flags for discretion / data quality
    fl = []
    for r in u.itertuples():
        f = []
        if r.iwf_src == "default100":
            f.append("float_default100")
        if r.iwf_src == "YF":
            f.append("float_yahoo")
        if r.so_src == "YF" and not r.member:
            f.append("multi_class_or_no_MiG_shares")
        if isinstance(r.so_yf_vs_mig, float) and not np.isnan(r.so_yf_vs_mig) and abs(r.so_yf_vs_mig - 1) > 0.10 and r.n_class_lines == 1:
            f.append(f"shares_yf/mig={r.so_yf_vs_mig:.2f}")
        if r.ratio_src == "median":
            f.append("cons_vol_ratio_median")
        if r.us_src == "MiG-guess":
            f.append("us_ticker_guess")
        if r.dom_unverified:
            f.append(f"foreign_HQ({r.hq_region})_incorp_unverified")
        if r.listing_src != "MiG":
            f.append("listing_date_from_prices")
        if r.listing_date > d - pd.Timedelta(days=365):
            f.append("short_history(<12m on TSX)")
        if np.isnan(r.vwap10):
            f.append("no_yfinance_prices")
        fl.append(";".join(f))
    u["flags"] = fl

    # ---------------------------------------------------------------- candidate list
    st = np.full(len(u), "", dtype=object)
    st[u.pred_add_rules] = "pred_add"
    st[u.pred_del_rules] = "pred_delete"
    near_add = ~u.member & ~u.pred_add_rules & u.elig & (
        (u.nonsize_in & (u.m_size >= -0.30)) |                               # size just short
        ((u.m_size >= 0) & ~u.nonsize_in))                                   # big enough, blocked by another rule
    st[near_add & (u.m_size >= 0)] = "blocked_add"
    st[near_add & (u.m_size < 0)] = "near_add"
    near_del = u.member & ~u.pred_del_rules & (u[["m_size", "m_turn", "m_turn_ca", "m_price"]].min(axis=1) < 0.30)
    st[near_del] = "near_delete"
    u["status"] = st
    u["review"], u["asof"], u["label"], u["screen_date"] = review, asof.date(), label, d.date()
    out_cols = ["review", "asof", "label", "screen_date", "status", "symbol", "yf", "issuer", "member",
                "pred_add_rules", "pred_del_rules", "pred_add_main", "pred_del_main", "ma_pending", "ma_source",
                "mem_reject", "mem_protect", "binding_rule", "margin", "failing_rules", "w", "thr_size", "m_size",
                "p_size_pass_R", "turn", "m_turn", "turn_ca", "m_turn_ca", "dual", "vwap10", "vwap3m", "m_price",
                "close_asof", "adjclose_asof", "qmv", "float_sh", "so", "so_src", "iwf", "iwf_src", "mcap",
                "tsx_vol12", "cons_ratio", "ratio_src", "ca_vol12", "us_ticker", "us_vol12", "listing_date",
                "listing_type", "seasoned", "banned", "hq_region", "dom_ok", "sigma", "xic_weight_pct", "flags"]
    order = {"pred_add": 0, "pred_delete": 1, "blocked_add": 2, "near_add": 3, "near_delete": 4}
    cand = u[u.status != ""].copy()
    cand["_o"] = cand.status.map(order)
    cand = cand.sort_values(["_o", "m_size"], ascending=[True, False]).drop(columns="_o")[out_cols]

    outdir = lc.RUNS / review / f"{asof:%Y-%m-%d}_{label}"
    outdir.mkdir(parents=True, exist_ok=True)
    p_pred = outdir / "predictions.csv"
    p_all = outdir / "screen_all.csv"
    cand.to_csv(p_pred, index=False, float_format="%.6g")
    u.assign(review=review)[out_cols + ["root", "n_class_lines", "nonsize_in", "elig", "dom_unverified", "conm", "fic",
                                        "loc", "ff_ds", "ff_ws", "yf_so", "yf_float", "mig_os", "mig_mcap"]] \
        .to_csv(p_all, index=False, float_format="%.6g")
    t1 = dt.datetime.now(dt.timezone.utc)
    cov = dict(n_lines=int(len(u)), n_members=int(u.member.sum()), members_in_xic=int(len(xic)),
               no_prices=int(u.vwap10.isna().sum()),
               iwf_src=u[~u.member].iwf_src.value_counts().to_dict(),
               iwf_src_candidates=cand[~cand.member].iwf_src.value_counts().to_dict(),
               ratio_median_candidates=int((cand.ratio_src == "median").sum()),
               us_guess=int((u.us_src == "MiG-guess").sum()), dual=int(u.dual.sum()),
               dom_unverified_candidates=int(cand["flags"].str.contains("incorp_unverified").sum()))
    meta = dict(review=review, asof=str(asof.date()), label=label, backfill=a.backfill, screen_date=str(d.date()),
                run_started_utc=t0.isoformat(timespec="seconds"), run_finished_utc=t1.isoformat(timespec="seconds"),
                run_seconds=round((t1 - t0).total_seconds(), 1),
                dates={k_: str(v.date()) if hasattr(v, "date") else v for k_, v in rd.items()},
                xic_file=p_xic.name, xic_holdings_date=str(xic_date.date()), xic_sha256=lc.sha256(p_xic),
                mig_file=p_mig.name, mig_data_date=mig_date, mig_sha256=lc.sha256(p_mig),
                tsxdir_file=p_dir.name, tsxdir_sha256=lc.sha256(p_dir),
                k_xic_share_of_float=k, k_n=nk, M_index_float_cap_cad=M, M_close_cad=M_close, max_weight=wmax,
                capped_switch=capped, entry_threshold_cad=R["size_entry"] * M, stay_threshold_cad=R["size_stay"] * M,
                coarse_cut_cad=cut, cons_ratio_median=med_ratio, t_sessions_to_vwap_mid=int(t_left),
                counts=dict(pred_add_rules=int(u.pred_add_rules.sum()), pred_del_rules=int(u.pred_del_rules.sum()),
                            pred_add_main=int(u.pred_add_main.sum()), pred_del_main=int(u.pred_del_main.sum()),
                            near_add=int((u.status == "near_add").sum()), blocked_add=int((u.status == "blocked_add").sum()),
                            near_delete=int((u.status == "near_delete").sum())),
                coverage=cov, banned=sorted(banned), memory_reject=sorted(rej), memory_protect=sorted(prot),
                ma_live_hand=ma.symbol.tolist(), notes=notes)
    p_meta = outdir / "run_meta.json"
    p_meta.write_text(json.dumps(meta, indent=1, default=str))
    hashes = {p.name: lc.sha256(p) for p in [p_pred, p_all, p_meta]}

    # ---------------------------------------------------------------- console summary
    pd.set_option("display.width", 250)
    show = ["status", "symbol", "issuer", "pred_add_main", "pred_del_main", "binding_rule", "m_size", "m_turn",
            "turn", "w", "p_size_pass_R", "close_asof", "iwf_src", "flags"]
    print(f"{review} {label} asof {asof.date()} screen {d.date()}  M=C${M/1e12:.3f}tn  entry C${R['size_entry']*M/1e9:.2f}bn"
          f"  stay C${R['size_stay']*M/1e9:.2f}bn  k={k:.5f}")
    print(cand[cand.status.isin(["pred_add", "pred_delete"])][show].to_string(index=False))
    for x in notes:
        print(" *", x)

    if not a.no_log:
        write_log(meta, outdir, hashes, cand, u)
    return 0


def near_lines(cand) -> str:
    out = ""
    for st, title in [("blocked_add", "Blocked adds (size passes, another rule fails)"),
                      ("near_add", "Near adds (size short by < 30%)"), ("near_delete", "Near deletes (< 30% above a stay test)")]:
        x = cand[cand.status == st]
        if len(x):
            out += f"- {title}: " + "; ".join(
                f"{r.symbol} {r.m_size:+.0%} size" + (f" (fails {r.failing_rules})" if isinstance(r.failing_rules, str) and r.failing_rules else
                                                      f" (binding {r.binding_rule} {r.margin:+.0%})")
                + f" C${r.close_asof:.2f}" for r in x.itertuples()) + "\n"
    return out


def write_log(meta, outdir, hashes, cand, u):
    hdr_needed = not lc.LOG.exists()
    rel = outdir.relative_to(lc.HERE)
    adds = cand[cand.pred_add_rules]
    dels = cand[cand.pred_del_rules]

    def fmt(df, side):
        if df.empty:
            return "  - none\n"
        s = ""
        for r in df.itertuples():
            main = r.pred_add_main if side == "add" else r.pred_del_main
            why = []
            if r.ma_pending:
                why.append("M&A pending")
            if r.mem_reject:
                why.append("memory: not added before")
            if r.mem_protect:
                why.append("memory: kept before")
            s += (f"  - {r.symbol} ({r.issuer}): margin {r.margin:+.0%} (binding: {r.binding_rule}), size {r.m_size:+.0%}, "
                  f"turnover {r.turn:.2f}, close C${r.close_asof:.2f}, main={'yes' if main else 'no (' + ', '.join(why) + ')'}"
                  f"{', flags: ' + r.flags if r.flags else ''}\n")
        return s

    kind = "BACKFILL (pseudo point-in-time, not out-of-sample)" if meta["backfill"] else "SHADOW (out-of-sample)"
    c = meta["counts"]
    cov = meta["coverage"]
    entry = (f"\n## {meta['review']} {meta['label']} | as-of {meta['asof']} | {kind}\n\n"
             f"- Run: {meta['run_started_utc']} to {meta['run_finished_utc']} UTC ({meta['run_seconds']} s). Screen date {meta['screen_date']}; "
             f"reference date {meta['dates']['R']} (alt {meta['dates']['R_alt']}); expected announcement {meta['dates']['A']} "
             f"(latest {meta['dates']['A_latest']}); trade close {meta['dates']['T']}.\n"
             f"- Output: `{rel}/`\n"
             + "".join(f"  - `{n}` sha256 `{h}`\n" for n, h in hashes.items())
             + f"- Inputs: XIC holdings as of {meta['xic_holdings_date']} (`{meta['xic_file']}`, sha256 `{meta['xic_sha256'][:16]}…`); "
               f"MiG {meta['mig_data_date']} (`{meta['mig_file']}`); TSX directory `{meta['tsxdir_file']}`; yfinance cache `cache/yf_*_{meta['asof'].replace('-', '')}.parquet`.\n"
               f"- Index float cap M = C${meta['M_index_float_cap_cad']/1e12:.3f}tn (close-priced C${meta['M_close_cad']/1e12:.3f}tn; "
               f"k = {meta['k_xic_share_of_float']:.5f} from {meta['k_n']} names); max weight {meta['max_weight']:.2%}; "
               f"entry 0.04% = C${meta['entry_threshold_cad']/1e9:.2f}bn, stay 0.025% = C${meta['stay_threshold_cad']/1e9:.2f}bn.\n"
               f"- Counts: rules adds {c['pred_add_rules']}, deletes {c['pred_del_rules']}; main adds {c['pred_add_main']}, "
               f"deletes {c['pred_del_main']}; near-adds {c['near_add']}, blocked adds {c['blocked_add']}, near-deletes {c['near_delete']}. "
               f"Lines screened {cov['n_lines']}, members {cov['n_members']}.\n"
             + "- Predicted adds (rules):\n" + fmt(adds, "add")
             + "- Predicted deletes (rules):\n" + fmt(dels, "delete")
             + near_lines(cand)
             + f"- Coverage: non-member float sources {cov['iwf_src']}; candidates {cov['iwf_src_candidates']}; "
               f"candidates on the median volume ratio {cov['ratio_median_candidates']}; US tickers guessed {cov['us_guess']}; "
               f"dual-listed {cov['dual']}; lines without prices {cov['no_prices']}.\n"
               f"- Known error sources: float proxies for non-members (Datastream/Worldscope as of {u.ff_month.max():%Y-%m} "
               f"from R2, then Yahoo floatShares, then 100%); members use XIC-implied float; Canadian volume = TSX volume "
               f"x R2 consolidated/TSX ratio (no consolidated feed after 2026-06-24); US volume from yfinance; "
               f"shares O/S from MiG {meta['mig_data_date']} (S&P uses the {meta['dates']['C']} cut-off); "
               f"rulebook = SPTSX-M June 2025, the 11 Sep 2026 S&P/TSX Indices Update is unread; GICS not checked; "
               f"M&A list hand-maintained.\n"
             + ("- Notes: " + " | ".join(meta["notes"]) + "\n" if meta["notes"] else ""))
    with open(lc.LOG, "a") as f:
        if hdr_needed:
            f.write("# Shadow log: S&P/TSX Composite quarterly-review predictions\n\n"
                    "Append-only. Each entry is written by predict.py or score.py. Do not edit past entries.\n")
        f.write(entry)


if __name__ == "__main__":
    sys.exit(main())
