"""Live shadow prediction of the December 2026 Russell US reconstitution (paper only; no orders).

Usage
  python3 predict.py                         # fetch everything today, predict on the latest close
  python3 predict.py --asof 2026-10-30       # rank-day run (run after ~18:00 ET so the close is in)
  python3 predict.py --asof 2026-10-30 --no-fetch --fetch-date 2026-10-30   # re-predict from saved raw data
  python3 predict.py --info-from 2026-10-09  # reuse an earlier Yahoo shares pull (fast; shares go stale)

Method (adapted from 12_studies/russell_june/s03_predict.py rebuild() / known_excluded()):
  membership  iShares holdings: IWB = R1000, IWM = R2000 (IWV, sampled, is a cross-check only)
  universe    nasdaqtrader symbol directories: Nasdaq, NYSE, NYSE American, NYSE Arca, Cboe BZX, TXSE;
              no ETFs/test issues/preferreds/warrants/units/rights/notes. Non-members only: no ADRs, LPs,
              LLCs, funds/BDCs/SPACs (name + Yahoo industry 'Shell Companies'), Yahoo country must be US,
              Yahoo float/total shares >= 5%. Members are exempt from the country/structure/float screens
              (Dec recon: no voting re-test, float only moves with 3% buffers).
  price/size  close >= $1 on the as-of date (members: 30-calendar-day average >= $1 also passes);
              company total cap >= $30m. Total cap = Yahoo impliedSharesOutstanding (all classes incl.
              unlisted / exchangeable units) x close of the pricing vehicle (class with the highest $ volume).
  known excl. non-members that our rebuild on the June rank day (30 Apr 2026 close, current shares) ranks
              1..2700 and that were listed then, yet are not in the index now -> Russell-ineligible.
  rank/band   rank by total cap; top 4000 = R3000E; cumulative-cap percentile over the top 4000;
              R1000 = #1-1000 with +/-2.5% band for existing R3000 members; R2000 = #1001-3000 (no band at
              #3000). The +/-0.5% band at #2000 is computed and reported but only affects the Microcap.
  confidence  margin = log distance of company cap to the edge that would flip its category;
              p_hold = Phi(margin / sqrt(sigma_rel^2 * h + sigma_data^2)), h = trading days to rank day.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C
import fetch as F

T0 = time.time()
TIMES = {}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def lap(name, t):
    TIMES[name] = round(time.time() - t, 1)


# ------------------------------------------------------------------ security-level screens
HARD_SYM = re.compile(r"[\$\^]|\.(?:U|W|WS|WT|R|RT|WI|V)$")
HARD_NAME = re.compile(r"warrant|\brights?\b|preferred|\bpfd\b|preference|depositary shares? (?:each )?represent(?:ing)? .*(?:preferred|interest)|"
                       r"\bnotes?\b|debenture|\bbonds?\b|\bunits?\b,? each|\bunits?\b.*consist|\bETN\b|exchange[- ]traded|"
                       r"cumulative|perpetual|subordinated|capital securities|\bwhen[- ]issued\b|contingent value", re.I)
ADR_NAME = re.compile(r"american depositary|\bADS\b|\bADR\b|depositary (?:share|receipt)|global depositary", re.I)
LP_NAME = re.compile(r"\bL\.\s?P\.|\bLP\b|limited partnership|common units|partnership units|\bunits representing", re.I)
LLC_NAME = re.compile(r"\bL\.?L\.?C\.?\b", re.I)
FUND_NAME = re.compile(r"\bfunds?\b|municipal|closed[- ]end|royalty trust|trust units?\b|units of beneficial|grantor trust|"
                       r"bitcoin|ethereum|\bether\b|\bcrypto|\bgold trust|\bsilver trust|commodity|\bBDC\b|"
                       r"income (?:trust|shares)|opportunities trust|investment trust\b(?!.*real)", re.I)
SPAC_NAME = re.compile(r"acquisition (?:corp|corporation|co\b|company|holdings)|blank check|\bSPAC\b", re.I)
BDC_HINT = re.compile(r"capital corp|lending|credit (?:corp|fund|company)|investment corp|specialty finance|private credit|"
                      r"floating rate|senior (?:secured|loan)|\bdebt\b", re.I)
ELIG_OTHER_EXCH = {"N", "A", "P", "Z", "F"}       # otherlisted: NYSE, NYSE American, NYSE Arca, Cboe BZX, TXSE


def security_universe(sd: pd.DataFrame, members: set) -> pd.DataFrame:
    sd = sd[(sd.test != "Y") & (sd.etf != "Y")].copy()
    sd = sd[(sd.src == "nasdaqlisted") | sd.exch.isin(ELIG_OTHER_EXCH)]
    sd["member_sec"] = sd.yf.isin(members)
    nas5 = (sd.src == "nasdaqlisted") & (sd.sym.str.len() == 5) & sd.sym.str[-1].isin(list("WRU")) & \
        sd.name.str.contains(r"warrant|right|unit", case=False)
    hard = sd.sym.str.contains(HARD_SYM) | sd.name.str.contains(HARD_NAME) | nas5
    sd["hard_excl"] = hard & ~sd.member_sec
    sd = sd[~sd.hard_excl].copy()
    sd["f_adr"] = sd.name.str.contains(ADR_NAME)
    sd["f_lp"] = sd.name.str.contains(LP_NAME)
    sd["f_llc"] = sd.name.str.contains(LLC_NAME)
    sd["f_fund"] = sd.name.str.contains(FUND_NAME)
    sd["f_spac"] = sd.name.str.contains(SPAC_NAME)
    return sd


# ------------------------------------------------------------------ union-find for share classes
class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def first_word(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    w = [x for x in s.split() if x not in {"the"}]
    return w[0] if w else ""


def group_classes(S: pd.DataFrame) -> pd.Series:
    uf = UF(S.yf.tolist())
    # (a) same root before the class suffix (BRK-A/BRK-B, HEI/HEI-A, MOG-A/MOG-B) and same first word of name
    S = S.assign(root=S.yf.str.split("-").str[0], fw=S.name.map(first_word),
                 fwy=S.shortName.fillna(S.name).map(first_word))
    for _, g in S[S.yf.str.contains("-") | S.root.duplicated(keep=False)].groupby("root"):
        if len(g) > 1:
            base = g.iloc[0]
            for _, r in g.iloc[1:].iterrows():
                if r.fw == base.fw or r.fwy == base.fwy:
                    uf.u(base.yf, r.yf)
    # (b) identical Yahoo impliedSharesOutstanding (company-level count) and the same first word
    imp = S[S.impliedSharesOutstanding.notna() & (S.impliedSharesOutstanding > 0)]
    for _, g in imp.groupby("impliedSharesOutstanding"):
        if len(g) > 1:
            for fw, gg in g.groupby("fwy"):
                if len(gg) > 1:
                    for t in gg.yf.iloc[1:]:
                        uf.u(gg.yf.iloc[0], t)
    # (c) Nasdaq screener company-level share counts within 0.1% and the same first two words of the listing name
    S["fw2"] = S.name.map(lambda x: " ".join(re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split()[:2]))
    nu = S[S.nd_units.notna() & (S.nd_units > 0)].sort_values("nd_units")
    v, ids, k2 = nu.nd_units.values, nu.yf.values, nu.fw2.values
    for i in range(1, len(nu)):
        if v[i] / v[i - 1] - 1 < 0.001 and k2[i] == k2[i - 1]:
            uf.u(ids[i - 1], ids[i])
    return S.yf.map(uf.f)


# ------------------------------------------------------------------ main build
def build(args):
    fetch_date = args.fetch_date
    rawdir = C.RAW / fetch_date
    rawdir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    if not args.no_fetch:
        log(f"fetching holdings / symbol directories / screener -> {rawdir}")
        F.fetch_holdings(rawdir)
        F.fetch_symdir(rawdir)
        try:
            F.fetch_screener(rawdir)
        except RuntimeError as e:
            log("  nasdaq screener failed (cross-check only):", e)
    lap("fetch_lists", t)

    H, h_asof = F.load_holdings(rawdir)
    live = H[~H.residual]
    r1 = set(live.loc[live.fund == "IWB", "yf"])
    r2 = set(live.loc[live.fund == "IWM", "yf"]) - r1
    iwv = set(live.loc[live.fund == "IWV", "yf"])
    iwv_only = iwv - r1 - r2          # in the (sampled) R3000 fund but in neither IWB nor IWM: treat as R2000
    r2 = r2 | iwv_only                # (IWB replicates the R1000 fully; IWM skips a few small names)
    members = r1 | r2
    log(f"holdings as of {', '.join(f'{k} {v:%Y-%m-%d}' for k, v in h_asof.items())}: "
        f"R1000 securities {len(r1)}, R2000 {len(r2)}, IWV {len(iwv)}; residual rows dropped {int(H.residual.sum())}")

    sd = F.load_symdir(rawdir)
    U = security_universe(sd, members)
    miss_mem = sorted(members - set(U.yf))
    if miss_mem:   # members not on an eligible listing today (OTC / delisted / symbol change)
        extra = live[live.yf.isin(miss_mem)].drop_duplicates("yf")
        U = pd.concat([U, pd.DataFrame({"sym": extra.sym, "name": extra.Name, "exch": "?", "src": "holdings_only",
                                        "yf": extra.yf, "member_sec": True})], ignore_index=True)
    U["listed_ok"] = U.src != "holdings_only"
    for c in ["f_adr", "f_lp", "f_llc", "f_fund", "f_spac"]:
        U[c] = U[c].fillna(False).astype(bool)
    log(f"universe: {len(U)} securities after listing screens ({len(miss_mem)} member securities not on an eligible listing)")

    nd = F.load_screener(rawdir)
    U = U.merge(nd, on="yf", how="left")
    U["nd_units"] = U.nd_cap / U.nd_px.where(U.nd_px > 0)

    # ---- Yahoo info (throttled; resumable). Members: all. Non-members: skip names the Nasdaq screener puts
    # below $20m (they cannot reach $30m by rank day without a +50% move; re-checked on every run).
    t = time.time()
    need = U.member_sec | U.nd_cap.isna() | (U.nd_cap >= 20e6)
    # priority: non-members first (their screens need Yahoo), then members closest to the #1000 / #3000 edges
    mc = U.loc[U.member_sec & U.nd_cap.notna(), "nd_cap"].drop_duplicates().sort_values(ascending=False)
    c1000, c3000 = (mc.iloc[min(999, len(mc) - 1)], mc.iloc[max(len(mc) - 60, 0)]) if len(mc) else (5e9, 1.5e8)
    lc = np.log(U.nd_cap.clip(lower=1e6))
    U["prio"] = np.where(~U.member_sec, -1.0, np.fmin(np.abs(lc - np.log(c1000)), np.abs(lc - np.log(c3000))).fillna(0))
    order = U[need].sort_values("prio", kind="stable").yf.tolist()
    I = F.load_info(C.RAW / args.info_from) if args.info_from else F.load_info(rawdir)
    if args.info_from:
        log(f"reusing Yahoo info from raw/{args.info_from} ({len(I)} securities; shares may be stale)")
    if not args.no_fetch:
        todo = [x for x in order if x not in set(I.t)]
        if todo:
            I2 = F.fetch_info(todo, rawdir, workers=args.workers, max_minutes=args.info_minutes, log=log)
            I = pd.concat([I, I2], ignore_index=True).drop_duplicates("t", keep="last")
    lap("yahoo_info", t)
    I = I.rename(columns={"t": "yf"})
    U = U.merge(I[[c for c in I.columns if c in ["yf"] + F.INFO_FIELDS]], on="yf", how="left")
    U["has_info"] = U.quoteType.notna()
    log(f"  yahoo info ok for {U.has_info.sum()}/{len(U)} securities ({int(need.sum())} requested; "
        f"members {U[U.member_sec].has_info.sum()}/{U.member_sec.sum()})")
    tick = U.yf.tolist()

    # ---- prices
    t = time.time()
    asof_req = pd.Timestamp(args.asof) if args.asof else None
    end = asof_req or pd.Timestamp(dt.date.today())
    start = min(C.JUNE_RANK_DAY - pd.Timedelta(days=45), end - pd.Timedelta(days=150))
    if args.no_fetch:
        P = F.load_prices(rawdir, "predict")
    else:
        log(f"downloading daily prices {start:%Y-%m-%d}..{end:%Y-%m-%d} for {len(tick) + 1} symbols")
        P = F.fetch_prices(tick + ["IWM"], start, end, rawdir, "predict", log=log)
    lap("prices", t)
    close, adj, vol = P["Close"], P["Adj Close"], P["Volume"]
    cov = close.notna().mean(axis=1)
    if asof_req is not None:
        if asof_req not in close.index:
            sys.exit(f"no closes for {asof_req:%Y-%m-%d} in the price download (holiday, or close not published yet)")
        asof = asof_req
    else:
        asof = cov[cov > 0.5].index.max()
    if cov.get(asof, 0) < 0.5:
        sys.exit(f"price coverage on {asof:%Y-%m-%d} only {cov.get(asof, 0):.0%}: close not complete yet?")
    close, adj, vol = close.loc[:asof], adj.loc[:asof], vol.loc[:asof]
    log(f"as-of close {asof:%Y-%m-%d}; price coverage {cov[asof]:.1%}")

    px = close.ffill(limit=3).iloc[-1]                     # allow a stale print for halted names
    px_fresh = close.iloc[-1].notna()
    px30 = close.loc[asof - pd.Timedelta(days=30):].mean()
    jr = close.index[close.index <= C.JUNE_RANK_DAY].max()
    px_jun = close.loc[jr]
    dv = (close * vol).iloc[-60:].mean()
    lr = np.log(close).diff().iloc[-C.VOL_WINDOW:]
    sig_abs = lr.std()
    sig_rel = lr.sub(lr["IWM"], axis=0).std() if "IWM" in lr else sig_abs
    nobs = lr.notna().sum()
    U["px"] = U.yf.map(px)
    U["px_fresh"] = U.yf.map(px_fresh).fillna(False)
    U["px30"] = U.yf.map(px30)
    U["px_jun"] = U.yf.map(px_jun)
    U["adv_usd"] = U.yf.map(dv)
    U["sig_abs"] = U.yf.map(sig_abs)
    U["sig_rel"] = U.yf.map(sig_rel)
    U["nobs"] = U.yf.map(nobs).fillna(0)
    U["adj_asof"] = U.yf.map(adj.iloc[-1])

    # ---- companies (share classes) ----
    U["shortName"] = U.shortName.where(U.shortName.notna(), U.name)
    U["cid"] = group_classes(U)
    U["cur_sec"] = np.where(U.yf.isin(r1), "R1000", np.where(U.yf.isin(r2), "R2000", "out"))
    U["sec_cap_listed"] = U.sharesOutstanding * U.px
    U["first_trade"] = pd.to_datetime(U.firstTradeDateMilliseconds, unit="ms", errors="coerce")

    rows = []
    for cid, g in U.groupby("cid", sort=False):
        gp = g[g.px.notna()]
        pvr = (gp.sort_values("adv_usd", ascending=False).iloc[0] if len(gp) else g.iloc[0])
        imp, so_pv, ndu = pvr.impliedSharesOutstanding, pvr.sharesOutstanding, pvr.nd_units
        flags = []
        n_cls = len(g)
        if n_cls > 1:
            flags.append("multi_class")
        if not g.has_info.any():
            flags.append("no_yahoo_info")
        # total shares of all classes, in pricing-vehicle units (Russell: all classes incl. unlisted x PV price)
        units, src = np.nan, None
        nd_other = (gp.nd_units * gp.px).max() / pvr.px if len(gp) and gp.nd_units.notna().any() and pd.notna(pvr.px) else np.nan
        if pd.notna(imp) and imp > 0 and (pd.isna(so_pv) or 0.5 <= imp / so_pv <= 20):
            units, src = imp, "yahoo_implied"
        elif pd.notna(ndu) and ndu > 0:
            units, src = ndu, "nasdaq_screener"
        elif pd.notna(nd_other) and nd_other > 0:
            units, src = nd_other, "nasdaq_other_class"
        elif n_cls > 1 and gp.sharesOutstanding.notna().all() and len(gp) == n_cls:
            units, src = (gp.sharesOutstanding * gp.px).sum() / pvr.px, "yahoo_sum_classes"
        elif pd.notna(so_pv) and so_pv > 0:
            units, src = so_pv, "yahoo_shares_out"
        if src not in (None, "yahoo_implied"):
            flags.append(f"shares_{src}")
        if pd.notna(imp) and src != "yahoo_implied":
            flags.append("implied_rejected")
        nd_ref = ndu if pd.notna(ndu) else nd_other
        if src == "yahoo_implied" and pd.notna(nd_ref) and nd_ref > 0 and not (0.8 <= units / nd_ref <= 1.25):
            flags.append("shares_mismatch_yahoo_nasdaq")
        cap = units * pvr.px if pd.notna(units) and pd.notna(pvr.px) else np.nan
        cap_jun = units * pvr.px_jun if pd.notna(units) and pd.notna(pvr.px_jun) else np.nan
        listed_sum = gp.sec_cap_listed.sum(min_count=1)
        if pd.notna(cap) and pd.notna(listed_sum) and listed_sum > 0 and cap / listed_sum > 1.05:
            flags.append("unlisted_or_exch_units")       # all-class count > listed classes (unlisted class / Up-C units)
        fl = pvr.floatShares
        float_ratio = fl / units if pd.notna(fl) and pd.notna(units) and units > 0 else np.nan
        cur = "R1000" if (g.cur_sec == "R1000").any() else ("R2000" if (g.cur_sec == "R2000").any() else "out")
        country = pvr.country if isinstance(pvr.country, str) and pvr.country else (
            pvr.nd_country if isinstance(pvr.nd_country, str) else None)
        rows.append(dict(
            cid=cid, pv=pvr.yf, tickers=",".join(g.yf), n_classes=n_cls, name=pvr.shortName, cur=cur,
            in_iwv=bool(g.yf.isin(iwv).any()), iwv_only=bool(g.yf.isin(iwv_only).any()), listed_ok=bool(g.listed_ok.any()),
            px=pvr.px, px_fresh=bool(pvr.px_fresh), px30=gp.px30.max() if len(gp) else np.nan, px_jun=pvr.px_jun,
            adj_asof=pvr.adj_asof, cap=cap, cap_jun=cap_jun, cap_listed_sum=listed_sum,
            nd_cap=(nd_ref * pvr.px if pd.notna(nd_ref) else np.nan), shares_total=units, shares_src=src,
            float_ratio=float_ratio, country=country, quoteType=pvr.quoteType, industry=pvr.industry,
            first_trade=g.first_trade.min(), adv_usd=pvr.adv_usd, sig_abs=pvr.sig_abs, sig_rel=pvr.sig_rel, nobs=pvr.nobs,
            f_adr=bool(g.f_adr.all()), f_lp=bool(g.f_lp.any()), f_llc=bool(g.f_llc.any()),
            f_fund=bool(g.f_fund.any()), f_spac=bool(g.f_spac.any()) or pvr.industry == "Shell Companies",
            f_bdc=bool(pvr.industry == "Asset Management" and bool(BDC_HINT.search(str(pvr.shortName) + " " + str(pvr["name"])))),
            flags=";".join(flags)))
    Co = pd.DataFrame(rows)
    Co["member"] = Co.cur != "out"
    log(f"companies: {len(Co)} ({(Co.n_classes > 1).sum()} multi-class); members {Co.member.sum()} "
        f"(R1000 {(Co.cur == 'R1000').sum()}, R2000 {(Co.cur == 'R2000').sum()})")
    return dict(Co=Co, U=U, H=H, h_asof=h_asof, asof=asof, rawdir=rawdir, r1=r1, r2=r2, iwv=iwv, iwv_only=iwv_only)


def screens(Co: pd.DataFrame, which: str) -> pd.DataFrame:
    """Eligibility. which='now' uses as-of price/cap; 'jun' uses the 30 Apr 2026 close (for known-excluded)."""
    px = Co.px if which == "now" else Co.px_jun
    cap = Co.cap if which == "now" else Co.cap_jun
    m = Co.member
    reason = pd.Series("", index=Co.index)

    def mark(mask, why):
        nonlocal reason
        reason = reason.where(~(mask & (reason == "")), why)

    mark(~Co.listed_ok, "not_on_eligible_listing")
    mark(cap.isna() | px.isna(), "no_price_or_shares")
    # non-member structure / country / float screens (members exempt)
    mark(~m & Co.quoteType.notna() & (Co.quoteType != "EQUITY"), "not_equity")
    mark(~m & Co.f_adr, "adr")
    mark(~m & (Co.f_lp | Co.f_llc), "lp_llc")
    mark(~m & (Co.f_fund | Co.f_bdc), "fund_bdc_trust")
    mark(~m & Co.f_spac, "spac_shell")
    mark(~m & Co.country.notna() & (Co.country != "United States"), "non_us")
    mark(~m & Co.float_ratio.notna() & (Co.float_ratio < C.MIN_FLOAT), "float_lt_5pct")
    # price and size (members: 30-day average >= $1 also passes; only for the live as-of date)
    ok_px = (px >= C.MIN_PRICE) | (m & (Co.px30 >= C.MIN_PRICE) if which == "now" else False)
    mark(~ok_px, "price_lt_1")
    mark(cap < C.MIN_CAP, "cap_lt_30m")
    return reason


def rank_and_band(E: pd.DataFrame, capcol="cap") -> tuple[pd.DataFrame, dict]:
    E = E.sort_values(capcol, ascending=False).reset_index(drop=True)
    E["rank"] = np.arange(1, len(E) + 1)
    top = E["rank"] <= C.N_R3000E
    E["cumpct"] = np.where(top, E[capcol].where(top).cumsum() / E.loc[top, capcol].sum() * 100, np.nan)
    bp = E.loc[E["rank"] == C.N_R1000, "cumpct"].iloc[0]
    lo, hi = bp - C.BAND_1000, bp + C.BAND_1000
    bp2 = E.loc[E["rank"] == 2000, "cumpct"].iloc[0]

    def cap_at_pct(p):
        return E.loc[E.cumpct >= p, capcol].iloc[0]
    br = dict(bp_pct=bp, band_lo=lo, band_hi=hi, cap1000=E.loc[E["rank"] == 1000, capcol].iloc[0],
              cap3000=E.loc[E["rank"] == 3000, capcol].iloc[0], cap4000=E.loc[E["rank"] == min(4000, len(E)), capcol].iloc[0],
              capB_up=cap_at_pct(lo), capB_dn=cap_at_pct(hi),
              bp2000_pct=bp2, band2000_lo=bp2 - C.BAND_2000, band2000_hi=bp2 + C.BAND_2000,
              cap2000=E.loc[E["rank"] == 2000, capcol].iloc[0],
              cap2000_up=cap_at_pct(bp2 - C.BAND_2000), cap2000_dn=cap_at_pct(bp2 + C.BAND_2000),
              n_ranked=len(E))
    E["pred"] = np.where(E["rank"] <= C.N_R1000, "R1000", np.where(E["rank"] <= C.N_R3000, "R2000", "out"))
    inband = E.cur.isin(["R1000", "R2000"]) & E.cumpct.between(lo, hi)
    E.loc[inband, "pred"] = E.loc[inband, "cur"]
    E["in_band1000"] = inband
    return E, br


def classify(cur, pred):
    return np.select([(cur == "out") & (pred == "R2000"), (cur == "R1000") & (pred == "R2000"),
                      (cur == "R2000") & (pred == "out"), (cur == "R2000") & (pred == "R1000"),
                      (cur == "out") & (pred == "R1000"), (cur == "R1000") & (pred == "out")],
                     C.CATS, "none")


def margins(E: pd.DataFrame, br: dict, h: int) -> pd.DataFrame:
    """Signed log distance to the edge that would flip the predicted category (+ = safely on the predicted side)."""
    c = E.cap
    L = lambda a, b: np.log(a / b)
    cand = {   # (cur, pred): [(edge name, margin)]
        ("out", "R1000"): [("#1000", L(c, br["cap1000"]))],
        ("out", "R2000"): [("#3000", L(c, br["cap3000"])), ("#1000", L(br["cap1000"], c))],
        ("out", "out"): [("#3000", L(br["cap3000"], c))],
        ("R2000", "R2000"): [("#3000", L(c, br["cap3000"])), ("band_up(#1000-2.5%)", L(br["capB_up"], c))],
        ("R2000", "out"): [("#3000", L(br["cap3000"], c))],
        ("R2000", "R1000"): [("band_up(#1000-2.5%)", L(c, br["capB_up"]))],
        ("R1000", "R1000"): [("band_dn(#1000+2.5%)", L(c, br["capB_dn"]))],
        ("R1000", "R2000"): [("band_dn(#1000+2.5%)", L(br["capB_dn"], c)), ("#3000", L(c, br["cap3000"]))],
        ("R1000", "out"): [("#3000", L(br["cap3000"], c))],
    }
    edge = pd.Series(None, index=E.index, dtype=object)
    marg = pd.Series(np.nan, index=E.index)
    for (cu, pr), lst in cand.items():
        m = (E.cur == cu) & (E.pred == pr)
        if not m.any():
            continue
        M = pd.concat([s[m] for _, s in lst], axis=1)
        M.columns = [n for n, _ in lst]
        marg[m] = M.min(axis=1)
        edge[m] = M.idxmin(axis=1)
    E["edge"], E["margin_log"] = edge, marg
    edge_cap = {"#1000": br["cap1000"], "#3000": br["cap3000"], "band_up(#1000-2.5%)": br["capB_up"],
                "band_dn(#1000+2.5%)": br["capB_dn"]}
    E["edge_cap"] = E.edge.map(edge_cap)
    E["sigma_data"] = np.where(E["flags"].str.contains("multi_class|unlisted|shares_|implied_rejected|mismatch|no_yahoo"),
                               C.SIGMA_DATA_FLAG, C.SIGMA_DATA)
    return E


def finish_probs(A: pd.DataFrame, h: int) -> pd.DataFrame:
    srel = A.sig_rel.where(A.nobs >= 20)
    sabs = A.sig_abs.where(A.nobs >= 20)
    srel = srel.fillna(srel.median())
    sabs = sabs.fillna(sabs.median())
    rule = A.edge.isin(["$1 price", "$30m cap"])
    s = np.where(rule, sabs, srel)
    sd = np.where(rule, 0.01, A.sigma_data)
    A["sigma_tot"] = np.sqrt(s ** 2 * h + sd ** 2)
    A["p_hold"] = norm.cdf(A.margin_log / A.sigma_tot)
    A.loc[A.margin_log == np.inf, "p_hold"] = 1.0
    A["margin_pct"] = np.exp(A.margin_log) - 1          # relative cap move vs the edge that flips the call
    return A


def archive_old(p: Path) -> Path:
    """Never overwrite a logged prediction: move an existing file to predictions/superseded/ first."""
    if p.exists():
        d = C.PRED / "superseded"
        d.mkdir(exist_ok=True)
        ts = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%dT%H%M%S")
        p.rename(d / f"{p.stem}__{ts}{p.suffix}")
    return p


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", help="close date to rank on (YYYY-MM-DD); default = latest complete close")
    ap.add_argument("--fetch-date", default=dt.date.today().isoformat(), help="raw/<fetch-date>/ folder")
    ap.add_argument("--no-fetch", action="store_true", help="use files already in raw/<fetch-date>/")
    ap.add_argument("--info-from", help="reuse raw/<date>/yf_info.jsonl from an earlier fetch (stale shares)")
    ap.add_argument("--workers", type=int, default=4, help="Yahoo info threads (shared throttle, >= 0.4 s apart)")
    ap.add_argument("--info-minutes", type=float, default=45, help="time budget for the Yahoo info pull")
    ap.add_argument("--tag", default="", help="label for the log, e.g. rankday / pre-prelim")
    args = ap.parse_args()
    run_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)

    B = build(args)
    Co, asof, rawdir = B["Co"], B["asof"], B["rawdir"]
    t = time.time()

    # ---- known excluded: June-2026 rank-day rebuild with current shares ----
    rj = screens(Co, "jun")
    Ej, brj = rank_and_band(Co[rj == ""].copy(), "cap_jun")
    excl = Ej[(Ej["rank"] >= C.EXCL_MINRANK) & (Ej["rank"] <= C.EXCL_MAXRANK) & ~Ej.member]
    Co["rank_jun"] = Co.cid.map(Ej.set_index("cid")["rank"])
    Co["known_excl"] = Co.cid.isin(set(excl.cid))
    log(f"known-excluded (ranked {C.EXCL_MINRANK}-{C.EXCL_MAXRANK} on 30 Apr 2026, listed, not in index): {len(excl)}")

    # ---- live screens and ranking ----
    reason = screens(Co, "now")
    reason = reason.where(~((reason == "") & Co.known_excl), "known_excluded_jun26")
    Co["excl_reason"] = reason
    E, br = rank_and_band(Co[reason == ""].copy(), "cap")
    h = C.trading_days_between(asof, C.RANK_DAY)
    E = margins(E, br, h)

    # members that fail the screens -> predicted out (backtest: 'gone' members)
    G = Co[(reason != "") & Co.member].copy()
    G["pred"] = "out"
    G["rank"] = np.nan
    G["edge"] = None
    G["margin_log"] = np.nan
    px_eff = np.fmax(G.px, G.px30.fillna(0))
    pr = G.excl_reason == "price_lt_1"
    G.loc[pr, "edge"], G.loc[pr, "margin_log"] = "$1 price", np.log(C.MIN_PRICE / px_eff[pr])
    cp = G.excl_reason == "cap_lt_30m"
    G.loc[cp, "edge"], G.loc[cp, "margin_log"] = "$30m cap", np.log(C.MIN_CAP / G.cap[cp])
    gone_unknown = G.excl_reason.isin(["no_price_or_shares"])
    G.loc[G.excl_reason == "not_on_eligible_listing", ["edge", "margin_log"]] = ["not listed (OTC/delisted)", np.inf]
    G["sigma_data"] = C.SIGMA_DATA
    G.loc[gone_unknown, "pred"] = "unknown"           # data gap: no prediction rather than a false delete
    A = pd.concat([E, G], ignore_index=True)
    A["category"] = classify(A.cur, A.pred)
    A.loc[A.pred == "unknown", "category"] = "data_gap"
    A = finish_probs(A, h)
    A["backtest_group"] = A.category.map(C.BACKTEST_GROUP).fillna("")
    A["side"] = A.category.map(C.SIDE).fillna(0).astype(int)
    A["eligible"] = A.excl_reason == ""
    A["asof"] = asof.date()
    lap("predict", t)

    # ---- outputs ----
    t = time.time()
    stamp = f"{asof:%Y-%m-%d}" + (f"_{args.tag}" if args.tag else "")
    keep = ["asof", "cid", "pv", "tickers", "name", "cur", "pred", "category", "backtest_group", "side", "rank",
            "cumpct", "in_band1000", "cap", "edge", "edge_cap", "margin_log", "margin_pct", "sigma_rel", "sigma_tot",
            "p_hold", "px", "px30", "adj_asof", "px_fresh", "adv_usd", "shares_total", "shares_src", "cap_listed_sum",
            "nd_cap", "float_ratio", "n_classes", "flags", "country", "industry", "quoteType", "first_trade",
            "rank_jun", "cap_jun", "in_iwv", "eligible", "excl_reason"]
    A = A.rename(columns={"sig_rel": "sigma_rel"})
    A = A.sort_values(["category", "p_hold", "margin_log"], ascending=[True, False, False])
    f_all = archive_old(C.PRED / f"pred_{stamp}_all.csv")
    A[keep].to_csv(f_all, index=False, float_format="%.6g")
    ch = A[A.category.isin(C.CATS)].copy()
    ch["cat_order"] = ch.category.map({c: i for i, c in enumerate(C.CATS)})
    ch = ch.sort_values(["cat_order", "p_hold", "margin_log"], ascending=[True, False, False]).drop(columns="cat_order")
    f_ch = archive_old(C.PRED / f"pred_{stamp}_changes.csv")
    ch[keep].to_csv(f_ch, index=False, float_format="%.6g")
    # paper portfolio: pricing-vehicle line per predicted change, equal weight within category
    pp = ch[["asof", "category", "backtest_group", "side", "pv", "name", "px", "adj_asof", "p_hold", "margin_pct"]].copy()
    pp["weight_in_cat"] = 1 / pp.groupby("category").pv.transform("count")
    f_pp = archive_old(C.PRED / f"paper_portfolio_{stamp}.csv")
    pp.to_csv(f_pp, index=False, float_format="%.6g")
    # watchlist: unchanged names within 15% of a flip edge
    wl = A[(A.category == "none") & (A.margin_log < np.log(1.15)) & A.cur.isin(["R1000", "R2000", "out"])
           & A["rank"].between(800, 3400)].sort_values("margin_log")
    f_wl = archive_old(C.PRED / f"watchlist_{stamp}.csv")
    wl[keep].to_csv(f_wl, index=False, float_format="%.6g")

    counts = {c: int((A.category == c).sum()) for c in C.CATS}
    exp_counts = {c: round(float(A.loc[A.category == c, "p_hold"].sum()), 1) for c in C.CATS}
    U, H = B["U"], B["H"]
    mem = A[A.cur != "out"]
    dq = dict(
        securities_in_universe=int(len(U)), yahoo_info_ok=int(U.quoteType.notna().sum()),
        member_companies=int(Co.member.sum()), member_companies_with_cap=int(Co.loc[Co.member, "cap"].notna().sum()),
        member_securities_not_listed=int((~U.listed_ok).sum()),
        companies_total=int(len(Co)), companies_with_cap=int(Co.cap.notna().sum()),
        shares_src=Co.shares_src.fillna("missing").value_counts().to_dict(),
        shares_src_members=Co.loc[Co.member, "shares_src"].fillna("missing").value_counts().to_dict(),
        multi_class_companies=int((Co.n_classes > 1).sum()),
        multi_class_examples=Co.loc[Co.n_classes > 1].sort_values("cap", ascending=False).tickers.head(25).tolist(),
        unlisted_or_exch_units=int(Co["flags"].str.contains("unlisted").sum()),
        shares_mismatch_yahoo_nasdaq=int(Co["flags"].str.contains("shares_mismatch").sum()),
        member_shares_mismatch_yahoo_nasdaq=int(Co.loc[Co.member, "flags"].str.contains("shares_mismatch").sum()),
        no_yahoo_info_companies=int(Co["flags"].str.contains("no_yahoo_info").sum()),
        price_stale_or_missing_members=int((~mem.px_fresh.astype(bool)).sum()),
        data_gap_members=int((A.category == "data_gap").sum()),
        residual_holdings_dropped=int(H.residual.sum()),
        iwv_only_treated_as_r2000=sorted(B["iwv_only"]),
        excl_reasons=Co.excl_reason.replace("", "eligible").value_counts().to_dict(),
        known_excluded=int(Co.known_excl.sum()),
    )
    summ = dict(asof=str(asof.date()), run_utc=run_utc.isoformat(), tag=args.tag, fetch_date=args.fetch_date,
                holdings_asof={k: str(v.date()) for k, v in B["h_asof"].items()},
                trading_days_to_rank_day=h, counts=counts, expected_counts_sum_p=exp_counts,
                breakpoints={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in br.items()},
                breakpoints_jun_rebuild={k: float(v) for k, v in brj.items()},
                data_quality=dq, timings_s=TIMES)
    f_sum = archive_old(C.PRED / f"summary_{stamp}.json")
    lap("write", t)
    TIMES["total"] = round(time.time() - T0, 1)
    f_sum.write_text(json.dumps(summ, indent=1, default=str))

    # ---- shadow log (append) ----
    files = [f_all, f_ch, f_pp, f_wl, f_sum]
    raw_files = sorted(p for p in rawdir.iterdir() if p.is_file())
    L = [f"\n## Run {run_utc:%Y-%m-%d %H:%M:%S} UTC — as-of close {asof:%Y-%m-%d}" + (f" ({args.tag})" if args.tag else ""),
         "",
         f"- Command: `python3 predict.py {' '.join(sys.argv[1:])}`; wall time {TIMES['total']:.0f}s "
         f"({', '.join(f'{k} {v:.0f}s' for k, v in TIMES.items() if k != 'total')}).",
         f"- Holdings as of: {', '.join(f'{k} {v:%Y-%m-%d}' for k, v in B['h_asof'].items())}. Raw inputs: `raw/{args.fetch_date}/` "
         f"({len(raw_files)} files). Trading days to rank day (30 Oct): {h}.",
         f"- Breakpoints: #1000 cap ${br['cap1000'] / 1e9:.2f}bn (cum. {br['bp_pct']:.2f}%), band {br['band_lo']:.2f}–{br['band_hi']:.2f}% "
         f"= ${br['capB_dn'] / 1e9:.2f}bn–${br['capB_up'] / 1e9:.2f}bn; #3000 cap ${br['cap3000'] / 1e6:.0f}m; "
         f"#2000 ${br['cap2000'] / 1e6:.0f}m (±0.5% band ${br['cap2000_dn'] / 1e6:.0f}m–${br['cap2000_up'] / 1e6:.0f}m, Microcap only).",
         "",
         "| Category | Predicted | Sum of p_hold |", "|---|---|---|"]
    L += [f"| {C.CAT_LABEL[c]} | {counts[c]} | {exp_counts[c]} |" for c in C.CATS]
    L += ["", "| File | SHA-256 |", "|---|---|"]
    L += [f"| `predictions/{p.name}` | `{sha(p)}` |" for p in files]
    L += ["",
          f"- Data quality: Yahoo info for {dq['yahoo_info_ok']}/{dq['securities_in_universe']} securities; member companies with a cap "
          f"{dq['member_companies_with_cap']}/{dq['member_companies']}; shares source {dq['shares_src']}; "
          f"{dq['multi_class_companies']} multi-class companies; {dq['unlisted_or_exch_units']} with unlisted classes/exchangeable units; "
          f"{dq['shares_mismatch_yahoo_nasdaq']} with Yahoo vs Nasdaq share counts >25% apart ({dq['member_shares_mismatch_yahoo_nasdaq']} members); {dq['data_gap_members']} members left unpredicted (data gap); "
          f"{dq['known_excluded']} non-members treated as Russell-ineligible from the June rebuild.",
          "- Known error sources: shares are Yahoo's current counts (not FTSE's rank-day shares, not point-in-time); "
          "FTSE float/voting/HCI country unknown (Yahoo country/float used for non-members only); "
          "share-class grouping by ticker root and identical Yahoo implied shares; iShares holdings lag Russell by a day and include residual "
          "lines (dropped below $1,000 or unlisted); IWV is sampled; pending M&A deletions, IPOs after this run, spin-offs and FTSE query-period "
          "changes are not modelled; prices move until the 30 Oct rank day (see p_hold).",
          ]
    new_log = not C.LOG.exists() or C.LOG.stat().st_size == 0
    with open(C.LOG, "a") as fh:
        if new_log:
            fh.write("# Shadow log — December 2026 Russell US reconstitution (paper only)\n\n"
                     "Append-only. Each run adds a block; never edit earlier blocks. Scores from `score.py` are appended too.\n")
        fh.write("\n".join(L) + "\n")
    log(f"done in {TIMES['total']:.0f}s; counts {counts}")
    for p in files:
        log(f"  {p.relative_to(C.HERE)}  {sha(p)[:16]}")


if __name__ == "__main__":
    main()
