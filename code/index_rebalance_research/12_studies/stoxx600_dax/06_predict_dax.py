"""Step 6: point-in-time predictions of DAX / MDAX / SDAX / TecDAX review changes at H6, H3 and the cut-off C.

Shadow selection list at horizon date d (data dated <= d only), rules by era from common.dax_rules:
  universe   companies with a German home listing, plus every company that was ever on a Datastream DAX-family
             list (foreign members such as Airbus or Qiagen; this inclusion uses hindsight and is flagged)
  rank       free-float market cap = 20-session mean price (Datastream VWAP where present, else close) x shares
             x float estimate (as in 05). One line per company (the list line for members).
  basic      float estimate >= 10%; listed >= 30 sessions; liquidity: 12-month German volume (Datastream German
             consolidated volume where present, else primary) x price >= EUR 1bn or turnover >= 20% (non-members),
             >= EUR 0.8bn or 10% (members) - current rule applied to all years
  EBITDA     era 3 (Sep 2021 ->) DAX newcomers need Worldscope item18198 > 0 in both of the two latest fiscal years
             available 90 days after fiscal year end; missing data = pass (flagged)
  tech       DAX Technology sector = Datastream 'Technology All Share' list (7100) at the last month-end, or a TecDAX
             listing on or before d
  segment    non-members must be on the Prime All Share list (4360; Prime Standard, the pre-2021 requirement) or,
             from Sep 2021, the CDAX list (269; Regulated Market), at the last month-end on or before d
  rules      Fast Exit, Fast Entry, then (regular reviews) Regular Exit and Regular Entry, DAX -> MDAX -> SDAX with
             relegation candidates and count restoration first; TecDAX independently on the technology ranks
             (era 1: TecDAX excludes DAX members; MDAX/SDAX exclude technology names and use ranks without them).
H6, H3 and C are literal screens of review q's rule set with membership as known at d (month-end lists plus
announced changes). Actual movers that the screen has never ranked are kept with no prediction (ranked=False), so
recall counts every review move. Naive benchmark: per index, top-N non-members by full market cap among the names ranked
inside the index's band / bottom-N members, N = changes per review over the last 4 reviews of that kind.
Outputs: predictions_dax.csv (company x review x horizon), prediction_diagnostics_dax.csv
"""
import numpy as np, pandas as pd
from common import DATA, OUT, eu_calendar, xetra_calendar, review_dates, dax_rules

cal = eu_calendar(); xcal = xetra_calendar()
rev = review_dates(xcal, "dax").reset_index(drop=True)
z = np.load(DATA / "panel.npz")
P, VP, SH, AJ, FX = (z[k] for k in ["P", "VP", "SH", "AJ", "FX"])
L = pd.read_parquet(DATA / "panel_lines.parquet")
nd, nl = P.shape
lid_ix = pd.Index(L.lid)
ev = pd.read_csv(OUT / "events_dax.csv", dtype={"gvkey": str}, parse_dates=["T", "eff"])
evr = ev[ev.etype == "review"]
rev = rev[rev.review <= evr.review.max()].reset_index(drop=True)
dl = pd.read_parquet(DATA / "dax_lists_mapped.parquet").dropna(subset=["gvkey"])
LISTS = {404: "DAX", 7137: "MDAX", 4750: "SDAX", 7046: "TecDAX"}
dl["idx"] = dl.list.map(LISTS)

# ---- Datastream VWAP / German consolidated volume aligned to the panel
VW = np.full((nd, nl), np.nan, np.float32); VC = np.full((nd, nl), np.nan, np.float32)
try:
    dp = pd.read_parquet(DATA / "ds_prim.parquet")
    dp["marketdate"] = pd.to_datetime(dp.marketdate)
    dp = dp[dp.marketdate.isin(cal)]
    ic2col = {}
    for i, ic in zip(L.index, L.ds_pq):
        if not np.isnan(ic) and L.home_cty[i] == "DEU":
            ic2col.setdefault(int(ic), i)
    dp = dp[dp.infocode.isin(ic2col)]
    r_ = cal.get_indexer(dp.marketdate); c_ = dp.infocode.map(ic2col).values
    VW[r_, c_] = dp.vwap.values; VC[r_, c_] = dp.consolvol.values
except FileNotFoundError:
    print("ds_prim.parquet missing: close-based price and primary volume only")
AJf = pd.DataFrame(AJ).ffill().values
PX = np.where(np.isnan(VW), P, VW)                 # unadjusted price for the 20-day mean (VWAP where available)
VOL = np.where(np.isnan(VC), VP / np.where(AJf > 0, AJf, 1), VC)   # unadjusted shares traded, German consolidated if present
VALd = VOL * P * FX
MC_LINE = P * SH / AJf * FX

# ---- companies in the DAX universe and their candidate lines (largest share class by FFMC is chosen at each date)
gvk = L.gvkey.values
list_lids = set(dl.lid.dropna())
german_line = (L.home_cty == "DEU") & L.home_line & L.tpci.isin(["0", "1"])
cand = L[german_line | L.lid.isin(list_lids)]
cand_cols = cand.groupby("gvkey").apply(lambda x: list(x.index), include_groups=False).to_dict()
# non-members: German home listing, operating company (GICS assigned), listing segment (Prime All Share list 4360
# before Sep 2021 = Prime Standard; CDAX 269 from Sep 2021 = Regulated Market) and domicile (German registered office
# or HQ before Sep 2021; EU/EFTA domicile or German HQ after; Compustat fic/loc are current, not point-in-time)
EU_EFTA = {"DEU", "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "GRC", "HUN", "IRL", "ITA", "LVA",
           "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "CHE", "NOR", "ISL", "LIE"}
german_g = set(L.loc[german_line & L.gsector.notna(), "gvkey"])
dom_de = set(L.loc[(L["fic"] == "DEU") | (L["loc"] == "DEU"), "gvkey"])
dom_eu = set(L.loc[L["fic"].isin(EU_EFTA) | (L["loc"] == "DEU"), "gvkey"])
print("DAX universe companies:", len(cand_cols), "| German operating companies:", len(german_g))
# tech flag (GICS proxy for the DAX Technology industry)
gs = L.set_index("lid")
gsub = L.gsubind.fillna("").astype(str).values
tec_list = dl[dl.idx == "TecDAX"]
SEG = {c: dl[dl.list == c] for c in (4360, 269, 7100)}       # Prime All Share, CDAX, Technology All Share


def list_members(code, km):
    x = SEG[code]
    return set(x[(x.m0 <= km) & (x.m1 >= km)].gvkey.dropna())

# float
fm = pd.read_parquet(DATA / "float_monthly.parquet")
fm["month"] = pd.to_datetime(fm.month)
months = pd.DatetimeIndex(sorted(fm.month.unique()))
FDS = np.full((len(months), nl), np.nan); FWS = np.full((len(months), nl), np.nan)
mi = months.get_indexer(fm.month)
FDS[mi, fm.line.values] = fm.ff_ds.values; FWS[mi, fm.line.values] = fm.ff_ws.values

# EBITDA (Worldscope via Datastream link)
lk = pd.read_parquet(DATA / "ds_ws_link.parquet").sort_values(["coderank", "inforank"]).drop_duplicates("infocode")
ic2code = dict(zip(lk.infocode, lk.code))
eb = pd.read_parquet(DATA / "ws_ebitda.parquet")
eb = eb[eb.ebitda.notna() & eb.fye.notna()].copy()
eb["fye"] = pd.to_datetime(eb.fye); eb["avail"] = eb.fye + pd.Timedelta(days=90)
eb = eb.sort_values(["code", "fye", "seq"]).drop_duplicates(["code", "fye"], keep="last")
ebg = dict(tuple(eb.groupby("code")))


def ebitda_ok(c, d):
    ic = L.ds_pq[c]
    if np.isnan(ic):
        return True, "missing"
    x = ebg.get(ic2code.get(int(ic)))
    if x is None:
        return True, "missing"
    x = x[x.avail <= d].tail(2)
    if len(x) < 2:
        return True, "missing"
    return bool((x.ebitda > 0).all()), "ws"


dl["m0"] = pd.to_datetime(dl.startdate).dt.to_period("M"); dl["m1"] = pd.to_datetime(dl.enddate).dt.to_period("M")


def state_at(d):
    """Membership as known at d: the last month-end list on or before d (Datastream lists are dated on the last
    trading day of the month), plus review results announced since that month-end."""
    d = pd.Timestamp(d)
    i = xcal.searchsorted(d, side="right")
    last_of_month = i >= len(xcal) or xcal[i].month != d.month
    km = d.to_period("M") if last_of_month else d.to_period("M") - 1
    me = km.to_timestamp(how="end").normalize()
    x = dl[(dl.m0 <= km) & (dl.m1 >= km)]
    lvl = {}
    for nm, l in [("SDAX", 1), ("MDAX", 2), ("DAX", 3)]:
        for g in x.loc[x.idx == nm, "gvkey"]:
            lvl[g] = max(lvl.get(g, 0), l)
    tec = set(x.loc[x.idx == "TecDAX", "gvkey"])
    for r in rev.itertuples():
        if r.A <= d and r.eff > me:
            e = evr[evr.review == r.review]
            for g, l1, t1 in zip(e.gvkey, e.lvl_post, e.tec_post):
                lvl[g] = l1
                if t1:
                    tec.add(g)
                else:
                    tec.discard(g)
    return {g: l for g, l in lvl.items() if l > 0}, tec, km


def select(order_rank, members, n, thr, regular, ok_entry, ok_any, superior):
    """One index. order_rank: dict g -> rank (eligible only). members: current set. ok_entry(g): extra entry test
    (EBITDA for DAX). Returns new set and removed set."""
    rk = order_rank
    cur = {g for g in members if g in rk and g not in superior}
    nonmem = lambda: [g for g in sorted(rk, key=rk.get) if g not in cur and g not in superior and ok_any(g)]
    # restore count
    while len(cur) < n:
        cands = [g for g in nonmem() if ok_entry(g) and rk[g] <= thr["rx"]] or [g for g in nonmem()]
        if not cands:
            break
        cur.add(cands[0])
    while len(cur) > n:
        cur.remove(max(cur, key=rk.get))
    # 1 fast exit
    for g in sorted([g for g in cur if rk[g] > thr["fx"]], key=rk.get, reverse=True):
        cands = [h for h in nonmem() if ok_entry(h) and rk[h] <= thr["rx"]] or nonmem()
        if cands:
            cur.discard(g); cur.add(cands[0])
    # 2 fast entry
    for g in [h for h in nonmem() if rk[h] <= thr["fe"] and ok_entry(h)]:
        low = max(cur, key=rk.get)
        if rk[low] > rk[g]:
            cur.discard(low); cur.add(g)
    if regular:
        # 3 regular exit
        for g in sorted([g for g in cur if rk[g] > thr["rx"]], key=rk.get, reverse=True):
            cands = [h for h in nonmem() if rk[h] <= thr["rxa"] and ok_entry(h)]
            if cands:
                cur.discard(g); cur.add(cands[0])
        # 4 regular entry
        for g in [h for h in nonmem() if rk[h] <= thr["re"] and ok_entry(h)]:
            out = [m for m in cur if rk[m] > thr["rea"]]
            if out:
                cur.discard(max(out, key=rk.get)); cur.add(g)
    return cur


gcol = {}
rows, diag = [], []
for q in rev.itertuples():
    rules = dax_rules(q.review)
    e = evr[evr.review == q.review].set_index("gvkey")
    for kind in ["H6", "H3", "C"]:
        d = getattr(q, kind)
        di = cal.get_loc(d)
        lvl, tec, km = state_at(d)
        seg_ok = list_members(269 if rules["era"] == 3 else 4360, km)
        dom_ok = dom_eu if rules["era"] == 3 else dom_de
        tech_now = list_members(7100, km)
        mi_ = months.searchsorted(d, side="right") - 1
        ffl = np.where(~np.isnan(FDS[mi_]), FDS[mi_], np.where(~np.isnan(FWS[mi_]), FWS[mi_], 1.0))
        fsrc = np.where(~np.isnan(FDS[mi_]), "DS", np.where(~np.isnan(FWS[mi_]), "WS", "default100"))
        p20 = np.nanmean(PX[max(0, di - 19):di + 1], 0)
        shr = SH[di] / np.where(AJf[di] > 0, AJf[di], 1)
        ffmc = p20 * shr * ffl * FX[di]
        mc = np.where(np.isnan(MC_LINE[di]), np.nan, MC_LINE[di])
        d0 = cal.searchsorted(d - pd.Timedelta(days=365), side="right")
        obv = np.nansum(VALd[d0:di + 1], 0)
        live = (~np.isnan(P[max(0, di - 9):di + 1])).any(0)
        first_ok = (~np.isnan(P[:di + 1])).sum(0) >= 30
        recs = []
        for g, cols in cand_cols.items():
            ism = g in lvl or g in tec
            if not ism and not (g in german_g and g in seg_ok and g in dom_ok):
                continue                      # non-member eligibility: home listing, segment, domicile
            cols = [c for c in cols if live[c] and not np.isnan(ffmc[c]) and ffmc[c] > 0 and (ism or L.tpci[c] in ("0", "1"))]
            if not cols:
                continue
            liq_of = {k: ((obv[k] >= 0.8e9 or obv[k] / ffmc[k] >= 0.10) if ism else (obv[k] >= 1e9 or obv[k] / ffmc[k] >= 0.20))
                      for k in cols}
            okc = [k for k in cols if liq_of[k]] or cols
            c = max(okc, key=lambda k: ffmc[k])           # largest liquid share class by free-float market cap
            gcol[g] = c
            tr = obv[c] / ffmc[c]
            liq = liq_of[c]
            elig = liq and ffl[c] >= 0.10 and first_ok[c]
            is_tech = (g in tech_now) or (g in tec) or bool(((tec_list.gvkey == g) & (tec_list.startdate <= d)).any())
            recs.append((g, c, ffmc[c], elig, is_tech, liq, ffl[c], fsrc[c], obv[c], tr, mc[c]))
        df = pd.DataFrame(recs, columns=["gvkey", "col", "ffmc", "elig", "tech", "liq", "ff", "ff_src", "obv", "turn", "mcap"])
        el = df[df.elig].sort_values("ffmc", ascending=False)
        rank = dict(zip(el.gvkey, range(1, len(el) + 1)))
        techset = set(df.loc[df.tech, "gvkey"])
        ebit = {}
        def ok_dax(g):
            if not rules["ebitda"] or lvl.get(g) == 3:
                return True
            if g not in ebit:
                ebit[g] = ebitda_ok(gcol[g], d)[0]
            return ebit[g]
        new = {}
        superior = set()
        for nm, lv in [("DAX", 3), ("MDAX", 2), ("SDAX", 1)]:
            members = {g for g, l in lvl.items() if l == lv}
            if rules["tech_split"] and nm in ("MDAX", "SDAX"):
                rk = {g: i + 1 for i, g in enumerate([g for g in el.gvkey if g not in techset or g in new.get("DAX", set())])}
            else:
                rk = rank
            new[nm] = select(rk, members, rules["n"][nm], rules["thr"][nm], rules["regular"][nm],
                             ok_dax if nm == "DAX" else (lambda g: True), lambda g: True, superior)
            superior = superior | new[nm]
        # TecDAX on technology ranks
        tl = [g for g in el.gvkey if g in techset and not (rules["tech_split"] and g in new["DAX"])]
        rkT = {g: i + 1 for i, g in enumerate(tl)}
        new["TecDAX"] = select(rkT, tec, rules["n"]["TecDAX"], rules["thr"]["TecDAX"], rules["regular"]["TecDAX"],
                               lambda g: True, lambda g: True, set())
        # naive benchmark (full market cap, no float, no rules)
        prev4 = rev[(rev.A <= d) & (rev.review < q.review)].tail(4).review
        e4 = evr[evr.review.isin(prev4)]
        full = df[df.elig].set_index("gvkey").mcap.dropna().sort_values(ascending=False)
        naive_up, naive_dn = set(), set()
        for nm, lv in [("DAX", 3), ("MDAX", 2), ("SDAX", 1)]:
            n_in = int(round(((e4.lvl_post == lv) & (e4.lvl_pre != lv)).sum() / max(len(prev4), 1)))
            n_out = int(round(((e4.lvl_pre == lv) & (e4.lvl_post != lv)).sum() / max(len(prev4), 1)))
            mem_i = [g for g in full.index if lvl.get(g, 0) == lv]
            non_i = [g for g in full.index if lvl.get(g, 0) < lv]
            naive_up |= set(non_i[:n_in]); naive_dn |= set(mem_i[::-1][:n_out])
        lvl_new = {}
        for nm, lv in [("SDAX", 1), ("MDAX", 2), ("DAX", 3)]:
            for g in new[nm]:
                lvl_new[g] = lv
        gall = set(lvl) | set(lvl_new) | tec | new["TecDAX"] | set(e.index) | naive_up | naive_dn
        n_up = n_dn = 0
        for g in gall:
            act = e.loc[g] if g in e.index else None
            if isinstance(act, pd.DataFrame):
                act = act.iloc[0]
            ranked = g in gcol
            if not ranked and act is None:
                continue
            # an actual mover that our screen has never ranked (no live eligible line so far, e.g. a recent IPO or a
            # name outside our segment/domicile proxies) is kept with no prediction, so it counts as a miss in recall
            l0, t0 = lvl.get(g, 0), g in tec
            l1, t1 = (lvl_new.get(g, 0), g in new["TecDAX"]) if ranked else (l0, t0)
            pred_up = (l1 > l0) or (l1 == l0 and t1 and not t0)
            pred_dn = (l1 < l0) or (l1 == l0 and t0 and not t1)
            a_up = act is not None and act.direction == "up"
            a_dn = act is not None and act.direction == "down"
            nv_up, nv_dn = g in naive_up, g in naive_dn
            if not (pred_up or pred_dn or a_up or a_dn or nv_up or nv_dn):
                continue
            n_up += pred_up; n_dn += pred_dn
            c = int(act.col) if act is not None and act.col >= 0 else gcol[g]
            rr = df[df.gvkey == g]
            rows.append(dict(review=q.review, year=q.year, horizon=kind, date=d, gvkey=g, col=c, lid=L.lid[c], conm=L.conm[c],
                             cty=L.home_cty[c], lvl_known=l0, lvl_pred=l1, tec_known=t0, tec_pred=t1,
                             lvl_act=int(act.lvl_post) if act is not None else l0,
                             tec_act=bool(act.tec_post) if act is not None else t0,
                             pred_up=pred_up, pred_down=pred_dn, act_up=a_up, act_down=a_dn,
                             naive_up=nv_up, naive_down=nv_dn,
                             rank=rank.get(g, np.nan), ffmc=float(rr.ffmc.iloc[0]) if len(rr) else np.nan,
                             ff_src=rr.ff_src.iloc[0] if len(rr) else "", mcap=float(rr.mcap.iloc[0]) if len(rr) else np.nan,
                             elig=bool(rr.elig.iloc[0]) if len(rr) else False, tech=g in techset,
                             ebitda_ok=ebitda_ok(gcol.get(g, c), d)[0] if rules["ebitda"] else True, era=rules["era"],
                             ranked=ranked))
        diag.append(dict(review=q.review, horizon=kind, era=rules["era"], n_ranked=len(rank),
                         n_dax=len(new["DAX"]), n_mdax=len(new["MDAX"]), n_sdax=len(new["SDAX"]), n_tec=len(new["TecDAX"]),
                         n_pred_up=int(n_up), n_pred_down=int(n_dn), n_act_up=int((e.direction == "up").sum()),
                         n_act_down=int((e.direction == "down").sum())))
    print(q.review, {k: v for k, v in diag[-1].items() if k.startswith("n_")}, flush=True)

pr = pd.DataFrame(rows)
# index-level flags (predicted and actual adds / deletes per index)
for nm, lv in [("DAX", 3), ("MDAX", 2), ("SDAX", 1)]:
    pr[f"pred_add_{nm}"] = (pr.lvl_pred == lv) & (pr.lvl_known != lv)
    pr[f"pred_del_{nm}"] = (pr.lvl_known == lv) & (pr.lvl_pred != lv)
    pr[f"act_add_{nm}"] = (pr.lvl_act == lv) & (pr.lvl_known != lv)
    pr[f"act_del_{nm}"] = (pr.lvl_known == lv) & (pr.lvl_act != lv)
pr["pred_add_TecDAX"] = pr.tec_pred & ~pr.tec_known; pr["pred_del_TecDAX"] = pr.tec_known & ~pr.tec_pred
pr["act_add_TecDAX"] = pr.tec_act & ~pr.tec_known; pr["act_del_TecDAX"] = pr.tec_known & ~pr.tec_act
# naive benchmark on direction: top-N non-members of each index by full market cap is approximated at company level by
# the N largest non-DAX-family German companies (up) and the N smallest family members (down)
# 'memory' variant (point-in-time; fixed a priori): drop a predicted move that our cut-off screen also predicted at one
# of the last two reviews already announced at d without it happening (structural exclusions: listing segment,
# controlled companies, non-German domicile, share-class quirks)
pC = pr[pr.horizon == "C"]
rej_up = pC[pC.pred_up & ~pC.act_up].groupby("gvkey").review.apply(set).to_dict()
rej_dn = pC[pC.pred_down & ~pC.act_down].groupby("gvkey").review.apply(set).to_dict()
A_of = dict(zip(rev.review, rev.A))


def recent(sets, g, d):
    s_ = sets.get(g)
    if not s_:
        return False
    known = sorted(r for r, a in A_of.items() if a <= d)[-2:]
    return any(r in s_ for r in known)


pr["pred_up_mem"] = pr.pred_up & ~np.array([recent(rej_up, g, d) for g, d in zip(pr.gvkey, pd.to_datetime(pr.date))])
pr["pred_down_mem"] = pr.pred_down & ~np.array([recent(rej_dn, g, d) for g, d in zip(pr.gvkey, pd.to_datetime(pr.date))])
pr.to_csv(OUT / "predictions_dax.csv", index=False)
pd.DataFrame(diag).to_csv(OUT / "prediction_diagnostics_dax.csv", index=False)


def acc(df, pc, ac):
    tp = (df[pc] & df[ac]).sum(); fp = (df[pc] & ~df[ac]).sum(); fn = (~df[pc] & df[ac]).sum()
    return int(tp + fp), int(tp + fn), int(tp), round(tp / max(tp + fp, 1), 3), round(tp / max(tp + fn, 1), 3)


for h in ["H6", "H3", "C"]:
    x = pr[pr.horizon == h]
    print(h, "UP", acc(x, "pred_up", "act_up"), "| DOWN", acc(x, "pred_down", "act_down"))
    print(h, "naive UP", acc(x, "naive_up", "act_up"), "| DOWN", acc(x, "naive_down", "act_down"))
    print(h, "memory UP", acc(x, "pred_up_mem", "act_up"), "| DOWN", acc(x, "pred_down_mem", "act_down"))
    for nm in ["DAX", "MDAX", "SDAX", "TecDAX"]:
        print("   ", nm, "add", acc(x, f"pred_add_{nm}", f"act_add_{nm}"), "del", acc(x, f"pred_del_{nm}", f"act_del_{nm}"))
