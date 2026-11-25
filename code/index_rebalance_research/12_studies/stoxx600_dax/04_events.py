"""Step 4: index-change events, Mar 2008 - Jun 2026.

STOXX Europe 600 (Compustat Global g_idxcst_his 150376, exact dates):
  review   = effective date in (T, eff + 1 day] of a quarterly review (eff = Monday after the 3rd-Friday close)
             and, for deletes, the line keeps trading (>= 5 sessions in the 30 days after eff)
  offcycle = any other date (M&A replacements, spin-offs, fast exits), or a review-date delete that stops trading
  linechg  = same gvkey leaves and re-enters within 5 days (line switch)
  transient= spell shorter than 10 calendar days (spin-off placeholder lines; r2_warehouse §11 trap 15)
DAX family: DAX 150095 / MDAX 153256 / TecDAX 153432 exact dates (stale 1988 LINDE AG spell dropped);
  -> REPLACED: Compustat's DAX history is incomplete (19-20 members 2008-21), so all four DAX-family indices use the
  Datastream month-end lists (DAX 404, MDAX 7137, SDAX 4750 without 450 junk spells, TecDAX 7046), which end
  2026-02-27: DAX-family reviews Mar 2008 - Dec 2025. Company-level moves per review: level 3 DAX / 2 MDAX / 1 SDAX /
  0 none, pre = list at the end of the month before the review month, post = list at the end of the review month;
  TecDAX in/out separately. A move is 'offcycle_in_review_month' when Compustat dates it away from eff and the
  stock shows no trade-day volume spike (< 1.5x).
Trade day T = review's 3rd-Friday close; volume multiple on T = primary volume(T) / median(T-20..T-1).
Outputs: events_stoxx.csv, events_dax.csv, reviews_stoxx.csv, reviews_dax.csv
"""
import numpy as np, pandas as pd
from common import DATA, OUT, eu_calendar, xetra_calendar, review_dates

cal = eu_calendar(); xcal = xetra_calendar()
rs = review_dates(cal, "stoxx"); rs.to_csv(OUT / "reviews_stoxx.csv", index=False)
rd = review_dates(xcal, "dax"); rd.to_csv(OUT / "reviews_dax.csv", index=False)
z = np.load(DATA / "panel.npz")
P, VP = z["P"], z["VP"]
L = pd.read_parquet(DATA / "panel_lines.parquet")
lid_ix = pd.Index(L.lid)
nd = len(cal)
mem = pd.read_parquet(DATA / "members_mapped.parquet")


def vol_mult(c, iT):
    if c < 0 or iT is None:
        return np.nan, np.nan
    pre = VP[max(0, iT - 20):iT, c]
    med = np.nanmedian(pre) if np.sum(~np.isnan(pre)) >= 10 else np.nan
    return VP[iT, c], (VP[iT, c] / med if med and med > 0 else np.nan)


def n_post(c, eff):
    if c < 0:
        return 0
    a = cal.searchsorted(eff); b = cal.searchsorted(eff + pd.Timedelta(days=30), side="right")
    return int((~np.isnan(P[a:b, c])).sum())


# ======================= STOXX 600
m = mem[mem.gvkeyx == "150376"].copy()
adds = m[(m.fromdate >= "2007-12-01")][["gvkey", "iid", "rlid", "fromdate", "thru"]].rename(columns={"fromdate": "eff_date"})
adds["side"] = "add"
dels = m[m.thru.notna() & (m.thru >= "2007-12-01")][["gvkey", "iid", "rlid", "fromdate", "thru"]].copy()
dels["eff_date"] = dels.thru + pd.Timedelta(days=1); dels["side"] = "delete"
ev = pd.concat([adds, dels], ignore_index=True)
ev["spell_days"] = (ev.thru.fillna(pd.Timestamp("2099-01-01")) - ev.fromdate.fillna(ev.eff_date)).dt.days
ev.loc[ev.side == "add", "spell_days"] = (ev.thru.fillna(pd.Timestamp("2099-01-01")) - ev.eff_date).dt.days
a = ev[ev.side == "add"]; d = ev[ev.side == "delete"]
pair = a.merge(d, on="gvkey", suffixes=("_a", "_d"))
pair = pair[(pair.eff_date_a - pair.eff_date_d).abs() <= pd.Timedelta(days=5)]
sw = set(zip(pair.gvkey, pair.iid_a, pair.eff_date_a, ["add"] * len(pair))) | \
     set(zip(pair.gvkey, pair.iid_d, pair.eff_date_d, ["delete"] * len(pair)))
ev["linechg"] = [(g, i, e, s) in sw for g, i, e, s in zip(ev.gvkey, ev.iid, ev.eff_date, ev.side)]
ev["review"] = None
for r in rs.itertuples():
    hit = (ev.eff_date > r.T) & (ev.eff_date <= r.eff + pd.Timedelta(days=1))
    ev.loc[hit, "review"] = r.review
ev["col"] = lid_ix.get_indexer(ev.rlid)
out = []
for e in ev.itertuples():
    T = rs.loc[rs.review == e.review, "T"].iloc[0] if e.review is not None else cal[cal.searchsorted(e.eff_date) - 1]
    iT = cal.get_loc(T)
    v, mlt = vol_mult(e.col, iT)
    out.append(dict(T=T, vol_T=v, mult_T_prim=mlt, n_post30=n_post(e.col, e.eff_date)))
ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def classify(r):
    if r.linechg:
        return "linechg"
    if r.side == "add" and r.spell_days < 10:
        return "transient"
    if r.review is None:
        return "offcycle"
    if r.side == "delete" and r.n_post30 < 5:
        return "offcycle_at_review"
    return "review"


ev["etype"] = ev.apply(classify, axis=1)
ev = ev.join(L.set_index("lid")[["conm", "excntry", "home_cty", "isin"]], on="rlid")
ev = ev[(ev.eff_date >= "2008-03-01") & (ev.eff_date <= "2026-06-30")].sort_values(["eff_date", "side", "conm"])
ev.to_csv(OUT / "events_stoxx.csv", index=False)
print("STOXX 600 events\n", ev.groupby(["etype", "side"]).size().unstack())
r = ev[ev.etype == "review"]
print("unmapped review events (no panel line):", int((r.col < 0).sum()))
print("median primary volume multiple on T:", r.groupby("side").mult_T_prim.median().round(2).to_dict(),
      "| share >= 2x:", r.groupby("side").mult_T_prim.apply(lambda x: (x >= 2).mean()).round(2).to_dict())
print(r.groupby(r.eff_date.dt.year).side.value_counts().unstack().fillna(0).astype(int).T.to_string())

# ======================= DAX family (Datastream month-end lists; Compustat exact dates as a cross-check)
LISTS = {"DAX": 404, "MDAX": 7137, "SDAX": 4750, "TecDAX": 7046}
dl = pd.read_parquet(DATA / "ds_lists.parquet").dropna(subset=["infocode"])
dl["infocode"] = dl.infocode.astype(int)
dl["startdate"] = pd.to_datetime(dl.startdate); dl["enddate"] = pd.to_datetime(dl.enddate)
# 450 junk SDAX spells (all 2005-10-31 -> 2024-03-19) make the list hold ~500 names; dropping them gives 50/70
dl = dl[~((dl.list == 4750) & (dl.startdate == "2005-10-31") & (dl.enddate == "2024-03-19"))]
# Empty month-ends: TecDAX (7046) and Technology All Share (7100) hold no members at the Aug 2017 month-end (every
# spell ends 2017-07-31 and restarts 2017-09-29). Left alone, Sep 2017 shows ~29 false 'TecDAX in' moves and the
# screens see an empty TecDAX at H6/H3/C dates in Aug-Sep 2017. A list-month with no members between the list's first
# and last months is filled with the previous month's list (spells ending in the month before are extended through it).
_ms = lambda s: s.dt.to_period("M")
for code, x in dl.groupby("list"):
    months = pd.period_range(_ms(x.startdate).min(), _ms(x.enddate.where(x.enddate < "2079-01-01")).max(), freq="M")
    m0, m1 = _ms(x.startdate).values, _ms(x.enddate).values
    for gm in [m for m in months if not ((m0 <= m) & (m1 >= m)).any()]:
        ext = x.index[_ms(x.enddate) == gm - 1]
        dl.loc[ext, "enddate"] = gm.to_timestamp(how="end").normalize()
        print(f"list {code}: empty month-end {gm} filled from {gm - 1} ({len(ext)} spells extended)")
dn = pd.read_parquet(DATA / "ds_names.parquet")
# infocode -> panel line: any historical ISIN of the infocode equal to a panel line's ISIN, else SEDOL, else the
# panel's own Datastream link
ic_isins = dn.groupby("infocode")["isin"].apply(lambda x: set(x.dropna())).to_dict()
ic_sed = dn.groupby("infocode")["primqtsedol"].apply(lambda x: set(x.dropna())).to_dict()
isin2lid = L.dropna(subset=["isin"]).groupby("isin")["lid"].apply(list).to_dict()
sed2lid = L.dropna(subset=["sedol"]).groupby("sedol")["lid"].apply(list).to_dict()
lid2gv = dict(zip(L.lid, L.gvkey))
ic2lid = {}
for ic in dl.infocode.unique():
    c = [l for i in ic_isins.get(ic, ()) for l in isin2lid.get(i, [])] or \
        [l for sd_ in ic_sed.get(ic, ()) for l in sed2lid.get(sd_, [])] or \
        list(L.loc[(L.ds_ic == ic) | (L.ds_pq == ic), "lid"])
    if c:
        c = sorted(set(c), key=lambda l: (-int(L.home_line[lid_ix.get_loc(l)]), -int(L.exchg[lid_ix.get_loc(l)] == 171),
                                          -int(L.n_px[lid_ix.get_loc(l)])))
        ic2lid[ic] = c[0]
dl["lid"] = dl.infocode.map(ic2lid)
dl["gvkey"] = dl.lid.map(lid2gv)
dl["m0"] = dl.startdate.dt.to_period("M"); dl["m1"] = dl.enddate.dt.to_period("M")
dl.to_parquet(DATA / "dax_lists_mapped.parquet", index=False)
print("DAX-family list spells", len(dl), "| mapped to a panel line:",
      dl.groupby("list").gvkey.apply(lambda x: round(x.notna().mean(), 3)).to_dict(),
      "| since 2007-06:", round(dl[dl.enddate >= "2007-06-01"].gvkey.notna().mean(), 3))


def dax_ret_line(g, T, list_lid=None):
    """Line used for returns: the list line if it is priced around T, else the company's priced home line
    (German home lines first, common before preferred, primary ROW line first, longest history)."""
    iT = cal.get_loc(T)
    alive = lambda l: (l is not None) and (l in lid_ix) and (~np.isnan(P[max(0, iT - 5):iT + 1, lid_ix.get_loc(l)])).any()
    if alive(list_lid):
        return list_lid
    x = L[L.gvkey == g].copy()
    x = x[[alive(l) for l in x.lid]]
    if not len(x):
        return list_lid
    x["k"] = list(zip(x.home_line & (x.home_cty == "DEU"), x.tpci == "0", x.is_prirow, x.n_px))
    return x.sort_values("k", ascending=False).lid.iloc[0]


def list_at(nm, m):
    """{gvkey: list line} of index nm in month m (Datastream month-end spells, compared by month)."""
    x = dl[(dl.list == LISTS[nm]) & (dl.m0 <= m) & (dl.m1 >= m) & dl.gvkey.notna()]
    return dict(zip(x.gvkey, x.lid))


cs = mem[mem.gvkeyx.isin(["150095", "153256", "153432"])]
cs_dates = pd.concat([cs[["gvkey", "fromdate"]].rename(columns={"fromdate": "d"}),
                      cs[["gvkey", "thru"]].dropna().assign(d=lambda x: x.thru + pd.Timedelta(days=1))[["gvkey", "d"]]])
cs_g = dict(tuple(cs_dates.groupby("gvkey")))
LAST_DS = dl.enddate[dl.enddate < "2079-01-01"].max()
rows = []
for q in rd.itertuples():
    m_post = pd.Period(f"{q.year}-{q.month:02d}", "M"); m_pre = m_post - 1
    me_post = (pd.Timestamp(q.year, q.month, 1) + pd.offsets.MonthEnd(0))
    if m_post > LAST_DS.to_period("M"):
        continue
    pre = {nm: list_at(nm, m_pre) for nm in LISTS}; post = {nm: list_at(nm, m_post) for nm in LISTS}
    allg = set().union(*[set(v) for v in pre.values()], *[set(v) for v in post.values()])
    for g in allg:
        lvl0 = 3 if g in pre["DAX"] else 2 if g in pre["MDAX"] else 1 if g in pre["SDAX"] else 0
        lvl1 = 3 if g in post["DAX"] else 2 if g in post["MDAX"] else 1 if g in post["SDAX"] else 0
        t0, t1 = g in pre["TecDAX"], g in post["TecDAX"]
        if lvl0 == lvl1 and t0 == t1:
            continue
        x = cs_g.get(g)
        cs_off = False
        if x is not None:
            inm = x[(x.d >= pd.Timestamp(q.year, q.month, 1)) & (x.d <= me_post)]
            if len(inm) and not ((inm.d >= q.eff - pd.Timedelta(days=1)) & (inm.d <= q.eff + pd.Timedelta(days=1))).any():
                cs_off = True
        list_lid = next((d_[g] for d_ in list(post.values()) + list(pre.values()) if g in d_), None)
        rows.append(dict(review=q.review, year=q.year, gvkey=g, lvl_pre=lvl0, lvl_post=lvl1, tec_pre=t0, tec_post=t1,
                         T=q.T, eff=q.eff, cs_offcycle=cs_off, rlid=dax_ret_line(g, q.T, list_lid)))
dx = pd.DataFrame(rows)
dx["col"] = lid_ix.get_indexer(dx.rlid.fillna(""))
names = {0: "none", 1: "SDAX", 2: "MDAX", 3: "DAX"}
dx["move"] = [f"{names[a]}->{names[b]}" if a != b else "" for a, b in zip(dx.lvl_pre, dx.lvl_post)]
dx["tec"] = np.where(dx.tec_pre & ~dx.tec_post, "TecDAX out", np.where(~dx.tec_pre & dx.tec_post, "TecDAX in", ""))
dx["direction"] = np.where(dx.lvl_post > dx.lvl_pre, "up", np.where(dx.lvl_post < dx.lvl_pre, "down",
                           np.where(dx.tec == "TecDAX in", "up", "down")))
out = []
for e in dx.itertuples():
    iT = cal.get_loc(e.T)
    v, mlt = vol_mult(e.col, iT)
    out.append(dict(vol_T=v, mult_T_prim=mlt, n_post30=n_post(e.col, e.eff)))
dx = pd.concat([dx.reset_index(drop=True), pd.DataFrame(out)], axis=1)
# Compustat's DAX-family dates are themselves partly month-end dated, so a Compustat date away from eff only marks
# an off-cycle change when the stock also shows no trade-day volume spike (primary multiple < 1.5x on T)
dx["etype"] = np.where(dx.cs_offcycle & ~(dx.mult_T_prim >= 1.5), "offcycle_in_review_month",
                       np.where((dx.direction == "down") & (dx.n_post30 < 5), "offcycle_at_review", "review"))
dx = dx.join(L.set_index("lid")[["conm", "excntry", "home_cty", "isin"]], on="rlid")
dx.to_csv(OUT / "events_dax.csv", index=False)
print("\nDAX family company-level moves by type:", dx.etype.value_counts().to_dict())
r = dx[dx.etype == "review"]
print("review moves:", len(r), "| no price line:", int((r.col < 0).sum()), "| reviews covered:", r.review.min(), "-", r.review.max())
print(r.groupby(["move", "tec"]).size().to_string())
print("median primary volume multiple on T by direction:", r.groupby("direction").mult_T_prim.median().round(2).to_dict(),
      "| share >= 2x:", r.groupby("direction").mult_T_prim.apply(lambda x: (x >= 2).mean()).round(2).to_dict())
print(r.groupby("year").direction.value_counts().unstack().fillna(0).astype(int).T.to_string())
