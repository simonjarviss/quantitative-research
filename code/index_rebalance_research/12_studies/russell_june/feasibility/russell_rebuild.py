"""Rebuild Russell 1000/2000 June recon from CRSP msf_v2 on rank day; compare to Datastream lists.
Usage: python3 russell_rebuild.py RANKDAY PRE_SNAP POST_SNAP   (run from the folder containing r2/)"""
import sys
import pandas as pd
import r2

rank_day, pre_snap, post_snap = sys.argv[1], sys.argv[2], sys.argv[3]
SNAP_TABLE = "tr_ds_equities.wrds_ds_indexconstmerged_mth"

with r2.session() as con:
    u = r2.sql(con, """
        SELECT m.permno, m.permco, substr(m.cusip, 1, 8) cusip8, m.ticker, m.issuernm, m.siccd, m.mthprc::DOUBLE prc,
               m.mthcap::DOUBLE * 1000 cap, m.usincflg, m.sharetype, m.primaryexch, m.issuertype,
               m.mthfloatshrqty::DOUBLE fl, m.shrout::DOUBLE shrout
        FROM crsp_a_stock.msf_v2 m
        WHERE m.mthcaldt = ? AND m.securitytype = 'EQTY' AND m.securitysubtype = 'COM'
          AND m.sharetype IN ('NS', 'SB', 'UG') AND m.primaryexch IN ('N', 'Q', 'A', 'R', 'B')""", [rank_day])
    loc = r2.sql(con, """
        SELECT DISTINCT l.lpermno::INT permno, c.loc, c.fic
        FROM crsp_a_ccm.ccmxpf_lnkhist l JOIN comp_na_daily_all.company c USING (gvkey)
        WHERE l.linktype IN ('LC', 'LU') AND l.linkprim IN ('P', 'C')
          AND CAST(? AS DATE) BETWEEN l.linkdt AND coalesce(l.linkenddt, DATE '9999-12-31')""", [rank_day])
    lists = r2.sql(con, f"""
        WITH s AS (SELECT DISTINCT indexlistintcode::INT code, date_, infocode FROM {SNAP_TABLE}
                   WHERE indexlistintcode IN (6967, 6968) AND date_ IN (CAST(? AS DATE), CAST(? AS DATE))),
        n AS (SELECT DISTINCT infocode, substr(isin, 3, 8) cusip8 FROM tr_ds_equities.wrds_ds_names
              WHERE infocode IN (SELECT infocode FROM s) AND isin IS NOT NULL
              UNION SELECT DISTINCT infocode, substr(isin2, 3, 8) FROM tr_ds_equities.wrds_ds_names
              WHERE infocode IN (SELECT infocode FROM s) AND isin2 IS NOT NULL
              UNION SELECT DISTINCT infocode, substr(dslocalcode, 2, 8) FROM tr_ds_equities.wrds_ds_names
              WHERE infocode IN (SELECT infocode FROM s) AND dslocalcode LIKE 'U%')
        SELECT s.code, s.date_, s.infocode::BIGINT infocode, n.cusip8
        FROM s LEFT JOIN n ON n.infocode = s.infocode""", [pre_snap, post_snap])
    crspcus = r2.sql(con, """
        SELECT DISTINCT substr(cusip, 1, 8) cusip8, permco FROM crsp_a_stock.msf_v2
        WHERE mthcaldt BETWEEN CAST(? AS DATE) - INTERVAL 400 DAY AND CAST(? AS DATE) AND cusip IS NOT NULL
        UNION SELECT DISTINCT substr(hdrcusip, 1, 8), permco FROM crsp_a_stock.msf_v2
        WHERE mthcaldt BETWEEN CAST(? AS DATE) - INTERVAL 400 DAY AND CAST(? AS DATE)""", [rank_day, post_snap, rank_day, post_snap])

u = u.merge(loc[["permno", "loc"]].drop_duplicates("permno"), on="permno", how="left")
n0 = len(u)
_L = lists.merge(crspcus.dropna().drop_duplicates(["cusip8", "permco"]), on="cusip8", how="left")
pre_any = set(_L.loc[_L.date_.astype(str) == pre_snap, "permco"].dropna().astype(int))
MEMORY = "--memory" in sys.argv
if MEMORY:
    keep_member = u.permco.isin(pre_any) & (u.sharetype != "AD")
# eligibility (approximate HCI: US-incorporated or US HQ in Compustat)
u = u[(u.usincflg == "Y") | (u.loc == "USA") | (keep_member if MEMORY else False)]
u = u[~u.siccd.isin([6726, 6770])]                                 # BDC / closed-end inv. cos., blank checks
u = u[~u.siccd.isin([6792, 6733])]                                      # royalty / other trusts
u = u[~u.issuernm.str.contains(r"ROYALTY TR| RTY TR| TRUST$| TR$| L\s?P$|L\.P\.|PARTNERS LP| LLC$", regex=True, na=False)]
u = u[~((u.fl.notna()) & (u.fl < 0.05 * u.shrout))]                      # Russell 5% float rule (CRSP float)
# company total cap = sum of listed classes; pricing vehicle price >= 1
comp = u.groupby("permco").agg(cap=("cap", "sum"), prc=("prc", "max"), cusips=("cusip8", lambda s: list(s)),
                               tic=("ticker", "first"), name=("issuernm", "first")).reset_index()
comp = comp[(comp.prc >= 1) & (comp.cap >= 30e6)].sort_values("cap", ascending=False).reset_index(drop=True)
comp["rank"] = comp.index + 1
top = comp[comp["rank"] <= 4000].copy()
top["cumpct"] = top.cap.cumsum() / top.cap.sum() * 100

# map Datastream lists to permco via cusip8
c2p = crspcus.dropna().drop_duplicates("cusip8")
L = lists.merge(c2p, on="cusip8", how="left")
L = L.drop_duplicates(["code", "date_", "infocode", "permco"])   # keep every permco an infocode maps to
def members(code, d):
    x = L[(L.code == code) & (L.date_.astype(str) == d)]
    g = x.groupby("infocode").permco.apply(lambda v: v.notna().any())
    return set(x.permco.dropna().astype(int)), x.infocode.nunique(), int((~g).sum())
pre1, n_pre1, miss_pre1 = members(6967, pre_snap)
pre2, n_pre2, miss_pre2 = members(6968, pre_snap)
post1, n_post1, miss_post1 = members(6967, post_snap)
post2, n_post2, miss_post2 = members(6968, post_snap)

bp = top.loc[top["rank"] == 1000, "cumpct"].iloc[0]
lo, hi = bp - 2.5, bp + 2.5
def assign(r):
    if r["rank"] > 3000:
        return "out"
    pc = int(r.permco)
    if (pc in pre1 or pc in pre2) and lo <= r.cumpct <= hi:
        return "R1000" if pc in pre1 else "R2000"
    return "R1000" if r["rank"] <= 1000 else "R2000"
top["pred"] = top.apply(assign, axis=1)
pred1 = set(top.loc[top.pred == "R1000", "permco"].astype(int))
pred2 = set(top.loc[top.pred == "R2000", "permco"].astype(int))
nob1 = set(top.loc[top["rank"] <= 1000, "permco"].astype(int))
nob2 = set(top.loc[(top["rank"] > 1000) & (top["rank"] <= 3000), "permco"].astype(int))

def pr(pred, act, label):
    tp = len(pred & act)
    p = tp / len(pred) if pred else float("nan")
    r = tp / len(act) if act else float("nan")
    print(f"{label:<44} pred {len(pred):5d}  actual {len(act):5d}  hit {tp:5d}  precision {p:6.1%}  recall {r:6.1%}")

print(f"rank day {rank_day}: CRSP rows {n0}, eligible companies {len(comp)}, R1000/R2000 breakpoint cap ${top.loc[top['rank']==1000,'cap'].iloc[0]/1e9:.2f}bn, "
      f"rank-3000 cap ${top.loc[top['rank']==3000,'cap'].iloc[0]/1e6:.0f}m, band {lo:.2f}-{hi:.2f} pct")
print(f"Datastream lists: pre R1000 {n_pre1} (unmapped {miss_pre1}), pre R2000 {n_pre2} (unmapped {miss_pre2}), "
      f"post R1000 {n_post1} (unmapped {miss_post1}), post R2000 {n_post2} (unmapped {miss_post2})")
pre3, post3 = pre1 | pre2, post1 | post2
pr(pred1, post1, "membership R1000 (banded)")
pr(pred2, post2, "membership R2000 (banded)")
pr(nob2, post2, "membership R2000 (no banding)")
pr(pred2 - pre2, post2 - pre2, "R2000 ADDS (banded)")
pr(nob2 - pre2, post2 - pre2, "R2000 ADDS (no banding)")
pr(pre2 - pred2, pre2 - post2, "R2000 DELETES (banded)")
pr((pred2 - pre3), (post2 - pre3), "R2000 adds from outside R3000 (new names)")
pr((pred1 & pre2), (post1 & pre2), "R2000 -> R1000 migrations")
pr((pred2 & pre1), (post2 & pre1), "R1000 -> R2000 migrations")
pr(pre3 - (pred1 | pred2), pre3 - post3, "R3000 deletes (fall below #3000)")
# scoring restricted to names CRSP could see on rank day (removes post-rank-day IPOs, mapping gaps)
seen = set(comp.permco.astype(int))
pr((pred2 - pre2) & seen, (post2 - pre2) & seen, "R2000 ADDS, names in CRSP universe")
miss_add = (post2 - pre2) - pred2
print("actual R2000 adds missed:", len(miss_add), " of which not in eligible CRSP universe:", len(miss_add - seen))

if len(sys.argv) > 4 and sys.argv[4] == "diag":
    fp = (pred2 - pre3) - (post2 - pre3)
    x = top[top.permco.isin(fp)].merge(u.groupby("permco").agg(usinc=("usincflg", "first"), loc=("loc", "first"), sic=("siccd", "first"),
                                                                  exch=("primaryexch", "first"), fl=("fl", "sum"), sh=("shrout", "sum")).reset_index(), on="permco")
    x["fl_ratio"] = (x.fl / x.sh).round(2)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 80)
    print(x[["rank", "tic", "name", "cap", "usinc", "loc", "sic", "exch", "fl_ratio", "cusips"]].sort_values("rank").head(60).to_string())
    print(x.groupby(["usinc", "loc"], dropna=False).size())
    print("rank buckets:", pd.cut(x["rank"], [1000, 1500, 2000, 2500, 2800, 3000]).value_counts().sort_index().to_dict())

if len(sys.argv) > 4 and sys.argv[4] == "dels":
    fp = (pre3 - (pred1 | pred2)) - (pre3 - post3)
    x = comp[comp.permco.isin(fp)]
    print("false-positive R3000 deletes:", len(fp), " in eligible CRSP universe:", len(x), " median rank", x["rank"].median())
    print(x.sort_values("rank")[["rank", "tic", "name", "cap"]].head(25).to_string())

if len(sys.argv) > 4 and sys.argv[4] == "why":
    with r2.session() as con:
        raw = r2.sql(con, """SELECT permno, permco, issuernm, siccd, usincflg, sharetype, securitysubtype, primaryexch, mthprc::DOUBLE prc,
                                    mthcap::DOUBLE*1000 cap, mthfloatshrqty::DOUBLE fl, shrout::DOUBLE shrout
                             FROM crsp_a_stock.msf_v2 WHERE mthcaldt = ?""", [rank_day])
    raw = raw.merge(loc[["permno", "loc"]].drop_duplicates("permno"), on="permno", how="left")
    fp = (pre3 - (pred1 | pred2)) - (pre3 - post3)
    r = raw[raw.permco.isin(fp)].copy()
    def reason(x):
        if x.securitysubtype != "COM" or x.sharetype not in ("NS", "SB", "UG"): return f"type {x.securitysubtype}/{x.sharetype}"
        if x.primaryexch not in ("N", "Q", "A", "R", "B"): return f"exch {x.primaryexch}"
        if not (x.usincflg == "Y" or x["loc"] == "USA"): return "foreign inc + non-US HQ"
        if x.siccd in (6726, 6770, 6792, 6733): return f"sic {x.siccd}"
        import re
        if re.search(r"ROYALTY TR| RTY TR| TRUST$| TR$| L\s?P$|L\.P\.|PARTNERS LP| LLC$", x.issuernm or ""): return "name LP/LLC/trust"
        if pd.notna(x.fl) and x.fl < 0.05 * x.shrout: return "float<5%"
        if x.prc < 1: return "price<1"
        return "other (cap/rank)"
    r["why"] = r.apply(reason, axis=1)
    print("FP deletes with a CRSP row on rank day:", r.permco.nunique(), "of", len(fp))
    print(r.drop_duplicates("permco").why.value_counts().to_string())
    print(r[r.why == "foreign inc + non-US HQ"].drop_duplicates("permco")[["issuernm", "loc", "siccd", "cap"]].head(12).to_string())
