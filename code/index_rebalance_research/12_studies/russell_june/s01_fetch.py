"""Step 1: pull every R2 input once and cache it as local Parquet (_cache/).

Reads (all read-only, via r2.sql):
  tr_ds_equities.wrds_ds_indexconstmerged_mth  Russell 1000/2000 month-end lists (6967/6968), 2007-06..2025-12
  tr_ds_equities.wrds_ds_names(_full)          infocode -> ISIN / local CUSIP (_full adds secondary lines)
  crsp_a_stock.stksecurityinfohist             permno CUSIP history
  crsp_a_stock.msf_v2                          monthly universe 2006-01..2025-12 (price, cap, float, flags)
  crsp_a_stock.dsf_v2                          daily 2007-06-01..2025-12-31, EQTY + IWM (sorted by permno ->
                                               pulled in permno chunks so the 4.8 GB file is read once)
  crsp_a_ccm.ccmxpf_lnkhist, comp_na_daily_all.company   gvkey<->permno, HQ location (current header)
  tr_13f.s34                                   17 pod filer numbers, fdate >= 2007-06-30
  comp_na_daily_all.sec_shortint               semi-monthly short interest >= 2007-01
  crsp_a_stock.dsf_v2 (bid/ask)                closing quotes around each horizon date (spread basis)
Usage: python3 s01_fetch.py [--force]
"""
import sys, time
import pandas as pd
import r2
from common import CACHE, PODS, IWM_PERMNO, R1000, R2000

FORCE = "--force" in sys.argv


def pull(name, query, params=None):
    path = CACHE / f"{name}.parquet"
    if path.exists() and not FORCE:
        print(f"  cached {name}")
        return
    t0 = time.time()
    with r2.session() as con:
        df = r2.sql(con, query, params)
    df.to_parquet(path, index=False)
    print(f"  {name}: {len(df):,} rows in {time.time()-t0:.0f}s")


pull("ds_lists", f"""
    SELECT DISTINCT indexlistintcode::INT code, date_, infocode::BIGINT infocode
    FROM tr_ds_equities.wrds_ds_indexconstmerged_mth
    WHERE indexlistintcode IN ({R1000}, {R2000}) AND date_ >= DATE '2007-06-01'""")

pull("ds_names", f"""
    SELECT infocode::BIGINT infocode, startdate, enddate, isin, isin2, dslocalcode, dsqtname, region, ismajorsec
    FROM tr_ds_equities.wrds_ds_names
    WHERE infocode IN (SELECT DISTINCT infocode FROM tr_ds_equities.wrds_ds_indexconstmerged_mth
                       WHERE indexlistintcode IN ({R1000}, {R2000}) AND date_ >= DATE '2007-06-01')""")

pull("ds_names_full", f"""
    SELECT infocode::BIGINT infocode, startdate, enddate, isin, isin2, dslocalcode, dsqtname, region, ismajorsec
    FROM tr_ds_equities.wrds_ds_names_full
    WHERE infocode IN (SELECT DISTINCT infocode FROM tr_ds_equities.wrds_ds_indexconstmerged_mth
                       WHERE indexlistintcode IN ({R1000}, {R2000}) AND date_ >= DATE '2007-06-01')""")

pull("crsp_secinfo", """
    SELECT permno, permco, secinfostartdt, secinfoenddt, cusip, hdrcusip, ticker, issuernm, sharetype,
           securitytype, securitysubtype
    FROM crsp_a_stock.stksecurityinfohist""")

pull("msf", """
    SELECT permno, permco, mthcaldt, cusip, hdrcusip, ticker, issuernm, siccd, mthprc::DOUBLE prc,
           mthcap::DOUBLE * 1000 cap, mthret::DOUBLE ret, mthvol::DOUBLE vol, usincflg, sharetype, securitytype,
           securitysubtype, primaryexch, issuertype, mthfloatshrqty::DOUBLE floatsh, shrout::DOUBLE shrout
    FROM crsp_a_stock.msf_v2
    WHERE mthcaldt >= DATE '2006-01-01' AND securitytype = 'EQTY'""")

pull("delists", "SELECT * FROM crsp_a_stock.stkdelists WHERE delistingdt >= DATE '2007-01-01'")
pull("ccm", "SELECT gvkey, liid, lpermno::BIGINT permno, lpermco::BIGINT permco, linktype, linkprim, linkdt, linkenddt "
             "FROM crsp_a_ccm.ccmxpf_lnkhist WHERE linktype IN ('LC','LU') AND linkprim IN ('P','C')")
pull("comp_company", "SELECT gvkey, conm, loc, fic, incorp, sic FROM comp_na_daily_all.company")

pull("pods13f", f"""
    SELECT fdate, rdate, mgrno::INT mgrno, cusip, shares, prc, shrout1, shrout2
    FROM tr_13f.s34 WHERE fdate >= DATE '2007-06-30' AND mgrno IN ({",".join(map(str, PODS))})""")

pull("shortint", """
    SELECT gvkey, iid, datadate, shortint::DOUBLE shortint, shortintadj::DOUBLE shortintadj, splitadjdate
    FROM comp_na_daily_all.sec_shortint WHERE datadate >= DATE '2007-01-01'""")

# daily panel, chunked by permno range (dsf_v2 is sorted by permno -> each chunk reads only its row groups)
edges = [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000,
         24000, 30000, 70000, 80000, 85000, 88000, 90000, 92000, 100000]
daily_dir = CACHE / "dsf"
daily_dir.mkdir(exist_ok=True)
for lo, hi in zip(edges[:-1], edges[1:]):
    path = daily_dir / f"dsf_{lo}_{hi}.parquet"
    if path.exists() and not FORCE:
        continue
    t0 = time.time()
    with r2.session() as con:
        df = r2.sql(con, """
            SELECT permno, permco, dlycaldt, dlyret::DOUBLE ret, dlyprc::DOUBLE prc, dlycap::DOUBLE * 1000 cap,
                   dlyvol::DOUBLE vol, shrout::DOUBLE shrout, dlydelflg
            FROM crsp_a_stock.dsf_v2
            WHERE permno >= ? AND permno < ? AND dlycaldt >= DATE '2007-06-01'
              AND (securitytype = 'EQTY' OR permno = ?)""", [lo, hi, IWM_PERMNO])
    df.to_parquet(path, index=False)
    print(f"  dsf {lo}-{hi}: {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

# closing bid/ask on every horizon date (basis for the spread estimate in s06_capacity.py)
import pandas as _pd
from common import CAL as _CAL
_dates = set()
for _y in _CAL.index:
    for _d in (_pd.Timestamp(_y - 1, 12, 31), _pd.Timestamp(_y, 3, 31), _CAL.loc[_y, "rank_day"], _CAL.loc[_y, "T"]):
        for _k in range(0, 5):
            _dates.add((_d - _pd.Timedelta(days=_k)).date())
_dl = ",".join(f"DATE '{d}'" for d in sorted(_dates))
pull("quotes", f"""
    SELECT permno, dlycaldt, dlybid::DOUBLE bid, dlyask::DOUBLE ask, dlyprc::DOUBLE prc
    FROM crsp_a_stock.dsf_v2 WHERE dlycaldt IN ({_dl}) AND securitytype = 'EQTY'""")
print("done")
