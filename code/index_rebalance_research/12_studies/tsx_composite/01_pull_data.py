"""Step 1: pull every R2 input for the S&P/TSX Composite study into ./data/*.parquet.

Run from <repo>/code with PYTHONPATH set to that folder:
    PYTHONPATH=. python3 index_rebalance_research/12_studies/tsx_composite/01_pull_data.py [step ...]

Read-only. Each step writes one parquet file and can be re-run on its own.
Cost notes: secd (2.3 GB) and ds2scdqtprc (13.6 GB, sorted by infocode) are the
two big reads; both are filtered by entity id, but the id lists span most row groups.
"""
import sys, time
from pathlib import Path
import pandas as pd
import r2

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
START = "2004-01-01"   # 12m volume + 6m momentum before the first H6 horizon (Sep 2005)


def save(df, name):
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows")


def step_members(con):
    m = r2.sql(con, """
        SELECT h.gvkeyx, h.gvkey, h.iid, h."from" AS fromdate, h.thru
        FROM comp_na_daily_all.idxcst_his h
        WHERE h.gvkeyx IN ('000010', '118341')""")
    save(m, "members")


def step_lines(con):
    # every line of every company that has a Canadian ('..C') common/unit line
    s = r2.sql(con, """
        WITH g AS (SELECT DISTINCT gvkey FROM comp_na_daily_all.security
                   WHERE iid LIKE '%C' AND tpci IN ('0', '4'))
        SELECT s.gvkey, s.iid, s.tic, s.cusip, s.isin, s.sedol, s.exchg, s.tpci, s.secstat, s.dldtei,
               s.excntry, c.conm, c.fic, c.loc, c.gsector, c.gsubind, c.sic, c.ipodate, c.costat, c.dldte, c.dlrsn
        FROM comp_na_daily_all.security s JOIN comp_na_daily_all.company c USING (gvkey)
        WHERE s.gvkey IN (SELECT gvkey FROM g)""")
    save(s, "lines")


def step_secd(con):
    t0 = time.time()
    d = r2.sql(con, f"""
        WITH g AS (SELECT DISTINCT gvkey FROM comp_na_daily_all.security
                   WHERE iid LIKE '%C' AND tpci IN ('0', '4'))
        SELECT gvkey, iid, datadate, prccd::DOUBLE AS prccd, prchd::DOUBLE AS prchd, prcld::DOUBLE AS prcld,
               cshtrd::DOUBLE AS cshtrd, cshoc::DOUBLE AS cshoc, ajexdi::DOUBLE AS ajexdi, trfd,
               exchg, curcdd, prcstd
        FROM comp_na_daily_all.secd
        WHERE gvkey IN (SELECT gvkey FROM g)
          AND datadate >= DATE '{START}' AND prccd IS NOT NULL
          AND ((iid LIKE '%C' AND tpci IN ('0', '4')) OR (iid NOT LIKE '%C' AND exchg IN (11, 12, 14)))""")
    save(d, "secd")
    (OUT / "tsx_calendar.parquet").unlink(missing_ok=True)   # common.tsx_calendar() rebuilds it from secd
    print(f"  secd took {time.time()-t0:.0f}s")


def step_index(con):
    x = r2.sql(con, f"""SELECT gvkeyx, datadate, prccd::DOUBLE AS level FROM comp_na_daily_all.idx_daily
                       WHERE gvkeyx IN ('000010', '118341') AND datadate >= DATE '{START}'""")
    save(x, "index_levels")


def step_dsnames(con):
    lines = pd.read_parquet(OUT / "lines.parquet")
    cus = sorted(set(lines.loc[lines.iid.str.endswith("C"), "cusip"].dropna().str[:9]))
    con.execute("CREATE OR REPLACE TEMP TABLE cus AS SELECT unnest(?) AS cusip9", [cus])
    sed = sorted(set(lines.loc[lines.iid.str.endswith("C"), "sedol"].dropna()))
    con.execute("CREATE OR REPLACE TEMP TABLE sd AS SELECT unnest(?) AS sedol", [sed])
    # match on the CUSIP embedded in any historical ISIN, or on the primary-quote SEDOL; any region
    # (interlisted names often have a NYSE primary quote and no TSX quote in Datastream)
    n = r2.sql(con, """
        SELECT infocode::BIGINT AS infocode, dsseccode::BIGINT AS dsseccode, region, isprimqt, typecode,
               primexchmnem, isocurrcode, startdate, enddate, isin, dsqtname, ticker, delistdate,
               substr(isin, 3, 9) AS cusip9, primqtsedol
        FROM tr_ds_equities.wrds_ds_names
        WHERE substr(isin, 3, 9) IN (SELECT cusip9 FROM cus) OR primqtsedol IN (SELECT sedol FROM sd)""")
    save(n, "ds_names")


def _infocodes(ca_only=False):
    n = pd.read_parquet(OUT / "ds_names.parquet")
    if ca_only:
        n = n[n.region == "CA"]
    return sorted(set(n.infocode.astype(int)))


def step_float(con):
    ic = _infocodes()
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS infocode", [ic])
    f = r2.sql(con, f"""
        SELECT infocode::BIGINT AS infocode, valdate, freefloatpct, stratholdpct, freefloataltpct
        FROM tr_ds_equities.ds2sharehldgs
        WHERE infocode IN (SELECT infocode FROM ic) AND valdate >= DATE '{START}'""")
    save(f, "ds_float")


def step_consvol(con):
    ic = _infocodes(ca_only=True)   # the Canadian Composite line (295) hangs off the TSX quote
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS infocode", [ic])
    t0 = time.time()
    v = r2.sql(con, """
        SELECT infocode::BIGINT AS infocode, marketdate, volume, close_
        FROM tr_ds_equities.ds2scdqtprc
        WHERE exchintcode = 295 AND infocode IN (SELECT infocode FROM ic) AND marketdate >= DATE '2013-01-01'""")
    save(v, "ds_consvol")
    print(f"  ds2scdqtprc took {time.time()-t0:.0f}s")


def step_worldscope(con):
    ic = _infocodes()
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS infocode", [ic])
    lk = r2.sql(con, """SELECT infocode::BIGINT AS infocode, code::BIGINT AS code, startdate, enddate, coderank, inforank
                       FROM wrdsapps_link_datastream_wscope.ds2ws_linktable WHERE infocode IN (SELECT infocode FROM ic)""")
    save(lk, "ds_ws_link")
    codes = sorted(set(lk.code.astype(int)))
    con.execute("CREATE OR REPLACE TEMP TABLE wc AS SELECT unnest(?)::DECIMAL(11,0) AS code", [codes])
    w = r2.sql(con, """SELECT code::BIGINT AS code, year_, freq, date_, item8021, item5475, item5301
                      FROM tr_worldscope.wrds_ws_stock WHERE code IN (SELECT code FROM wc) AND freq = 'A' AND year_ >= 2002""")
    save(w, "ws_closely_held")


STEPS = {"members": step_members, "lines": step_lines, "secd": step_secd, "index": step_index,
         "dsnames": step_dsnames, "float": step_float, "consvol": step_consvol, "worldscope": step_worldscope}



def step_sdc(con):
    # Canadian public targets: used for the pending-M&A-target rule (SPDJI-PP p.7)
    x = r2.sql(con, """
        SELECT master_deal_no, dateann, dateeff, datewith, dateuncon, status, tnation, tpublic, master_cusip,
               tticker, tmanames, amanames, anation, pctacq, psought, deal_value, attitude
        FROM tr_sdc_ma.wrds_ma_details
        WHERE tnation = 'Canada' AND tpublic = 'Public' AND dateann >= DATE '2004-01-01'""")
    save(x, "sdc_ma_canada")


STEPS["sdc"] = step_sdc


if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    with r2.session() as con:
        for s in todo:
            print(s); STEPS[s](con)
