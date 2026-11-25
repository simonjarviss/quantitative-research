"""Step 1: pull every R2 input for the STOXX Europe 600 / DAX-family study into ./data/*.parquet.

Run from <repo>/code with PYTHONPATH set to that folder:
    PYTHONPATH=. python3 index_rebalance_research/12_studies/stoxx600_dax/01_pull_data.py [step ...]

Read-only. Each step writes one parquet file and can be re-run on its own (order: members lines secm fx
index exch eushort sdc -> 02_universe.py -> secd dsnames float worldscope dsprim -> ... -> dsmtf).
Cost notes: g_secd (4.3 GB, sorted by gvkey) is read twice (month-end pass, then daily for the candidate
universe); ds2primqtprc (12 GB) and ds2scdqtprc (13.6 GB) are sorted by infocode and filtered by the
candidate infocodes.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import r2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, START, MAIN_EXCH

OUT = DATA
IDX = ("150376", "150095", "153256", "153432")       # STOXX 600, DAX, MDAX, TecDAX (Compustat Global)
CTY = ("GBR", "DEU", "FRA", "CHE", "SWE", "NLD", "ITA", "ESP", "DNK", "NOR", "FIN", "BEL", "AUT", "IRL", "PRT",
       "LUX", "POL", "GRC", "ISL", "CZE")


def save(df, name):
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)


def tmp_list(con, name, vals, typ="VARCHAR"):
    con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT unnest(?)::{typ} AS v", [list(vals)])


def step_members(con):
    m = r2.sql(con, f"""SELECT gvkeyx, gvkey, iid, "from" AS fromdate, thru FROM comp_global_daily.g_idxcst_his
                        WHERE gvkeyx IN {IDX}""")
    save(m, "members")
    # DAX family month-end lists from Datastream: DAX 404, MDAX 7137, SDAX 4750, TecDAX ("TechDAX") 7046.
    # Compustat's DAX history is incomplete (19-20 members 2008-2021, 30 after 2021) and MDAX/TecDAX counts drift,
    # so the DAX family uses these lists (clean counts 30/40, 50/60/50, 50/70, 30 at every month-end checked).
    # Eligibility universes: Prime All Share 4360 (Prime Standard), CDAX 269 (Regulated Market), Technology All Share
    # 7100 (the Prime Standard technology sector = TecDAX universe), all month-end.
    s = r2.sql(con, """SELECT c.indexlistintcode::INT AS list, c.infocode::BIGINT AS infocode, c.startdate, c.enddate
                      FROM tr_ds_equities.ds2constmth c WHERE c.indexlistintcode IN (4750, 404, 7137, 7046, 4360, 269, 7100)""")
    save(s, "ds_lists")


def step_lines(con):
    """Every line of every company whose primary ROW issue is listed in a European country, plus every
    line of every index member."""
    s = r2.sql(con, f"""
        WITH g AS (SELECT c.gvkey FROM comp_global_daily.g_company c
                   JOIN comp_global_daily.g_security s ON s.gvkey = c.gvkey AND s.iid = c.prirow
                   WHERE s.excntry IN {CTY}
                   UNION SELECT gvkey FROM comp_global_daily.g_idxcst_his WHERE gvkeyx IN {IDX})
        SELECT s.gvkey, s.iid, s.isin, s.sedol, s.exchg, s.excntry, s.tpci, s.epf, s.secstat, s.dldtei,
               c.conm, c.fic, c.loc, c.prirow, c.gsector, c.ggroup, c.gind, c.gsubind, c.sic, c.ipodate, c.costat,
               c.dldte, c.dlrsn
        FROM comp_global_daily.g_security s JOIN comp_global_daily.g_company c USING (gvkey)
        WHERE s.gvkey IN (SELECT gvkey FROM g)""")
    save(s, "lines")


def _cand_lines():
    """Primary-ROW line of each European company + every index-member line."""
    L = pd.read_parquet(OUT / "lines.parquet")
    m = pd.read_parquet(OUT / "members.parquet")
    prim = L[(L.iid == L.prirow) & L.excntry.isin(CTY)]
    keep = set(zip(prim.gvkey, prim.iid)) | set(zip(m.gvkey, m.iid))
    return sorted(keep)


def step_secm(con):
    """Pass 1: month-end rows of the candidate lines (sizes the universe)."""
    keep = _cand_lines()
    con.execute("CREATE OR REPLACE TEMP TABLE kl AS SELECT * FROM (SELECT unnest(?) AS gvkey, unnest(?) AS iid)",
                [[k[0] for k in keep], [k[1] for k in keep]])
    t0 = time.time()
    d = r2.sql(con, """
        SELECT s.gvkey, s.iid, s.datadate, s.prccd::DOUBLE AS prccd, s.cshoc::DOUBLE AS cshoc,
               s.cshtrd::DOUBLE AS cshtrd, s.curcdd, s.exchg
        FROM comp_global_daily.g_secd s JOIN kl USING (gvkey, iid)
        WHERE s.monthend = 1 AND s.datadate >= DATE '2005-12-01' AND s.prccd IS NOT NULL""")
    save(d, "secm")
    print(f"  secm took {time.time()-t0:.0f}s")


def step_fx(con):
    x = r2.sql(con, """SELECT datadate, tocurd, exratd::DOUBLE AS exratd FROM comp_global_daily.g_exrt_dly
                      WHERE fromcurd = 'GBP' AND datadate >= DATE '2005-01-01'
                        AND tocurd IN ('EUR','GBP','CHF','SEK','DKK','NOK','PLN','CZK','ISK','USD','HUF')""")
    save(x, "fx")


def step_secd(con):
    """Pass 2: daily rows for the universe chosen in 02_universe.py (data/universe_lines.parquet)."""
    u = pd.read_parquet(OUT / "universe_lines.parquet")
    con.execute("CREATE OR REPLACE TEMP TABLE kl AS SELECT * FROM (SELECT unnest(?) AS gvkey, unnest(?) AS iid)",
                [u.gvkey.tolist(), u.iid.tolist()])
    t0 = time.time()
    d = r2.sql(con, f"""
        SELECT s.gvkey, s.iid, s.datadate, s.prccd::DOUBLE AS prccd,
               s.cshtrd::DOUBLE AS cshtrd, s.cshoc::DOUBLE AS cshoc, s.ajexdi::DOUBLE AS ajexdi, s.trfd,
               s.curcdd, s.exchg
        FROM comp_global_daily.g_secd s JOIN kl USING (gvkey, iid)
        WHERE s.datadate >= DATE '{START}' AND s.prccd IS NOT NULL""")
    save(d, "secd")
    print(f"  secd took {time.time()-t0:.0f}s")


def step_index(con):
    x = r2.sql(con, f"""SELECT gvkeyx, datadate, prccd::DOUBLE AS level FROM comp_global_daily.g_idx_daily
                       WHERE gvkeyx IN {IDX} AND datadate >= DATE '{START}'""")
    save(x, "index_levels")


def step_dsnames(con):
    u = pd.read_parquet(OUT / "universe_lines.parquet")
    sel = pd.read_parquet(OUT / "lines.parquet").merge(u, on=["gvkey", "iid"])
    isin = sorted(set(sel["isin"].dropna())); sed = sorted(set(sel["sedol"].dropna()))
    tmp_list(con, "isl", isin); tmp_list(con, "sdl", sed)
    cols = """infocode::BIGINT AS infocode, dsseccode::BIGINT AS dsseccode, region, isprimqt, typecode, primexchmnem,
              isocurrcode, startdate, enddate, isin, dsqtname, delistdate, primqtsedol,
              primqtinfocode::BIGINT AS primqtinfocode, ibesticker, cmpyctrycode"""
    n = r2.sql(con, f"""SELECT {cols} FROM tr_ds_equities.wrds_ds_names
                       WHERE isin IN (SELECT v FROM isl) OR primqtsedol IN (SELECT v FROM sdl)""")
    dl = pd.read_parquet(OUT / "ds_lists.parquet")
    tmp_list(con, "sic", sorted(set(dl.infocode.dropna().astype(int).tolist())), "BIGINT")
    n2 = r2.sql(con, f"SELECT {cols} FROM tr_ds_equities.wrds_ds_names WHERE infocode IN (SELECT v FROM sic)")
    save(pd.concat([n, n2]).drop_duplicates(), "ds_names")


def _infocodes():
    n = pd.read_parquet(OUT / "ds_names.parquet")
    return sorted(set(n.infocode.astype(int)) | set(n.primqtinfocode.dropna().astype(int)))


def step_float(con):
    tmp_list(con, "ic", _infocodes(), "DECIMAL(11,0)")
    f = r2.sql(con, """
        SELECT infocode::BIGINT AS infocode, valdate, freefloatpct, stratholdpct
        FROM tr_ds_equities.ds2sharehldgs
        WHERE infocode IN (SELECT v FROM ic) AND valdate >= DATE '2005-06-01'""")
    save(f, "ds_float")


def step_worldscope(con):
    tmp_list(con, "ic", _infocodes(), "DECIMAL(11,0)")
    lk = r2.sql(con, """SELECT infocode::BIGINT AS infocode, code::BIGINT AS code, startdate, enddate, coderank, inforank
                       FROM wrdsapps_link_datastream_wscope.ds2ws_linktable WHERE infocode IN (SELECT v FROM ic)""")
    save(lk, "ds_ws_link")
    tmp_list(con, "wc", sorted(set(lk.code.astype(int))), "DECIMAL(11,0)")
    w = r2.sql(con, """SELECT code::BIGINT AS code, year_, freq, date_, item8021 FROM tr_worldscope.wrds_ws_stock
                      WHERE code IN (SELECT v FROM wc) AND freq = 'A' AND year_ >= 2003""")
    save(w, "ws_closely_held")
    # EBITDA: item18198 'EBIT & DEPRECIATION' = the Refinitiv field 18198 named in DAX-MG §7.1.2
    e = r2.sql(con, """SELECT code::BIGINT AS code, year_, freq, seq, item5350 AS fye, item18198 AS ebitda
                      FROM tr_worldscope.wrds_ws_funda WHERE code IN (SELECT v FROM wc) AND freq = 'A' AND year_ >= 2003""")
    save(e, "ws_ebitda")


def step_dsprim(con):
    """Datastream primary quote for the universe: volume, German consolidated volume (consolvol), VWAP."""
    tmp_list(con, "ic", _infocodes(), "DECIMAL(11,0)")
    t0 = time.time()
    v = r2.sql(con, f"""
        SELECT infocode::BIGINT AS infocode, marketdate, volume::DOUBLE AS volume,
               consolvol::DOUBLE AS consolvol, vwap::DOUBLE AS vwap
        FROM tr_ds_equities.ds2primqtprc
        WHERE infocode IN (SELECT v FROM ic) AND marketdate >= DATE '{START}'""")
    save(v, "ds_prim")
    print(f"  ds2primqtprc took {time.time()-t0:.0f}s")


def step_dsmtf(con):
    """Lit MTF book volume (Cboe DXE 338 / CXE 234 / BXE 239, Turquoise Europe 352 / Turquoise 238,
    Aquis Europe 351 / Aquis MTF 289, Equiduct 292) for the names in data/mtf_infocodes.parquet (07_volume.py)."""
    ic = pd.read_parquet(OUT / "mtf_infocodes.parquet").infocode.astype(int).tolist()
    tmp_list(con, "ic", ic, "DECIMAL(11,0)")
    t0 = time.time()
    v = r2.sql(con, """
        SELECT infocode::BIGINT AS infocode, marketdate, exchintcode::INT AS exch, volume::DOUBLE AS volume
        FROM tr_ds_equities.ds2scdqtprc
        WHERE infocode IN (SELECT v FROM ic) AND marketdate >= DATE '2013-01-01'
          AND exchintcode IN (338, 234, 239, 352, 238, 351, 289, 292)""")
    save(v, "ds_mtf")
    print(f"  ds2scdqtprc took {time.time()-t0:.0f}s")


def step_exch(con):
    save(r2.sql(con, "SELECT * FROM tr_ds_equities.ds2exchange"), "ds_exchange")


def step_eushort(con):
    save(r2.sql(con, """SELECT country, position_holder, issuer, isin, position_date, net_short_position
                       FROM wrdsapps_eushort.eushort"""), "eushort")


def step_sdc(con):
    x = r2.sql(con, """
        SELECT master_deal_no, dateann, dateeff, datewith, dateuncon, status, tnation, tpublic, master_cusip,
               target_sedol, tticker, tmanames, amanames, anation, pctacq, psought, deal_value, attitude
        FROM tr_sdc_ma.wrds_ma_details
        WHERE tpublic = 'Public' AND dateann >= DATE '2006-01-01'""")
    save(x, "sdc_ma_all")


STEPS = {"members": step_members, "lines": step_lines, "secm": step_secm, "fx": step_fx, "secd": step_secd,
         "index": step_index, "dsnames": step_dsnames, "float": step_float, "worldscope": step_worldscope,
         "dsprim": step_dsprim, "exch": step_exch, "eushort": step_eushort, "sdc": step_sdc, "dsmtf": step_dsmtf}

if __name__ == "__main__":
    with r2.session() as con:
        for s in sys.argv[1:]:
            print(s, flush=True); STEPS[s](con)
