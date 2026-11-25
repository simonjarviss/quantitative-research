"""Read-only R2 pulls the engine needs that the sister studies did not cache. Writes data/r2/*.parquet.

Run from <repo>/code:
    PYTHONPATH=. python3 index_rebalance_research/16_flow_engine/r2_pull.py [step ...]
Steps:
  ftse_pi      FTSE 100 / FTSE 250 physical tracker AUM (wrds_ds2dsf.mktcap) and index MV (ds2indexdata 36888/36890)
  ftse_lists   FTSE 100 / 250 daily membership spells (ds2constdly 3670/3671) touching Jun 2026, with ISIN
  members_jun  S&P/TSX (Compustat NA) and STOXX 600 / DAX / MDAX / TecDAX (Compustat Global) changes in Jun 2026
  fx           EUR/CAD/GBP/USD/CHF/SEK/DKK/NOK/PLN daily from comp_global_daily.g_exrt_dly (GBP base), 2026
  vol <isins.csv> <start> <end> <tag>
               daily volume per ISIN on the Datastream primary quote (ds2primqtprc) plus every secondary line in
               ds2scdqtprc (lit MTFs for Europe; exch 295 = Canadian Composite = consolidated for Canada), and the
               primary close; writes data/r2/vol_<tag>.parquet and data/r2/vol_<tag>_map.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import r2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import R2DIR  # noqa: E402

# Physical FTSE trackers (synthetic Invesco/Amundi swap lines and leveraged/short lines excluded)
FTSE100_ETF = {151139: "ISF", 229455: "CUKX", 301310: "ISF USD hdg", 375093: "CUKX EUR hdg", 214295: "HUKX",
               261456: "VUKE", 316846: "VUKG", 136629: "XUKX 1D", 265689: "Xtrackers FTSE 100 1C",
               151405: "UBS FTSE 100"}
FTSE250_ETF = {150843: "MIDD", 278798: "VMID", 316849: "VMIG", 234765: "HSBC FTSE 250", 151266: "XMCX"}


def save(df, name):
    df.to_parquet(R2DIR / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)


def step_ftse_pi(con):
    ics = list(FTSE100_ETF) + list(FTSE250_ETF)
    a = r2.sql(con, f"""SELECT infocode::BIGINT infocode, marketdate, "close"::DOUBLE AS px, numshrs::DOUBLE numshrs,
                               mktcap::DOUBLE mktcap, mktcap_usd::DOUBLE mktcap_usd
                        FROM tr_ds_equities.wrds_ds2dsf
                        WHERE infocode IN ({",".join(map(str, ics))}) AND marketdate >= DATE '2024-01-01'""")
    a["family"] = a.infocode.map(lambda i: "FTSE100" if i in FTSE100_ETF else "FTSE250")
    a["fund"] = a.infocode.map({**FTSE100_ETF, **FTSE250_ETF})
    save(a, "ftse_etf_aum")
    m = r2.sql(con, """SELECT dsindexcode::BIGINT dsindexcode, valuedate, mv::DOUBLE mv
                       FROM tr_ds_equities.ds2indexdata WHERE dsindexcode IN (36888, 36890)
                       AND valuedate >= DATE '2024-01-01'""")
    m["family"] = m.dsindexcode.map({36888: "FTSE100", 36890: "FTSE250"})
    save(m, "ftse_index_mv")


def step_ftse_lists(con):
    s = r2.sql(con, """SELECT c.indexlistintcode::INT AS list, c.infocode::BIGINT AS infocode, c.startdate, c.enddate,
                              n.isin, n.dsqtname, n.isocurrcode
                       FROM tr_ds_equities.ds2constdly c
                       LEFT JOIN (SELECT infocode, any_value(isin) isin, any_value(dsqtname) dsqtname,
                                         any_value(isocurrcode) isocurrcode
                                  FROM tr_ds_equities.wrds_ds_names GROUP BY infocode) n ON n.infocode = c.infocode
                       WHERE c.indexlistintcode IN (3670, 3671)
                         AND c.enddate >= DATE '2026-05-01' AND c.startdate <= DATE '2026-07-01'""")
    save(s, "ftse_lists_jun2026")


def step_members_jun(con):
    a = r2.sql(con, """SELECT h.gvkeyx, h.gvkey, h.iid, h."from" AS fromdate, h.thru, s.tic, s.isin, c.conm
                       FROM comp_na_daily_all.idxcst_his h JOIN comp_na_daily_all.security s USING (gvkey, iid)
                       JOIN comp_na_daily_all.company c USING (gvkey)
                       WHERE h.gvkeyx IN ('000010', '118341')
                         AND (h."from" BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'
                              OR h.thru BETWEEN DATE '2026-06-01' AND DATE '2026-06-30')""")
    b = r2.sql(con, """SELECT h.gvkeyx, h.gvkey, h.iid, h."from" AS fromdate, h.thru, NULL AS tic, s.isin, c.conm
                       FROM comp_global_daily.g_idxcst_his h JOIN comp_global_daily.g_security s USING (gvkey, iid)
                       JOIN comp_global_daily.g_company c USING (gvkey)
                       WHERE h.gvkeyx IN ('150376', '150095', '153256', '153432')
                         AND (h."from" BETWEEN DATE '2026-06-01' AND DATE '2026-06-30'
                              OR h.thru BETWEEN DATE '2026-06-01' AND DATE '2026-06-30')""")
    m = pd.concat([a, b], ignore_index=True)
    m["index"] = m.gvkeyx.map({"000010": "SPTSX_COMP", "118341": "SPTSX_60", "150376": "STOXX600",
                               "150095": "DAX", "153256": "MDAX", "153432": "TECDAX"})
    save(m, "members_jun2026")


def step_fx(con):
    f = r2.sql(con, """SELECT datadate, fromcurd, tocurd, exratd::DOUBLE exratd FROM comp_global_daily.g_exrt_dly
                       WHERE (fromcurd = 'GBP' OR tocurd = 'GBP') AND (tocurd IN ('EUR','CAD','USD','CHF','SEK','DKK','NOK','PLN') OR fromcurd IN ('EUR','CAD','USD','CHF','SEK','DKK','NOK','PLN'))
                         AND datadate >= DATE '2026-01-01'""")
    save(f, "fx_gbp_base_2026")


# lit order-book lines (ds2scdqtprc exchintcode) counted as on-exchange lit volume in Europe; 295 = Canadian Composite
LIT_MTF = (338, 234, 239, 352, 238, 351, 289, 292)
CA_CONSOL = 295
# home venues (ds2exchange): Datastream's "primary quote" is sometimes not the home market (AT&S -> Xetra,
# SES FDR -> Luxembourg), so the home venue's own line is pulled from ds2scdqtprc as well.
HOME = {"NL": 6, "BE": 28, "DK": 45, "IE": 50, "FI": 74, "DE": 78, "PT": 98, "GB": 104, "ES": 109, "IT": 116,
        "NO": 152, "FR": 163, "SE": 188, "CA": 197, "AT": 206, "PL": 210, "CH": 215}
MNEM_CTRY = {"XET": "DE", "FRA": "DE", "LON": "GB", "PAR": "FR", "AMS": "NL", "BRU": "BE", "LIS": "PT", "DUB": "IE",
             "MIL": "IT", "XMD": "ES", "MAD": "ES", "SWX": "CH", "OME": "SE", "CSE": "DK", "HEL": "FI", "OSL": "NO",
             "WBO": "AT", "XWR": "PL", "WAR": "PL", "TSE": "CA", "LUX": "LU"}


def step_vol(con, isin_file, start, end, tag):
    want = pd.read_csv(isin_file)
    if "country" not in want:
        want["country"] = ""
    want = want.dropna(subset=["isin"]).drop_duplicates("isin")
    isins = sorted(set(want["isin"].astype(str)))
    con.execute("CREATE OR REPLACE TEMP TABLE want AS SELECT unnest(?)::VARCHAR AS isin", [isins])
    mp = r2.sql(con, """SELECT n.isin, n.infocode::BIGINT infocode, n.isprimqt::INT isprimqt, n.primqtinfocode::BIGINT pq,
                               n.primexchmnem, n.isocurrcode, n.dsqtname, n.typecode, n.startdate, n.enddate
                        FROM tr_ds_equities.wrds_ds_names n JOIN want w ON w.isin = n.isin""")
    mp = mp[mp.typecode.isin(["EQ"])].copy()
    mp["pq"] = mp.infocode
    mp["prim_country"] = mp.primexchmnem.map(MNEM_CTRY)
    mp = mp.merge(want[["isin", "country"]], on="isin", how="left")
    # prefer a quote on the listing market (interlisted Canadian names have a NYSE infocode with the same ISIN)
    mp["pref"] = ((mp.prim_country == mp.country) * 4 + (mp.isprimqt == 1) * 2
                  + (pd.to_datetime(mp.enddate) >= pd.Timestamp("2026-06-01")) * 1)
    best = mp.sort_values("pref", ascending=False).drop_duplicates("isin").copy()
    ics = sorted(set(best.pq.astype(int)))
    print("  isins", len(isins), "mapped", len(best), "infocodes", len(ics), flush=True)
    con.execute("CREATE OR REPLACE TEMP TABLE wic AS SELECT unnest(?)::BIGINT AS infocode", [ics])
    p = r2.sql(con, f"""SELECT p.infocode::BIGINT infocode, p.marketdate, p.volume::DOUBLE volume,
                               p.close_::DOUBLE close_, p.consolvol::DOUBLE consolvol
                        FROM tr_ds_equities.ds2primqtprc p JOIN wic USING (infocode)
                        WHERE p.marketdate BETWEEN DATE '{start}' AND DATE '{end}'""")
    codes = LIT_MTF + (CA_CONSOL,) + tuple(HOME.values())
    sc = r2.sql(con, f"""SELECT s.infocode::BIGINT infocode, s.marketdate, s.exchintcode::INT exch, s.volume::DOUBLE volume
                         FROM tr_ds_equities.ds2scdqtprc s JOIN wic USING (infocode)
                         WHERE s.marketdate BETWEEN DATE '{start}' AND DATE '{end}'
                           AND s.exchintcode IN ({",".join(map(str, codes))})""")
    lit = sc[sc.exch.isin(LIT_MTF)].groupby(["infocode", "marketdate"]).volume.sum().rename("mtf_vol")
    ca = sc[sc.exch == CA_CONSOL].groupby(["infocode", "marketdate"]).volume.sum().rename("ca_consol_vol")
    home = sc[sc.exch.isin(list(HOME.values()))].rename(columns={"volume": "home_vol"})
    v = p.set_index(["infocode", "marketdate"]).join(lit, how="left").join(ca, how="left").reset_index()
    save(v, f"vol_{tag}")
    save(home, f"vol_{tag}_home")
    save(best[["isin", "pq", "isocurrcode", "dsqtname", "primexchmnem", "prim_country", "country"]]
         .rename(columns={"pq": "infocode"}), f"vol_{tag}_map")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["ftse_pi", "ftse_lists", "members_jun", "fx"]
    con = r2.connect()
    if steps[0] == "vol":
        step_vol(con, *steps[1:5])
        sys.exit()
    for s in steps:
        print(s, flush=True)
        globals()[f"step_{s}"](con)
