"""Step 2: R2 reads for the net-flow study (read-only). Writes data/*.parquet.

Run from <repo>/code with PYTHONPATH set:
    python3 index_rebalance_research/12_studies/net_flow/s02_pull_r2.py [step ...]
Steps (default: all):
  us_hold  CRSP MF month-end holdings of the US trackers used as holdings rulers: IVV (S&P 500), IWB (Russell 1000),
           IWM (Russell 2000), IWV (Russell 3000), plus the Vanguard 500 / Russell 1000/2000/3000 portfolios.
  us_tna   CRSP MF fund_hdr of every pure index fund (index_fund_flag = 'D') and its monthly TNA (visible pi, US).
  sp500    Compustat S&P 500 membership spells (gvkeyx 000003; current members' history only = survivor-biased).
  s12      Thomson s12 quarterly holdings of the European and Canadian trackers (STOXX 600, DAX, MDAX, TecDAX,
           SDAX, FTSE 100, FTSE 250; XIC, XIU, ZCN as a cross-check of the iShares files).
  gsec     Compustat Global security identifiers (its cusip column is empty in R2; kept for the record).
  dscusip  Datastream CUSIP/CINS history (ds2cusipchg) for the s12 holdings' 8-character cusips -> infocode.
  ds_pi    Datastream tracker AUM (close x numshrs) and index MV for the families not already in
           ../close_dislocation/data (S&P/TSX 60, S&P 500, Russell 2000 trackers, FTSE 100, FTSE 250).
"""
import sys, time
from pathlib import Path
import pandas as pd
import r2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA

US_PORT = {1025811: "IVV", 1025812: "IWB", 1025818: "IWM", 1025817: "IWV",
           1022248: "V500", 1020913: "VRU1000", 1020915: "VRU2000", 1020916: "VRU3000"}
S12_FUNDS = {45947: ("STOXX", "iShares STOXX Europe 600 (DE) EXSA"), 138577: ("STOXX", "Xtrackers STOXX Europe 600"),
             101864: ("STOXX", "Invesco/Source STOXX Europe 600"), 154209: ("STOXX", "Amundi STOXX Europe 600"),
             121870: ("STOXX", "Lyxor STOXX Europe 600"),
             64843: ("DAX", "iShares Core DAX (DE) EXS1"), 64844: ("MDAX", "iShares MDAX (DE) EXS3"),
             64845: ("TecDAX", "iShares TecDAX (DE) EXS2"), 103374: ("SDAX", "ComStage SDAX"),
             80260: ("FTSE100", "iShares Core FTSE 100 ISF"), 80252: ("FTSE250", "iShares FTSE 250 MIDD"),
             63567: ("TSX", "iShares S&P/TSX Capped Composite XIC"), 60269: ("TSX60", "iShares S&P/TSX 60 XIU"),
             137778: ("TSX", "BMO S&P/TSX Capped Composite ZCN")}
DS_ETF = {"TSX60": [150280, 367270, 242167],                      # XIU, ZIU, HXT
          "SP500": [73987, 73929, 241985],                        # SPY, IVV, VOO
          "R2000": [73898, 242287],                               # IWM, VTWO
          "FTSE100": [151139, 261456],                            # ISF, VUKE
          "FTSE250": [150843, 278798]}                            # MIDD, VMID
DS_MV = {"TSX60": 48553, "SP500": 41620, "FTSE100": 36888, "FTSE250": 36890}


def save(df, name):
    df.to_parquet(DATA / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)


def step_us_hold(con):
    ports = ",".join(str(p) for p in US_PORT)
    d = r2.sql(con, f"""
        SELECT crsp_portno::BIGINT AS portno, report_dt, permno::BIGINT AS permno, permco::BIGINT AS permco,
               cusip, ticker, security_name, nbr_shares::DOUBLE AS nbr_shares, market_val::DOUBLE AS market_val
        FROM crsp_q_mutualfunds.holdings
        WHERE crsp_portno IN ({ports}) AND report_dt >= DATE '2006-12-01'""")
    d["fund"] = d.portno.map(US_PORT)
    save(d, "us_holdings")


def step_us_tna(con):
    h = r2.sql(con, """SELECT crsp_fundno::BIGINT AS fundno, crsp_portno::BIGINT AS portno, fund_name, ticker,
                              index_fund_flag, et_flag, first_offer_dt, end_dt
                       FROM crsp_q_mutualfunds.fund_hdr WHERE index_fund_flag = 'D'""")
    save(h, "us_index_funds")
    con.execute("CREATE OR REPLACE TEMP TABLE f AS SELECT unnest(?)::DOUBLE AS v", [h.fundno.astype(float).tolist()])
    t = r2.sql(con, """SELECT crsp_fundno::BIGINT AS fundno, caldt, mtna::DOUBLE AS mtna
                       FROM crsp_q_mutualfunds.monthly_tna
                       WHERE crsp_fundno IN (SELECT v FROM f) AND caldt >= DATE '2006-12-01'""")
    save(t, "us_index_tna")


def step_sp500(con):
    d = r2.sql(con, """SELECT h.gvkey, h.iid, h."from" AS dfrom, h.thru, s.tic, c.conm
                       FROM comp_na_daily_all.idxcst_his h
                       JOIN comp_na_daily_all.security s USING (gvkey, iid)
                       JOIN comp_na_daily_all.company c USING (gvkey)
                       WHERE h.gvkeyx = '000003'""")
    save(d, "sp500_spells")


def step_s12(con):
    fl = ",".join(str(f) for f in S12_FUNDS)
    d = r2.sql(con, f"""SELECT fundno::BIGINT AS fundno, rdate, fdate, cusip, ticker, stkname, shares::DOUBLE AS shares,
                               change::DOUBLE AS change, prc::DOUBLE AS prc, shrout2::DOUBLE AS shrout2, country
                        FROM tr_mutualfunds.s12 WHERE fundno IN ({fl}) AND rdate >= DATE '2006-06-30'""")
    d["family"] = d.fundno.map(lambda f: S12_FUNDS[f][0])
    d["fund"] = d.fundno.map(lambda f: S12_FUNDS[f][1])
    save(d, "s12_holdings")


def step_gsec(con):
    d = r2.sql(con, """SELECT gvkey, iid, cusip, isin, sedol, exchg, excntry, tic FROM comp_global_daily.g_security
                       WHERE cusip IS NOT NULL""")
    save(d, "g_security")


def step_ds_pi(con):
    ics = sorted({i for v in DS_ETF.values() for i in v})
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [ics])
    a = r2.sql(con, """SELECT infocode::BIGINT AS infocode, marketdate, currency, close::DOUBLE AS close,
                              numshrs::DOUBLE AS numshrs, mktcap::DOUBLE AS mktcap, mktcap_usd::DOUBLE AS mktcap_usd
                       FROM tr_ds_equities.wrds_ds2dsf
                       WHERE infocode IN (SELECT v FROM ic) AND marketdate >= DATE '2005-06-01'""")
    fam = {i: k for k, v in DS_ETF.items() for i in v}
    a["family"] = a.infocode.map(fam)
    save(a, "ds_etf_aum_extra")
    m = r2.sql(con, f"""SELECT dsindexcode::BIGINT AS dsindexcode, valuedate, mv::DOUBLE AS mv, pi_::DOUBLE AS pi_
                       FROM tr_ds_equities.ds2indexdata
                       WHERE dsindexcode IN ({",".join(map(str, DS_MV.values()))}) AND valuedate >= DATE '2005-06-01'""")
    inv = {v: k for k, v in DS_MV.items()}
    m["family"] = m.dsindexcode.map(inv)
    save(m, "ds_index_mv_extra")


def step_dscusip(con):
    s = pd.read_parquet(DATA / "s12_holdings.parquet", columns=["cusip"])
    cu = sorted(set(s.cusip.dropna().str[:8]))
    con.execute("CREATE OR REPLACE TEMP TABLE cu AS SELECT unnest(?) AS v", [cu])
    d = r2.sql(con, """SELECT infocode::BIGINT AS infocode, startdate, enddate, cusip
                       FROM tr_ds_equities.ds2cusipchg WHERE substr(cusip, 1, 8) IN (SELECT v FROM cu)""")
    con.execute("CREATE OR REPLACE TEMP TABLE ic2 AS SELECT unnest(?)::DECIMAL(11,0) AS v", [d.infocode.astype(float).tolist()])
    # cusip line (often a US OTC quote) -> security (dsseccode) -> its primary quote, which is what the panels carry
    n = r2.sql(con, """SELECT a.infocode::BIGINT AS infocode, a.dsseccode::BIGINT AS dsseccode,
                              b.infocode::BIGINT AS prim_infocode, b.region AS prim_region, b.dsqtname AS prim_name
                       FROM tr_ds_equities.ds2ctryqtinfo a
                       JOIN tr_ds_equities.ds2ctryqtinfo b ON a.dsseccode = b.dsseccode AND b.isprimqt = 1
                       WHERE a.infocode IN (SELECT v FROM ic2)""")
    d = d.merge(n, on="infocode", how="left")
    save(d, "ds_cusip")


STEPS = {"us_hold": step_us_hold, "us_tna": step_us_tna, "sp500": step_sp500, "s12": step_s12,
         "gsec": step_gsec, "ds_pi": step_ds_pi, "dscusip": step_dscusip}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    con = r2.connect()
    for s in todo:
        t0 = time.time()
        print(s, flush=True)
        STEPS[s](con)
        print(f"  {time.time() - t0:.0f}s", flush=True)
