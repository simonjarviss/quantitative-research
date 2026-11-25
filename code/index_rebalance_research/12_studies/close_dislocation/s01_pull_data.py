"""Step 1: the R2 reads this study adds to the sister studies' caches (opens, ETF AUM, index MV, FTSE lists).

Run from <repo>/code with PYTHONPATH set to that folder:
    python3 index_rebalance_research/12_studies/close_dislocation/s01_pull_data.py [step ...]
Steps (each writes one parquet in data/ and can be re-run on its own):
  opens_us   crsp_a_stock.dsf_v2: dlyopen, dlyclose, dlyprc, dlyprcflg, dlyret, dlyvol on T-1, T, T+1 of every
             Russell June recon 2008-2025, all EQTY permnos + IWM (88222). Full-file scan (sorted by permno).
  opens_ca   comp_na_daily_all.secd: prcod, prccd, cshtrd on T-1, T, T+1 of every S&P/TSX review 2006-2026,
             every TSX ('..C') line of the TSX study panel.
  opens_eu   comp_global_daily.g_secd: prcod, prccd, cshtrd on T-1, T, T+1 of every STOXX/DAX review 2008-2026,
             every line of the STOXX/DAX study panel.
  opens_ds   tr_ds_equities.ds2primqtprc: open_, close_ on the same EU dates (fallback / cross-check for g_secd).
  pi         Datastream tracker AUM (wrds_ds2dsf.mktcap) and index MV (ds2indexdata.mv) for the S&P/TSX
             Composite, STOXX 600, DAX, MDAX, SDAX, TecDAX (passive share pi = AUM / MV).
  ftse       Datastream month-end lists FTSE 100 (3670), FTSE 250 (3671), FTSE AIM All-Share (3673).
Read-only.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import r2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RUS, TSX, EU, IWM_PERMNO


def save(df, name):
    df.to_parquet(DATA / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)


def around(cal, Ts, ks=(-1, 0, 1)):
    cal = pd.DatetimeIndex(cal)
    out = set()
    for T in pd.to_datetime(pd.Series(Ts)):
        i = cal.searchsorted(T)
        for k in ks:
            if 0 <= i + k < len(cal):
                out.add(cal[i + k].date())
    return sorted(out)


def us_dates():
    c = duckdb.connect()
    td = c.execute(f"SELECT dlycaldt, count(*) n FROM read_parquet('{RUS}/_cache/dsf/*.parquet') GROUP BY 1 HAVING count(*) > 1000 ORDER BY 1").df()
    cal = pd.DatetimeIndex(pd.to_datetime(td.dlycaldt))
    return around(cal, russell_cal()["T"].values)


def russell_cal():
    """CAL table of russell_june/common.py (loaded under another module name; it creates no files)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("russell_common", RUS / "common.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CAL


def step_opens_us(con):
    dl = ",".join(f"DATE '{d}'" for d in us_dates())
    t0 = time.time()
    d = r2.sql(con, f"""
        SELECT permno, dlycaldt, dlyopen::DOUBLE AS dlyopen, dlyclose::DOUBLE AS dlyclose, dlyprc::DOUBLE AS dlyprc,
               dlyprcflg, dlyret::DOUBLE AS dlyret, dlyvol::DOUBLE AS dlyvol, dlycumfacpr::DOUBLE AS cfacpr
        FROM crsp_a_stock.dsf_v2
        WHERE dlycaldt IN ({dl}) AND (securitytype = 'EQTY' OR permno = {IWM_PERMNO})""")
    save(d, "opens_us")
    print(f"  dsf_v2 scan {time.time()-t0:.0f}s")


def step_opens_ca(con):
    cal = pd.read_parquet(TSX / "data" / "tsx_calendar.parquet").date
    rev = pd.read_csv(TSX / "reviews.csv", parse_dates=["T"])
    dl = ",".join(f"DATE '{d}'" for d in around(cal, rev["T"]))
    L = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
    con.execute("CREATE OR REPLACE TEMP TABLE kl AS SELECT * FROM (SELECT unnest(?) AS gvkey, unnest(?) AS iid)",
                [L.gvkey.tolist(), L.iid.tolist()])
    t0 = time.time()
    d = r2.sql(con, f"""
        SELECT s.gvkey, s.iid, s.datadate, s.prcod::DOUBLE AS prcod, s.prccd::DOUBLE AS prccd, s.prchd::DOUBLE AS prchd,
               s.prcld::DOUBLE AS prcld, s.cshtrd::DOUBLE AS cshtrd, s.prcstd
        FROM comp_na_daily_all.secd s JOIN kl USING (gvkey, iid)
        WHERE s.datadate IN ({dl})""")
    save(d, "opens_ca")
    print(f"  secd {time.time()-t0:.0f}s")


def eu_dates():
    cal = pd.read_parquet(EU / "data" / "eu_calendar.parquet").date
    rs = pd.read_csv(EU / "reviews_stoxx.csv", parse_dates=["T"])
    rd = pd.read_csv(EU / "reviews_dax.csv", parse_dates=["T"])
    return around(cal, pd.concat([rs["T"], rd["T"]]).unique())


def step_opens_eu(con):
    dl = ",".join(f"DATE '{d}'" for d in eu_dates())
    L = pd.read_parquet(EU / "data" / "panel_lines.parquet")
    con.execute("CREATE OR REPLACE TEMP TABLE kl AS SELECT * FROM (SELECT unnest(?) AS gvkey, unnest(?) AS iid)",
                [L.gvkey.tolist(), L.iid.tolist()])
    t0 = time.time()
    d = r2.sql(con, f"""
        SELECT s.gvkey, s.iid, s.datadate, s.prcod::DOUBLE AS prcod, s.prccd::DOUBLE AS prccd, s.prchd::DOUBLE AS prchd,
               s.prcld::DOUBLE AS prcld, s.cshtrd::DOUBLE AS cshtrd, s.curcdd, s.prcstd
        FROM comp_global_daily.g_secd s JOIN kl USING (gvkey, iid)
        WHERE s.datadate IN ({dl})""")
    save(d, "opens_eu")
    print(f"  g_secd {time.time()-t0:.0f}s")


def step_opens_ds(con):
    dl = ",".join(f"DATE '{d}'" for d in eu_dates())
    L = pd.read_parquet(EU / "data" / "panel_lines.parquet")
    ic = sorted(set(L.ds_pq.dropna().astype(int)) | set(L.ds_ic.dropna().astype(int)))
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [ic])
    t0 = time.time()
    d = r2.sql(con, f"""
        SELECT infocode::BIGINT AS infocode, marketdate, open_::DOUBLE AS open_, close_::DOUBLE AS close_,
               volume::DOUBLE AS volume
        FROM tr_ds_equities.ds2primqtprc
        WHERE infocode IN (SELECT v FROM ic) AND marketdate IN ({dl})""")
    save(d, "opens_ds")
    print(f"  ds2primqtprc {time.time()-t0:.0f}s")


# Tracker ETFs (Datastream infocodes), physical funds on the headline index only (as in raw/audits/passive_share_estimates.md)
ETFS = {
    "TSX": [150287, 213129, 322922],                                  # XIC, ZCN, HXCN
    "STOXX": [151392, 361486, 266664, 203911, 362505, 169001, 397365],  # EXSA, EXIE, MEUD, XSX6, XSX7, Lyxor C060, STT6
    "DAX": [147278, 338947, 151297, 353986, 151171, 152741, 151362, 169764, 169002, 259161, 286647, 391519, 328582],
    "MDAX": [151146, 338969, 275895, 276734, 315092, 396352],
    "SDAX": [251401, 293592],
    "TecDAX": [151419, 338948, 293591],
}
INDEX_MV = {"TSX": 48552, "STOXX": 13635, "DAX": 13095, "MDAX": 40753, "SDAX": 46391, "TecDAX": 41379}


def step_pi(con):
    ics = sorted({i for v in ETFS.values() for i in v})
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [ics])
    a = r2.sql(con, """
        SELECT infocode::BIGINT AS infocode, marketdate, currency, close::DOUBLE AS close, numshrs::DOUBLE AS numshrs,
               mktcap::DOUBLE AS mktcap, mktcap_usd::DOUBLE AS mktcap_usd
        FROM tr_ds_equities.wrds_ds2dsf
        WHERE infocode IN (SELECT v FROM ic) AND marketdate >= DATE '2005-06-01'""")
    fam = {i: k for k, v in ETFS.items() for i in v}
    a["family"] = a.infocode.map(fam)
    save(a, "etf_aum")
    m = r2.sql(con, f"""SELECT dsindexcode::BIGINT AS dsindexcode, valuedate, mv::DOUBLE AS mv, pi_::DOUBLE AS pi_
                       FROM tr_ds_equities.ds2indexdata
                       WHERE dsindexcode IN ({",".join(map(str, INDEX_MV.values()))}) AND valuedate >= DATE '2005-06-01'""")
    inv = {v: k for k, v in INDEX_MV.items()}
    m["family"] = m.dsindexcode.map(inv)
    save(m, "index_mv")


def step_ftse(con):
    s = r2.sql(con, """SELECT c.indexlistintcode::INT AS list, c.infocode::BIGINT AS infocode, c.startdate, c.enddate
                      FROM tr_ds_equities.ds2constmth c WHERE c.indexlistintcode IN (3670, 3671, 3673)""")
    save(s, "ftse_lists")


STEPS = {"opens_us": step_opens_us, "opens_ca": step_opens_ca, "opens_eu": step_opens_eu, "opens_ds": step_opens_ds,
         "pi": step_pi, "ftse": step_ftse}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    with r2.session() as con:
        for s in todo:
            print(s, flush=True)
            STEPS[s](con)
