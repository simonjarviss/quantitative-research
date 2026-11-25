"""Read-only R2 pulls for the MSCI shadow, cached under cache/ (run from <repo>/code with
PYTHONPATH=<repo>/code so `import r2` resolves; rerun.sh sets this).

  ds_names(isins)          Datastream quote lines (infocode, company code dscmpycode, region, currency) per ISIN
  ds_mktval(infocodes)     company-level consolidated shares (consolnumshrs, thousands, ALL share classes incl.
                           unlisted ones, checked on H&M, Ericsson, Empire 2026-10-12) and market value (consolmktval,
                           quote-currency millions), 2026-04-01 .. 2026-06-24 (the table's last date)
R2 stops around 2026-06/07 (06_data/r2_warehouse.md); everything after comes from fund holdings files and Yahoo.
"""
from __future__ import annotations

import pandas as pd

import config as C


def _con():
    import r2
    return r2, r2.connect()


def ds_names(isins: list[str], tag: str) -> pd.DataFrame:
    p = C.CACHE / f"ds_names_{tag}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    r2, con = _con()
    isins = sorted({i for i in isins if isinstance(i, str) and len(i) == 12})
    con.execute("CREATE OR REPLACE TEMP TABLE ii AS SELECT unnest(?) AS v", [isins])
    d = r2.sql(con, """SELECT infocode::BIGINT AS ic, isin, dsqtname, isprimqt, dscmpycode::BIGINT AS cmp, region,
                              primisocurrcode AS qccy, isocurrcode, typecode, startdate, enddate, dscmpyname
                       FROM tr_ds_equities.wrds_ds_names WHERE isin IN (SELECT v FROM ii)
                         AND enddate >= DATE '2025-06-01'""")
    d.to_parquet(p, index=False)
    return d


def ds_mktval(ics: list[int], tag: str, start: str = "2025-12-01") -> pd.DataFrame:
    p = C.CACHE / f"ds_mktval_{tag}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    r2, con = _con()
    ics = sorted({int(i) for i in ics if pd.notna(i)})
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [ics])
    d = r2.sql(con, f"""SELECT infocode::BIGINT AS ic, valdate, consolnumshrs::DOUBLE AS cns, consolmktval::DOUBLE AS cmv
                        FROM tr_ds_equities.ds2mktval WHERE infocode IN (SELECT v FROM ic) AND valdate >= DATE '{start}'""")
    d["valdate"] = pd.to_datetime(d.valdate)
    d.to_parquet(p, index=False)
    return d
