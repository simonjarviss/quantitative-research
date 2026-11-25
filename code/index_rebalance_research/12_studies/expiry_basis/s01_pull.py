"""Pull Datastream futures contracts, daily settlements and index levels from R2 into data/ (read-only queries)."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # code/, where the r2 package lives
import r2
from common import DATA, PAIRS

DATA.mkdir(exist_ok=True)
with r2.session() as con:
    cc = ",".join(str(p[1]) for p in PAIRS)
    ci = r2.sql(con, f"SELECT futcode, contrcode, dsmnem, contrdate, lasttrddate, isocurrcode "
                     f"FROM tr_ds_fut.wrds_contract_info WHERE contrcode IN ({cc})")
    ci.to_parquet(DATA / "contracts.parquet")
    codes = ",".join(str(int(x)) for x in ci.futcode.unique())
    px = r2.sql(con, f"SELECT futcode, date_, settlement, volume, openinterest FROM tr_ds_fut.wrds_fut_contract "
                     f"WHERE futcode IN ({codes}) AND date_ >= DATE '2005-01-01'")
    px.to_parquet(DATA / "fut_px.parquet")
    ic = ",".join(str(p[2]) for p in PAIRS)
    ix = r2.sql(con, f"SELECT dsindexcode, valuedate, pi_, ri FROM tr_ds_equities.ds2indexdata "
                     f"WHERE dsindexcode IN ({ic}) AND valuedate >= DATE '2005-01-01'")
    ix.to_parquet(DATA / "idx.parquet")
print(f"contracts {len(ci)}, settlements {len(px)} to {px.date_.max()}, index rows {len(ix)} to {ix.valuedate.max()}")
