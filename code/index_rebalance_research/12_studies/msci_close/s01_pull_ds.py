"""Step 1: Datastream primary-quote prices around every MSCI review close, Feb 2018 - May 2026 (R2, read-only).

Lines: every line of the two sister panels, so events and their matched controls come from one source.
  Canada: the CA-region (Toronto) Datastream quotes of each TSX panel line (panel_lines.ds_ca_ics; a line can carry
          more than one, e.g. after a relisting; s03 keeps the one that traded at T). Interlisted names are read on
          their TSX quote, never the US one.
  Europe: the primary-quote infocode of each STOXX/DAX panel line (panel_lines.ds_pq).
Dates: sessions T-7 .. T+21 of every MSCI review in common.REVIEWS with T <= 2026-06-24 (end of ds2primqtprc), on
the TSX calendar (Canada) and the pan-European calendar (Europe).
Tables: tr_ds_equities.ds2primqtprc (open_, close_, volume, bid, ask, exchintcode, currency) and
        tr_ds_equities.ds2primqtri (ri = total return index of the primary quote).
Output: data/ds_px.parquet (one row per infocode x date). About 5-8 minutes (both files are sorted by infocode, so
the id list touches most row groups).
Step ca_extra: interlisted Canadian names whose Datastream primary quote is a US one (Open Text, Ovintiv, Pan American
Silver, BlackBerry, Ballard, ...) carry no CA-region infocode in panel_lines.ds_ca_ics. Their TSX quote is a separate
infocode (region CA, exchange 197 Toronto) of the same Datastream security (dsseccode) in wrds_ds_names_full. This
step maps each such panel line to its CAD Toronto quote(s) (data/ca_extra_ics.parquet: col, infocode) and appends
their prices and ri to data/ds_px.parquet.
Usage: s01_pull_ds.py [main] [ca_extra]   (no argument = both)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import r2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, EU, TSX, ca_calendar, eu_calendar, reviews, window_dates


def infocodes():
    Lc = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
    ca = set()
    for s in Lc.ds_ca_ics.dropna().astype(str):
        for x in s.split(","):
            x = x.strip()
            if x and x.replace(".0", "").isdigit():
                ca.add(int(float(x)))
    Le = pd.read_parquet(EU / "data" / "panel_lines.parquet")
    eu = set(Le.ds_pq.dropna().astype(int)) | set(Le.ds_ic.dropna().astype(int))
    return sorted(ca), sorted(eu)


def main():
    rv = reviews()
    dca = window_dates(ca_calendar(), rv["T"])
    deu = window_dates(eu_calendar(), rv["T"])
    dates = sorted(set(dca) | set(deu))
    ca, eu = infocodes()
    ic = sorted(set(ca) | set(eu))
    print(f"infocodes CA {len(ca)}, EU {len(eu)}; dates {len(dates)} ({dates[0]} .. {dates[-1]})", flush=True)
    dl = ",".join(f"DATE '{d}'" for d in dates)
    with r2.session() as con:
        con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [ic])
        t0 = time.time()
        p = r2.sql(con, f"""
            SELECT infocode::BIGINT AS infocode, marketdate, exchintcode::INT AS exchintcode, isocurrcode,
                   open_::DOUBLE AS open_, close_::DOUBLE AS close_, volume::DOUBLE AS volume,
                   bid::DOUBLE AS bid, ask::DOUBLE AS ask
            FROM tr_ds_equities.ds2primqtprc
            WHERE infocode IN (SELECT v FROM ic) AND marketdate IN ({dl})""")
        print(f"  ds2primqtprc {len(p):,} rows {time.time()-t0:.0f}s", flush=True)
        p.to_parquet(DATA / "ds_prc.parquet", index=False)
        t0 = time.time()
        r = r2.sql(con, f"""
            SELECT infocode::BIGINT AS infocode, marketdate, ri::DOUBLE AS ri
            FROM tr_ds_equities.ds2primqtri
            WHERE infocode IN (SELECT v FROM ic) AND marketdate IN ({dl})""")
        print(f"  ds2primqtri {len(r):,} rows {time.time()-t0:.0f}s", flush=True)
        r.to_parquet(DATA / "ds_ri.parquet", index=False)
    p = pd.read_parquet(DATA / "ds_prc.parquet")
    r = pd.read_parquet(DATA / "ds_ri.parquet")
    x = p.merge(r, on=["infocode", "marketdate"], how="outer")
    x["marketdate"] = pd.to_datetime(x.marketdate)
    x = x.sort_values(["infocode", "marketdate"]).drop_duplicates(["infocode", "marketdate"])
    x.to_parquet(DATA / "ds_px.parquet", index=False)
    print(f"ds_px: {len(x):,} rows, {x.infocode.nunique():,} infocodes")


def ca_extra():
    Lc = pd.read_parquet(TSX / "data" / "panel_lines.parquet")
    n = pd.read_parquet(TSX / "data" / "ds_names.parquet")
    miss = Lc[Lc.ds_ca_ics.fillna("").astype(str).str.strip() == ""].copy()
    known = []                                         # (col, infocode) of any Datastream quote of the line
    cus2ic = n.groupby("cusip9").infocode.apply(lambda v: sorted(set(v.astype(int))))
    for col, r in miss.iterrows():
        ics = set()
        if np.isfinite(r.ds_float_ic):
            ics.add(int(r.ds_float_ic))
        if isinstance(r.cusip, str):
            ics |= set(cus2ic.get(r.cusip[:9], []))
        known += [(col, ic) for ic in ics]
    known = pd.DataFrame(known, columns=["col", "ic0"])
    rv = reviews()
    dates = window_dates(ca_calendar(), rv["T"])
    dl = ",".join(f"DATE '{d}'" for d in dates)
    with r2.session() as con:
        con.execute("CREATE OR REPLACE TEMP TABLE k AS SELECT unnest(?)::DECIMAL(11,0) AS v", [sorted(set(known.ic0))])
        nm = r2.sql(con, """
            SELECT DISTINCT a.infocode::BIGINT AS ic0, b.infocode::BIGINT AS infocode, b.region, b.dsqtname
            FROM tr_ds_equities.wrds_ds_names_full a JOIN tr_ds_equities.wrds_ds_names_full b USING (dsseccode)
            WHERE a.infocode IN (SELECT v FROM k) AND b.region = 'CA'""")
        nm = nm[~nm.dsqtname.str.contains(r"\(U\$\)", regex=True, na=False)]
        mp = known.merge(nm, on="ic0")[["col", "infocode", "dsqtname"]].drop_duplicates(["col", "infocode"])
        mp.to_parquet(DATA / "ca_extra_ics.parquet", index=False)
        ic = sorted(set(mp.infocode))
        print(f"ca_extra: {miss.shape[0]} lines without a CA quote, {mp.col.nunique()} mapped to {len(ic)} TSX quotes", flush=True)
        con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [ic])
        t0 = time.time()
        p = r2.sql(con, f"""
            SELECT infocode::BIGINT AS infocode, marketdate, exchintcode::INT AS exchintcode, isocurrcode,
                   open_::DOUBLE AS open_, close_::DOUBLE AS close_, volume::DOUBLE AS volume,
                   bid::DOUBLE AS bid, ask::DOUBLE AS ask
            FROM tr_ds_equities.ds2primqtprc
            WHERE infocode IN (SELECT v FROM ic) AND marketdate IN ({dl})""")
        r = r2.sql(con, f"""
            SELECT infocode::BIGINT AS infocode, marketdate, ri::DOUBLE AS ri
            FROM tr_ds_equities.ds2primqtri
            WHERE infocode IN (SELECT v FROM ic) AND marketdate IN ({dl})""")
        print(f"  ca_extra prices {len(p):,} / ri {len(r):,} rows {time.time()-t0:.0f}s", flush=True)
    x = p.merge(r, on=["infocode", "marketdate"], how="outer")
    x["marketdate"] = pd.to_datetime(x.marketdate)
    old = pd.read_parquet(DATA / "ds_px.parquet")
    old = old[~old.infocode.isin(ic)]
    y = pd.concat([old, x], ignore_index=True).sort_values(["infocode", "marketdate"]).drop_duplicates(["infocode", "marketdate"])
    y.to_parquet(DATA / "ds_px.parquet", index=False)
    print(f"ds_px: {len(y):,} rows, {y.infocode.nunique():,} infocodes")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["main", "ca_extra"]
    if "main" in steps:
        main()
    if "ca_extra" in steps:
        ca_extra()
