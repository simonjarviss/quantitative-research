"""Step 2: R2 reads for the msci_pi study (read-only). Writes data/*.parquet.

Steps (default: all):
  refs    Thomson s12 quarter-end holdings of the reference iShares funds (the ruler of each segment):
            CSCA  iShares MSCI Canada UCITS (s12 165547)         MSCI Canada Standard
            IMEA  iShares Core MSCI Europe UCITS (167255, 149579) MSCI Europe Standard
            WSML  iShares MSCI World Small Cap UCITS (178312)     MSCI World Small Cap (Canadian and European lines)
            SCZ   iShares MSCI EAFE Small-Cap (82876)             European Small Cap before WSML (2017-12 .. 2018-12)
            SWDA  iShares Core MSCI World UCITS (163277), URTH (168687)   cross-checks
  s12agg  For EVERY fund in s12 (2017-10 .. 2025-12): holdings in the reference names, aggregated in SQL to one row
          per fund x report date x segment x country: matched names, reference-weight coverage, median and IQR of
          log(fund shares / reference shares), matched value. A cap-weighted tracker of an index that contains the
          segment holds every name in proportion to the ruler (tight IQR, coverage ~1).
  fnames  s12type1 fund names, countries and assets (fund-level), 2017-10 .. 2025-12.
  dsetf   Datastream daily shares outstanding x close for the main MSCI ETFs (extends visible AUM past 2025-12).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import r2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, TSX  # noqa: E402

REF_FUNDS = {165547: "CSCA", 167255: "IMEA", 149579: "IMEU", 178312: "WSML", 82876: "SCZ", 163277: "SWDA",
             168687: "URTH"}
START = "2017-10-01"
# CINS first character -> ISO2 country of a European issuer (CUSIP Global Services country codes); X = other Europe
CINS_CTRY = {"G": "GB", "D": "DE", "F": "FR", "H": "CH", "W": "SE", "N": "NL", "T": "IT", "E": "ES", "K": "DK",
             "B": "BE", "R": "NO", "A": "AT", "L": "LU", "X": "XE"}


def save(df, name):
    df.to_parquet(DATA / f"{name}.parquet", index=False)
    print(f"  {name}: {len(df):,} rows", flush=True)


def step_refs(con):
    fl = ",".join(str(f) for f in REF_FUNDS)
    d = r2.sql(con, f"""SELECT fundno::BIGINT AS fundno, rdate, fdate, substr(cusip, 1, 8) AS c8, stkname, ticker,
                               shares::DOUBLE AS shares, prc::DOUBLE AS prc, shrout2::DOUBLE AS shrout2
                        FROM tr_mutualfunds.s12 WHERE fundno IN ({fl}) AND fdate >= DATE '{START}' AND shares > 0""")
    d = d.sort_values("fdate").groupby(["fundno", "rdate", "c8"]).tail(1)     # latest filing of each report date
    d["ref"] = d.fundno.map(REF_FUNDS)
    save(d, "s12_refs")


def ref_table():
    """Reference names per quarter-end q: segment (STD/SC), market (CA / EU country), ruler shares and weight."""
    d = pd.read_parquet(DATA / "s12_refs.parquet")
    d["rdate"] = pd.to_datetime(d.rdate)
    d["q"] = d.rdate + pd.offsets.QuarterEnd(0)
    pl = pd.read_parquet(TSX / "data" / "panel_lines.parquet", columns=["cusip", "isin"])
    ca8 = set(pl["cusip"].dropna().str[:8]) | set(pl["isin"].dropna().str[2:10])
    d["ctry"] = np.where(d.c8.isin(ca8), "CA", d.c8.str[0].map(CINS_CTRY))
    out = []
    for q, g in d.groupby("q"):
        def pick(names):
            for n in names:
                x = g[g.ref == n]
                if len(x) > 20:
                    return x[x.rdate == x.rdate.max()]
            return None
        ca = pick(["CSCA"])
        eu = pick(["IMEA", "IMEU"])
        sc = pick(["WSML", "SCZ"])
        if ca is not None:
            x = ca[ca.ctry == "CA"].assign(seg="STD", mkt="CA")
            out.append(x)
        if eu is not None:
            x = eu[eu.ctry.notna() & (eu.ctry != "CA")].assign(seg="STD", mkt="EU")
            out.append(x)
        if sc is not None:
            std = set(ca.c8) if ca is not None else set()
            std |= set(eu.c8) if eu is not None else set()
            x = sc[sc.ctry.notna() & ~sc.c8.isin(std)].copy()
            x["seg"] = "SC"
            x["mkt"] = np.where(x.ctry == "CA", "CA", "EU")
            out.append(x)
    R = pd.concat(out, ignore_index=True)
    R["val"] = R.shares * R.prc
    R["w"] = R.val / R.groupby(["q", "seg", "mkt"]).val.transform("sum")
    R["own"] = R.shares / (R.shrout2 * 1000)
    return R[["q", "seg", "mkt", "ctry", "c8", "ref", "rdate", "shares", "prc", "shrout2", "own", "w", "stkname"]]


def step_s12agg(con):
    R = ref_table()
    R.to_parquet(DATA / "ref_names.parquet", index=False)
    con.register("ref_df", R[["q", "seg", "mkt", "ctry", "c8", "shares", "w"]].rename(columns={"shares": "h_ref"}))
    con.execute("CREATE OR REPLACE TEMP TABLE ref AS SELECT q::DATE AS q, seg, mkt, ctry, c8, h_ref, w FROM ref_df")
    con.execute("CREATE OR REPLACE TEMP TABLE refc8 AS SELECT DISTINCT c8 FROM ref")
    src = r2.source("tr_mutualfunds.s12")
    t0 = time.time()
    d = r2.q(con, f"""
        WITH h0 AS (
            SELECT fundno::BIGINT AS fundno, rdate, fdate, substr(cusip, 1, 8) AS c8, shares::DOUBLE AS shares,
                   prc::DOUBLE AS prc
            FROM {src}
            WHERE fdate >= DATE '{START}' AND shares > 0 AND substr(cusip, 1, 8) IN (SELECT c8 FROM refc8)),
        h1 AS (SELECT * FROM h0 QUALIFY fdate = max(fdate) OVER (PARTITION BY fundno, rdate)),
        h AS (SELECT fundno, rdate, c8, sum(shares) AS sh, any_value(prc) AS prc,
                     (date_trunc('quarter', rdate) + INTERVAL 3 MONTH - INTERVAL 1 DAY)::DATE AS q
              FROM h1 GROUP BY fundno, rdate, c8),
        j AS (SELECT h.fundno, h.rdate, r.q, r.seg, r.mkt, r.ctry, h.c8, h.sh, h.prc, r.h_ref, r.w,
                     ln(h.sh / r.h_ref) AS lr
              FROM h JOIN ref r ON r.q = h.q AND r.c8 = h.c8)
        SELECT fundno, rdate, q, seg, mkt, ctry, count(*) AS n, sum(w) AS wcov,
               quantile_cont(lr, 0.5) AS lr_med, quantile_cont(lr, 0.25) AS lr_p25, quantile_cont(lr, 0.75) AS lr_p75,
               sum(sh * prc) AS val, sum(w * lr) / sum(w) AS lr_wmean
        FROM j GROUP BY ALL""")
    print(f"  s12agg query {time.time() - t0:.0f}s", flush=True)
    save(d, "s12_agg")


def step_fnames(con):
    d = r2.sql(con, f"""SELECT fundno::BIGINT AS fundno, rdate, fdate, fundname, country, assets::DOUBLE AS assets
                        FROM tr_mutualfunds.s12type1 WHERE fdate >= DATE '{START}'""")
    d = d.sort_values("fdate").groupby(["fundno", "rdate"]).tail(1)
    save(d, "s12_fnames")


STEPS = {"refs": step_refs, "s12agg": step_s12agg, "fnames": step_fnames}  # + dsetf below



# ---- Datastream ETF shares outstanding (extends visible pi past the last s12 quarter, 2025-12) ----
ETF_ISIN = {  # ticker: (ISIN, s12 fundno)
    "IEFA": ("US46432F8427", 166623), "EFA": ("US4642874659", 67052), "EFV": ("US4642888774", 79069),
    "EFG": ("US4642888857", 79070), "IXUS": ("US46432F8344", 159772), "IDEV": ("US46435G3267", 137257),
    "IEUR": ("US46434V7385", 91849), "EZU": ("US4642866085", 55899), "EWC": ("US4642865095", 2074),
    "SCZ": ("US4642882736", 82876), "ACWI": ("US4642882579", 82911), "ACWX": ("US4642882405", 82910),
    "URTH": ("US4642863926", 168687), "SWDA": ("IE00B4L5Y983", 163277), "IMEA": ("IE00B4K48X80", 167255),
    "IMEU": ("IE00B1YZSC51", 149579), "CSCA": ("IE00B52SF786", 165547), "WSML": ("IE00BF4RFH31", 178312),
    "CSEMU": ("IE00B53QG562", 161071), "SSAC": ("IE00B6R52259", 165644), "IWRD": ("IE00B0M62Q58", 80263)}


def step_dsetf(con):
    isins = [v[0] for v in ETF_ISIN.values()]
    con.execute("CREATE OR REPLACE TEMP TABLE ei AS SELECT unnest(?) AS isin", [isins])
    n = r2.sql(con, """SELECT DISTINCT infocode::BIGINT AS infocode, isin, isprimqt, dsqtname, region
                       FROM tr_ds_equities.wrds_ds_names WHERE isin IN (SELECT isin FROM ei)""")
    con.execute("CREATE OR REPLACE TEMP TABLE ic AS SELECT unnest(?)::DECIMAL(11,0) AS v", [n.infocode.astype(float).tolist()])
    a = r2.sql(con, """SELECT infocode::BIGINT AS infocode, marketdate, currency, close::DOUBLE AS close,
                              numshrs::DOUBLE AS numshrs, mktcap_usd::DOUBLE AS mktcap_usd
                       FROM tr_ds_equities.wrds_ds2dsf
                       WHERE infocode IN (SELECT v FROM ic) AND marketdate >= DATE '2017-06-01'""")
    a = a.merge(n, on="infocode", how="left")
    inv = {v[0]: k for k, v in ETF_ISIN.items()}
    a["etf"] = a["isin"].map(inv)
    a["fundno"] = a.etf.map({k: v[1] for k, v in ETF_ISIN.items()})
    save(a, "ds_etf")


STEPS["dsetf"] = step_dsetf


# ---- historical ISIN -> current ISIN (Datastream ds2isinchg), for matching fund lines to the sister panels ----
def step_isinmap(con):
    import json
    from common import ISHCH
    isins = set()
    for f in ISHCH.glob("*.json"):
        for r in json.loads(f.read_text(encoding="utf-8-sig")).get("aaData", []):
            if len(r) >= 13:
                v = r[8 + len(r) - 13]
                if isinstance(v, str) and len(v) == 12:
                    isins.add(v)
    con.execute("CREATE OR REPLACE TEMP TABLE hi AS SELECT unnest(?) AS isin", [sorted(isins)])
    d = r2.sql(con, """WITH s AS (SELECT DISTINCT dsseccode FROM tr_ds_equities.ds2isinchg WHERE isin IN (SELECT isin FROM hi))
                       SELECT dsseccode::BIGINT AS dsseccode, isin, startdate, enddate
                       FROM tr_ds_equities.ds2isinchg WHERE dsseccode IN (SELECT dsseccode FROM s)""")
    cur = d.sort_values("startdate").groupby("dsseccode").isin.last().rename("isin_now")
    m = d[d["isin"].isin(isins)].merge(cur, on="dsseccode")[["isin", "dsseccode", "isin_now"]].drop_duplicates("isin")
    save(m, "isin_map")


STEPS["isinmap"] = step_isinmap


if __name__ == "__main__":
    todo = sys.argv[1:] or list(STEPS)
    con = r2.connect()
    for s in todo:
        t0 = time.time()
        print(s, flush=True)
        STEPS[s](con)
        print(f"  {time.time() - t0:.0f}s", flush=True)
