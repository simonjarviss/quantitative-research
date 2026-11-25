"""Rebuild catalog/tables.csv and catalog/columns.csv from the bucket itself.

Read-only. Parquet objects are described from their footers (schema, row
counts, min/max statistics), so no table data is downloaded; the small CSVs
under Data_Jayden/ are read in full, through r2.source() so their types match
what r2.sql returns. The bucket name never appears in the output: paths are
bucket-relative keys (e.g. wrds/crsp_a_stock/dsf.parquet).

    python3 -m r2.build_catalog     # from the folder holding r2/; ~90 s; credentials from r2/r2.env

The warehouse notes explain the layout, identifiers and caveats; the
DATASETS table below is the per-dataset part of that schema and feeds the
`domain` column.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

import r2

OUT = Path(__file__).resolve().parent / "catalog"

# dataset -> (domain, status, description). status: licensed | trial | sample | duplicate
DATASETS = {
    "auditsmp_all": ("governance", "sample", "Audit Analytics sample: auditor changes, fees, opinions, SOX 302/404"),
    "bank_all": ("banks", "licensed", "Bank regulatory: FFIEC call reports, FR Y-9C holding companies, NIC structure, bank-CRSP link"),
    "block_all": ("ownership", "licensed", "Blockholder data for US firms, 1996-2001 (three versions)"),
    "boardex_trial": ("governance", "trial", "BoardEx trial: boards, directors, networks, compensation"),
    "bvd_amadeus_trial": ("fundamentals_global", "trial", "Bureau van Dijk Amadeus trial"),
    "bvd_bvdbankf_trial": ("banks", "trial", "Bureau van Dijk BankFocus trial"),
    "bvd_orbis_trial": ("fundamentals_global", "trial", "Bureau van Dijk Orbis trial"),
    "calcbench_trial": ("fundamentals_us", "trial", "Calcbench XBRL trial"),
    "cboe_all": ("derivatives", "licensed", "CBOE volatility indices, daily OHLC: VIX, VXO, VXN, VXD"),
    "comp_execucomp": ("governance", "licensed", "ExecuComp executive and director compensation"),
    "comp_global_daily": ("fundamentals_global", "licensed", "Compustat Global: fundamentals, daily/monthly security prices, FX, indices, code tables"),
    "comp_na_daily_all": ("fundamentals_us", "licensed", "Compustat North America: fundamentals, headers, daily/monthly prices, segments, short interest, index constituents, WRDS ratios, code tables"),
    "contrib_as_filed_financials": ("fundamentals_us", "licensed", "As-filed XBRL financial statement tags"),
    "contrib_bond_firm_link": ("links", "licensed", "Bond issuer CUSIP to GVKEY/PERMNO, monthly"),
    "contrib_ceo_turnover": ("governance", "licensed", "CEO turnover events"),
    "contrib_char_returns": ("research", "licensed", "Characteristic-based benchmark returns by PERMNO"),
    "contrib_corp_fed_litigation": ("legal", "licensed", "Corporate federal litigation, firm-year and lawsuit level"),
    "contrib_corporate_culture": ("governance", "licensed", "Text-based corporate culture scores"),
    "contrib_general": ("research", "licensed", "Contributed research data: common ownership, better beta, CIK types, classified boards, marginal tax rates, risk-factor text, XBRL complexity, blockholders, shale wells, international factors"),
    "contrib_global_factor": ("research", "licensed", "Jensen-Kelly-Pedersen global factor data: stock characteristics, daily returns, factor definitions"),
    "contrib_intangible_value": ("research", "licensed", "Intangible-adjusted value factors"),
    "contrib_kpss": ("research", "licensed", "Kogan-Papanikolaou-Seru-Stoffman patent values and PERMCO link"),
    "contrib_liva": ("research", "licensed", "LIVA (same file as contrib_general.liva)"),
    "crsp_a_ccm": ("links", "licensed", "CRSP/Compustat Merged: GVKEY-PERMNO link history, CCM company/security headers"),
    "crsp_a_stock": ("equity_us", "licensed", "CRSP US stocks (annual update): daily/monthly prices and returns in legacy (to 2024-12) and v2/CIZ (to 2025-12) formats, names, distributions, delistings, shares, indices"),
    "crsp_q_indexhist": ("equity_us", "licensed", "CRSP index history (quarterly update): intraday and EOD levels of CRSP market-cap, sector, value/growth and ESG indices"),
    "crsp_q_mutualfunds": ("ownership", "licensed", "CRSP survivor-bias-free mutual funds: NAV, returns, TNA, holdings, fees, flows, headers"),
    "djones_all": ("equity_us", "licensed", "Dow Jones averages, daily/monthly, to 2008"),
    "dmef_all": ("other", "licensed", "DMEF academic direct-marketing datasets"),
    "doe_all": ("equity_us", "licensed", "Rule 11Ac1-5 disclosure of order execution statistics, 2001-2005"),
    "ff_all": ("research", "licensed", "Fama-French factors (3/5 + momentum), portfolios, industry definitions, liquidity factors"),
    "fjc_linking": ("links", "licensed", "Federal Judicial Center cases to firms (bankruptcy, civil, criminal)"),
    "fjc_litigation": ("legal", "licensed", "Federal Judicial Center Integrated Database: appeals, bankruptcy, civil, criminal"),
    "frb_all": ("macro_rates_fx", "licensed", "Federal Reserve H.15 rates and H.10 FX, daily/monthly"),
    "ims_obp_trial": ("other", "trial", "IMS trial, 50 personal contact records"),
    "iri_all": ("other", "licensed", "IRI annual data books 1983-1997, page/row layout"),
    "macrofin_comm_trade": ("research", "licensed", "Hou-Xue-Zhang q-factors and portfolios, cay, macro/financial uncertainty, commodity trade and FX portfolios"),
    "msrb_all": ("fixed_income", "licensed", "MSRB municipal bond transactions"),
    "otc_endofday": ("equity_us", "licensed", "OTC Markets end-of-day quotes and volume"),
    "phlx_all": ("derivatives", "licensed", "PHLX currency options: daily trades and implied vols, to 1998"),
    "pitchsmp": ("private_markets", "sample", "PitchBook sample"),
    "public_all": ("other", "licensed", "MEPS health expenditure summary tables"),
    "pwt_all": ("macro_rates_fx", "licensed", "Penn World Table 6.1-7.1"),
    "rq_all": ("research", "licensed", "Research Quotient: R&D productivity by GVKEY-year"),
    "tfn": ("ownership", "duplicate", "s12 only; same file as tr_mutualfunds.s12"),
    "totalq_all": ("research", "licensed", "Peters-Taylor total Q and intangible capital"),
    "tr_13f": ("ownership", "licensed", "Thomson/Refinitiv 13F institutional holdings (s34)"),
    "tr_common": ("links", "licensed", "Refinitiv PermID security master and cross-reference (quote/instrument/org; ISIN, CUSIP, SEDOL, RIC)"),
    "tr_ds_comds": ("macro_rates_fx", "licensed", "Datastream commodity prices"),
    "tr_ds_econ": ("macro_rates_fx", "licensed", "Datastream economic time series"),
    "tr_ds_equities": ("equity_global", "licensed", "Datastream global equities: daily prices/returns, market values, dividends, shares, indices and constituents"),
    "tr_ds_fut": ("derivatives", "licensed", "Datastream futures: contracts, continuous series, COT"),
    "tr_ibes": ("analysts", "licensed", "I/B/E/S detail and summary estimates (US/international, adjusted/unadjusted), actuals, surprises, recommendations, price targets"),
    "tr_insiders": ("governance", "licensed", "Thomson/Refinitiv insider filings: Forms 3/4/5, Form 144, 10b5-1"),
    "tr_mutualfunds": ("ownership", "licensed", "Thomson/Refinitiv s12 mutual fund holdings"),
    "tr_sdc_joint_ventures": ("events", "licensed", "SDC joint ventures and alliances"),
    "tr_sdc_ma": ("events", "licensed", "SDC mergers and acquisitions"),
    "tr_sdc_ni": ("events", "licensed", "SDC new issues"),
    "tr_worldscope": ("fundamentals_global", "licensed", "Worldscope global fundamentals (item-coded), company and stock data"),
    "trace": ("fixed_income", "duplicate", "trace_enhanced only; same file as trace_enhanced.trace_enhanced"),
    "trace_enhanced": ("fixed_income", "licensed", "TRACE Enhanced corporate bond trades (uncapped), agency, 144A and securitized products, master files"),
    "trace_standard": ("fixed_income", "licensed", "TRACE standard disseminated trades and daily trade summaries"),
    "wrdsapps_eushort": ("equity_global", "licensed", "EU net short position disclosures"),
    "wrdsapps_evtstudy_int": ("research", "licensed", "WRDS event-study inputs, international"),
    "wrdsapps_evtstudy_lr": ("research", "licensed", "WRDS long-run event-study inputs"),
    "wrdsapps_evtstudy_us": ("research", "licensed", "WRDS event-study inputs, US"),
    "wrdsapps_finratio": ("fundamentals_us", "licensed", "WRDS financial ratios, firm level"),
    "wrdsapps_finratio_ccm": ("fundamentals_us", "licensed", "WRDS financial ratios, CCM-keyed"),
    "wrdsapps_finratio_ibes": ("fundamentals_us", "licensed", "WRDS financial ratios, IBES-keyed"),
    "wrdsapps_finratio_ibes_ccm": ("fundamentals_us", "licensed", "WRDS financial ratios, IBES+CCM-keyed"),
    "wrdsapps_link_comp_eushort": ("links", "licensed", "Compustat to EU short positions"),
    "wrdsapps_link_crsp_bond": ("links", "licensed", "Bond CUSIP (TRACE) to PERMNO"),
    "wrdsapps_link_crsp_factset": ("links", "licensed", "FactSet to PERMNO"),
    "wrdsapps_link_crsp_ibes": ("links", "licensed", "IBES ticker to PERMNO"),
    "wrdsapps_link_crsp_taq": ("links", "licensed", "TAQ symbol to PERMNO, to 2014"),
    "wrdsapps_link_datastream_wscope": ("links", "licensed", "Datastream INFOCODE to Worldscope code"),
    "wrdsapps_patents": ("research", "licensed", "US patents, citations and GVKEY links"),
    "wrdsapps_plink_exec_trinsider": ("links", "licensed", "ExecuComp executive to Thomson insider person"),
    "wrdsapps_subsidiary": ("other", "licensed", "Exhibit 21 subsidiaries from SEC filings"),
    "wrdsapps_windices": ("equity_global", "licensed", "WRDS world country indices and constituents"),
    "wrdssec_midas": ("equity_us", "licensed", "SEC MIDAS market-structure metrics by security-day"),
    "options": ("derivatives", "licensed", "Option trade ticks on ETF underlyings, one file per symbol-month"),
    "desk": ("macro_rates_fx", "desk", "Desk CSVs: UST/NY Fed rates, Eris and Treasury futures settlements, DV01"),
}

# Grains checked on 2026-09-29: no two rows share these values. NULLs in a
# key column are noted in the warehouse notes. Tables whose obvious key is NOT unique
# (cboe, IBES statsum/act, wrds_*v2_query, bondcrsp_link) are left out on purpose.
GRAINS = {
    "wrds/crsp_a_stock/dsf": "permno, date",
    "wrds/crsp_a_stock/dsf_v2": "permno, dlycaldt",
    "wrds/crsp_a_stock/msf": "permno, date",
    "wrds/crsp_a_stock/msf_v2": "permno, mthcaldt",
    "wrds/crsp_a_stock/stocknames": "permno, namedt",
    "wrds/crsp_a_stock/dsenames": "permno, namedt",
    "wrds/crsp_a_stock/dsedelist": "permno",
    "wrds/crsp_a_stock/stksecurityinfohist": "permno, secinfostartdt",
    "wrds/crsp_a_ccm/ccmxpf_lnkhist": "gvkey, liid, linkdt",
    "wrds/comp_na_daily_all/company": "gvkey",
    "wrds/comp_na_daily_all/security": "gvkey, iid",
    "wrds/comp_na_daily_all/funda": "gvkey, datadate, indfmt, datafmt, popsrc, consol",
    "wrds/comp_na_daily_all/fundq": "gvkey, datadate, indfmt, datafmt, popsrc, consol, fyr",
    "wrds/comp_na_daily_all/secd": "gvkey, iid, datadate",
    "wrds/comp_global_daily/g_funda": "gvkey, datadate, indfmt, datafmt, popsrc, consol",
    "wrds/tr_ibes/id": "ticker, sdates",
    "wrds/tr_ibes/recdsum": "ticker, statpers",
    "wrds/wrdsapps_link_crsp_ibes/ibcrsphist": "ticker, permno, sdate",
    "wrds/wrdsapps_link_datastream_wscope/ds2ws_linktable": "infocode, code, startdate",
    "wrds/crsp_q_mutualfunds/fund_hdr": "crsp_fundno",
    "wrds/crsp_q_mutualfunds/monthly_returns": "crsp_fundno, caldt",
    "wrds/crsp_q_mutualfunds/daily_nav": "crsp_fundno, caldt",
    "wrds/tr_13f/s34": "fdate, mgrno, cusip",
    "wrds/tr_ds_equities/wrds_ds_names": "infocode, startdate",
    "wrds/tr_worldscope/wrds_ws_funda": "code, year_, freq, seq",
    "wrds/contrib_global_factor/global_factor": "id, eom",
    "wrds/ff_all/factors_daily": "date",
    "wrds/frb_all/rates_daily": "date",
    "Data_Jayden/rates/*.csv": "observation_date, series_id",
    "Data_Jayden/futures/*.csv": "observation_date, instrument_id",
    "Data_Jayden/contract_risk/*.csv": "observation_date, instrument_id",
}

# First match wins when picking a table's main date column.
DATE_PRIORITY = ["date", "caldt", "dlycaldt", "mthcaldt", "datadate", "marketdate", "trd_exctn_dt", "trans_dt",
                 "trade_date", "statpers", "anndats", "actdats", "fdate", "rdate", "trandate", "pends", "fpedats",
                 "exdt", "dlstdt", "namedt", "linkdt", "sdates", "sdate", "startdate", "date_", "ts",
                 "observation_date"]

NUMERIC_TYPE = r"^(DOUBLE|FLOAT|DECIMAL|U?(TINY|SMALL|BIG|HUGE)?INT(EGER)?)"
REF_PREFIXES = ("r_",)
META_PREFIXES = ("dd_", "xfl_", "meta", "g_tmptable")


def role_of(dataset: str, table: str) -> str:
    if dataset == "_recycle_bin":
        return "recycle_bin"
    if table.startswith(REF_PREFIXES) and dataset.startswith("comp_"):
        return "reference"
    if table.startswith(META_PREFIXES) or table.endswith("_qvards"):
        return "metadata"
    if table.endswith("_legacy"):
        return "legacy"
    if DATASETS.get(dataset, ("",))[0] == "links" or "link" in table:
        return "link"
    return "data"


def parquet_footers(con, glob: str, bucket_prefix: str):
    strip = f"replace(file_name, '{bucket_prefix}', '')"
    files = con.execute(f"""
        SELECT {strip} AS file, num_rows, file_size_bytes FROM parquet_file_metadata('{glob}')""").df()
    schema = con.execute(f"""
        SELECT {strip} AS file, column_id - 1 AS pos, name, duckdb_type AS type
        FROM parquet_schema('{glob}') WHERE column_id > 0""").df()
    stats = con.execute(f"""
        SELECT {strip} AS file, column_id AS pos,
               min(stats_min_value) AS min_str, max(stats_max_value) AS max_str,
               min(try_cast(stats_min_value AS DOUBLE)) AS min_num,
               max(try_cast(stats_max_value AS DOUBLE)) AS max_num,
               sum(stats_null_count) AS nulls, count(*) = count(stats_null_count) AS nulls_complete
        FROM parquet_metadata('{glob}') GROUP BY ALL""").df()
    return files, schema.merge(stats, on=["file", "pos"], how="left")


def column_rows(cols: pd.DataFrame, num_rows: pd.Series) -> pd.DataFrame:
    """min/max for dates, times and numbers only: string stats are truncated and can hold names."""
    c = cols.copy()
    is_time = c["type"].str.startswith(("DATE", "TIMESTAMP", "TIME"))
    is_num = c["type"].str.match(NUMERIC_TYPE)
    c["min"] = c["min_str"].where(is_time, c["min_num"].where(is_num))
    c["max"] = c["max_str"].where(is_time, c["max_num"].where(is_num))
    n = c["file"].map(num_rows)
    c["null_pct"] = (100 * c["nulls"] / n).round(2).where(c["nulls_complete"].fillna(False) & (n > 0))
    return c


def _short(s: pd.Series) -> pd.Series:
    """Drop fractional seconds from timestamps; keep any UTC offset (options `ts` is UTC)."""
    return s.astype("string").str.replace(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+", r"\1", regex=True)


def main_date(cols: pd.DataFrame) -> pd.DataFrame:
    d = cols[cols["type"].str.startswith(("DATE", "TIMESTAMP"))].copy()
    rank = {n: i for i, n in enumerate(DATE_PRIORITY)}
    d["rk"] = d["name"].map(rank).fillna(len(rank))
    d = d.sort_values(["file", "rk", "pos"]).drop_duplicates("file")
    return d.set_index("file")[["name", "min_str", "max_str"]].rename(
        columns={"name": "date_col", "min_str": "date_min", "max_str": "date_max"})


def build(con, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    B = f"s3://{cfg.bucket}/"
    today = dt.date.today().isoformat()
    tables, columns = [], []

    # wrds/<dataset>/<table>.parquet and _recycle_bin/**.parquet
    for area, glob in (("wrds", B + "wrds/*/*.parquet"), ("_recycle_bin", B + "_recycle_bin/**/*.parquet")):
        files, cols = parquet_footers(con, glob, B)
        cols = column_rows(cols, files.set_index("file")["num_rows"])
        dates = main_date(cols)
        # Content fingerprint: names, types and per-column min/max/null counts
        # (numbers compared as numbers: row order changes the text-wise max).
        # Byte size is not enough: two all-DOUBLE tables with the same row
        # count can have identical sizes and different values.
        cols = cols.sort_values(["file", "pos"])
        num = cols["type"].str.match(NUMERIC_TYPE)
        lo = cols["min_num"].astype("string").where(num, cols["min_str"].astype("string")).fillna("")
        hi = cols["max_num"].astype("string").where(num, cols["max_str"].astype("string")).fillna("")
        fp = cols["name"] + ":" + cols["type"] + ":" + lo + ":" + hi + ":" + cols["nulls"].astype("string").fillna("")
        sig = fp.groupby(cols["file"]).agg("|".join).rename("sig")
        t = files.join(dates, on="file").join(sig, on="file")
        t["key"] = t["file"].str.removesuffix(".parquet")
        parts = t["key"].str.split("/")
        t["dataset"] = parts.str[1] if area == "_recycle_bin" else parts.str[-2]
        t["table"] = parts.str[2:].str.join("/") if area == "_recycle_bin" else parts.str[-1]
        t["dataset"] = t["dataset"] if area == "wrds" else "_recycle_bin/" + t["dataset"]
        t["area"] = area
        t["n_cols"] = t["file"].map(cols.groupby("file").size())
        tables.append(t)
        if area == "wrds":
            c = cols.merge(t[["file", "dataset", "table"]], on="file")
            c["area"] = area
            columns.append(c)

    # options/<SYM>/tick/<YYYY-MM>.parquet, rolled up to one row per symbol
    files, cols = parquet_footers(con, B + "options/*/tick/*.parquet", B)
    cols = column_rows(cols, files.set_index("file")["num_rows"])
    files["sym"] = files["file"].str.split("/").str[1]
    ts = cols[cols["name"] == "ts"].set_index("file")
    files = files.join(ts[["min_str", "max_str"]], on="file")
    for sym, g in files.groupby("sym"):
        months = g["file"].str.extract(r"(\d{4}-\d{2})")[0]
        tables.append(pd.DataFrame([dict(
            area="options", dataset="options", table=sym, file=f"options/{sym}/tick/*.parquet",
            num_rows=g["num_rows"].sum(), file_size_bytes=g["file_size_bytes"].sum(),
            date_col="ts", date_min=g["min_str"].min(), date_max=g["max_str"].max(),
            n_cols=int(cols[cols["file"] == g["file"].iloc[0]].shape[0]),
            n_files=len(g), months=f"{months.min()}..{months.max()}")]))
        first = cols[cols["file"] == g["file"].iloc[0]].copy()
        agg = cols[cols["file"].isin(g["file"])].groupby("pos").agg(
            min_=("min", "min"), max_=("max", "max"), nulls=("nulls", "sum"))
        first = first.drop(columns=["min", "max", "nulls"]).join(agg, on="pos").rename(
            columns={"min_": "min", "max_": "max"})
        first["null_pct"] = (100 * first["nulls"] / g["num_rows"].sum()).round(2)
        first["file"], first["dataset"], first["table"], first["area"] = (
            f"options/{sym}/tick/*.parquet", "options", sym, "options")
        columns.append(first)

    # Data_Jayden/<name>/*.csv, one logical table per folder, named desk.<name>
    base = B + "Data_Jayden/"
    subs = sorted({f.split("/")[0] for (f,) in con.execute(
        f"SELECT replace(file, '{base}', '') FROM glob('{base}*/*.csv')").fetchall()})
    for sub in subs:
        g = f"{base}{sub}/*.csv"
        if sub in r2.DESK_TABLES:
            rel = r2.source(f"desk.{sub}", cfg)
        else:
            print(f"desk.{sub}: not in r2.DESK_TABLES, types auto-detected; add it there", file=sys.stderr)
            rel = f"read_csv('{g}', union_by_name=true)"
        size, nfiles = con.execute(f"SELECT sum(size), count(*) FROM read_blob('{g}')").fetchone()
        summ = con.execute(f"SUMMARIZE SELECT * FROM {rel}").df()
        n = int(summ["count"].iloc[0])
        c = pd.DataFrame(dict(
            area="Data_Jayden", dataset="desk", table=sub, file=f"Data_Jayden/{sub}/*.csv",
            pos=range(len(summ)), name=summ["column_name"], type=summ["column_type"],
            min=summ["min"].where(summ["column_type"] != "VARCHAR"),
            max=summ["max"].where(summ["column_type"] != "VARCHAR"),
            null_pct=summ["null_percentage"].astype(float).round(2)))
        columns.append(c)
        od = c[c["name"] == "observation_date"]
        tables.append(pd.DataFrame([dict(
            area="Data_Jayden", dataset="desk", table=sub, file=f"Data_Jayden/{sub}/*.csv",
            num_rows=n, file_size_bytes=size, n_cols=len(c), n_files=nfiles,
            date_col="observation_date" if len(od) else None,
            date_min=od["min"].iloc[0] if len(od) else None, date_max=od["max"].iloc[0] if len(od) else None)]))

    t = pd.concat(tables, ignore_index=True)
    t["domain"] = t["dataset"].map(lambda d: DATASETS.get(d, ("recycle_bin" if d.startswith("_") else "unmapped",))[0])
    t["status"] = t["dataset"].map(lambda d: DATASETS.get(d, ("", "recycle_bin" if d.startswith("_") else ""))[1])
    t["role"] = [role_of(a if a == "_recycle_bin" else d, tb) for a, d, tb in zip(t["area"], t["dataset"], t["table"])]
    t["key"] = t["file"].str.removesuffix(".parquet")
    t["grain"] = t["key"].map(GRAINS)
    t["future_dates"] = t["date_max"].fillna("").str[:10] > today

    # Twins: same columns, row count and per-column statistics. The copy in
    # the dataset with more tables is treated as the home copy.
    w = t[t["area"] == "wrds"].copy()
    w["ds_size"] = w["dataset"].map(w["dataset"].value_counts())
    w = w.sort_values(["ds_size", "dataset", "table"], ascending=[False, True, True])
    home = w.groupby(["sig", "num_rows"])["key"].transform("first")
    t.loc[w.index, "duplicate_of"] = home.where(home != w["key"]).str.removeprefix("wrds/")
    t.loc[t["duplicate_of"].notna(), "role"] = "duplicate"

    t["size_mb"] = (t["file_size_bytes"] / 1e6).round(2)
    t["date_min"] = _short(t["date_min"])
    t["date_max"] = _short(t["date_max"])
    t = t.rename(columns={"num_rows": "rows", "file": "path"})
    t = t[["area", "domain", "dataset", "table", "status", "role", "rows", "size_mb", "n_cols", "date_col",
           "date_min", "date_max", "future_dates", "grain", "duplicate_of", "n_files", "months", "path"]]
    t = t.sort_values(["area", "dataset", "table"]).reset_index(drop=True)

    c = pd.concat(columns, ignore_index=True)
    c = c.rename(columns={"name": "column"})[["area", "dataset", "table", "pos", "column", "type", "null_pct", "min", "max"]]
    for col in ("min", "max"):
        c[col] = _short(c[col])
    c = c.sort_values(["area", "dataset", "table", "pos"]).reset_index(drop=True)
    return t, c


def main() -> int:
    cfg = r2.load_config()
    con = r2.connect(cfg)
    try:
        con.execute("SET threads=32")  # footer reads are latency-bound; more in flight
        tables, columns = build(con, cfg)
    finally:
        con.close()
    OUT.mkdir(exist_ok=True)
    tables.to_csv(OUT / "tables.csv", index=False)
    columns.to_csv(OUT / "columns.csv", index=False)
    by_area = tables.groupby("area").agg(tables=("table", "size"), rows=("rows", "sum"), gb=("size_mb", "sum"))
    by_area["gb"] = (by_area["gb"] / 1e3).round(1)
    print(by_area.to_string())
    print(f"wrote {OUT / 'tables.csv'} ({len(tables)} rows) and {OUT / 'columns.csv'} ({len(columns)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
