"""Step 3: visible passive share pi_visible per index family per date (tracker AUM / index market value).

Daily, from Datastream (method of raw/audits/passive_share_estimates.md and flow_sizing.md §4.4):
  AUM = wrds_ds2dsf.mktcap (close x numshrs) of the headline physical trackers; MV = ds2indexdata.mv (level x divisor,
  the provider's float-adjusted index cap, millions). Families and trackers:
    TSX (S&P/TSX Capped Composite)  XIC, ZCN, HXCN                 MV 48552
    TSX60                           XIU, ZIU, HXT                  MV 48553
    SP500                           SPY, IVV, VOO                  MV 41620
    STOXX                           EXSA, MEUD, XSX6, XSX7, EXIE, Lyxor C060, STT6     MV 13635
    DAX / MDAX / SDAX / TecDAX      as ../close_dislocation/s01_pull_data.py           MV 13095 / 40753 / 46391 / 41379
    FTSE100 / FTSE250               ISF, VUKE / MIDD, VMID         MV 36888 / 36890
  (TSX, STOXX and DAX-family AUM and MV are read from ../close_dislocation/data; the rest from data/ds_*_extra.)
Monthly, from CRSP MF (US only):
    R2000 / R1000 / R3000 ETF lines (IWM+VTWO, IWB+VONE, IWV+VTHR) and ALL US-registered pure index funds whose name
    tracks the S&P 500 / Russell 1000 / 2000 / 3000 (index_fund_flag = 'D', growth/value/leveraged/ESG etc. excluded).
    Russell MV is holdings-implied: the iShares fund holds a fraction f of every member's float, and f is the 90th
    percentile of (fund shares / CRSP shares outstanding) across its holdings (full-float names; validated on the
    S&P 500 in the audit); MV = fund equity value / f.

Frozen share counts: a tracker whose Datastream numshrs has not changed for >= 120 sessions is flagged stale on that
day. pi_visible uses all trackers (stale ones at their stale count); `stale_share` reports the AUM share that is stale,
and `pi_visible_live` drops the stale trackers. The check against independent AUM (XIC/XIU holdings files, CRSP TNA)
is in s09_checks.py.
Output: data/pi_daily.parquet (family, date, aum, mv, pi_visible, pi_visible_live, stale_share, n_trackers) and
data/pi_us_monthly.parquet (family, date, pi_etf, pi_allfunds, mv, src).
"""
import re
import numpy as np
import pandas as pd
import duckdb
from common import DATA, CD, RUS

FAM_CCY = {"TSX": "CAD", "TSX60": "CAD", "SP500": "USD", "R2000": "USD", "STOXX": "EUR", "DAX": "EUR", "MDAX": "EUR",
           "SDAX": "EUR", "TecDAX": "EUR", "FTSE100": "GBP", "FTSE250": "GBP"}
STALE_RUN = 120


def daily():
    a = pd.concat([pd.read_parquet(CD / "data" / "etf_aum.parquet"), pd.read_parquet(DATA / "ds_etf_aum_extra.parquet")])
    a["marketdate"] = pd.to_datetime(a.marketdate)
    a = a.dropna(subset=["mktcap"]).sort_values(["infocode", "marketdate"])
    # stale flag: run length of unchanged numshrs (in trading rows of that fund)
    chg = a.groupby("infocode").numshrs.diff().fillna(1).ne(0)
    grp = chg.groupby(a.infocode).cumsum()
    a["run"] = a.groupby([a.infocode, grp]).cumcount()
    a["stale"] = a.run >= STALE_RUN
    # local-currency AUM: lines in another currency go through mktcap_usd and the family's USD rate on the day
    rows = []
    for fam, g in a.groupby("family"):
        ccy = FAM_CCY[fam]
        loc = g[g.currency == ccy]
        rate = (loc.mktcap / loc.mktcap_usd).groupby(loc.marketdate).median()
        g = g.copy()
        g["aum"] = np.where(g.currency == ccy, g.mktcap, g.mktcap_usd * g.marketdate.map(rate))
        d = g.groupby("marketdate").agg(aum=("aum", "sum"), n_trackers=("aum", "size"),
                                        aum_stale=("aum", lambda s: s[g.loc[s.index, "stale"]].sum()))
        d["family"] = fam
        rows.append(d.reset_index())
    A = pd.concat(rows)
    m = pd.concat([pd.read_parquet(CD / "data" / "index_mv.parquet"), pd.read_parquet(DATA / "ds_index_mv_extra.parquet")])
    m["valuedate"] = pd.to_datetime(m.valuedate)
    m = m.dropna(subset=["mv"])
    x = A.merge(m[["family", "valuedate", "mv"]].rename(columns={"valuedate": "marketdate"}), on=["family", "marketdate"])
    x["pi_visible"] = x.aum / (x.mv * 1e6)
    x["pi_visible_live"] = (x.aum - x.aum_stale) / (x.mv * 1e6)
    x["stale_share"] = x.aum_stale / x.aum
    x = x.rename(columns={"marketdate": "date"})
    x[["family", "date", "aum", "mv", "pi_visible", "pi_visible_live", "stale_share", "n_trackers"]].to_parquet(
        DATA / "pi_daily.parquet", index=False)
    return x


EXCL = re.compile(r"GROWTH|VALUE|EQUAL|ESG|LOW VOL|DIVIDEND|HIGH|PURE|EX-|SECTOR|LEVERAG|ULTRA|INVERSE|BEAR|BULL|"
                  r"2X|3X|MOMENTUM|QUALITY|ENHANCED|COVERED|BUFFER|HEDGED|DOUBLE|SCREEN|SHARIAH|CLIMATE|FOSSIL|"
                  r"CATHOLIC|PARIS|MIDCAP|MID-CAP|MID CAP|SMALLCAP|400|600|1500|TOP|CAPPED|DYNAMIC|OPTION|INCOME|"
                  r"PROTECT|DEFINED|TARGET|RISK|MANAGED|STRATEGY|STRATEGIC|ALLOCATION|PLUS|TILT|SELECT|FUTURES|"
                  r"PROSHARES|DIREXION|RYDEX|GUGGENHEIM|COLLATERAL|TAX|TR\b|NASDAQ|GLOBAL|INTERNATIONAL|WORLD|"
                  r"FIXED|BOND|TREASURY|MORTGAGE|SHORT|LONG|LADDER|EQUALIZED|MAX|ACCELERATED|POWER|GEARED")
PATS = {"SP500": re.compile(r"S&P 500|\b500 INDEX\b|500 STOCK INDEX|S&P500"),
        "R1000": re.compile(r"RUSSELL 1000\b"), "R2000": re.compile(r"RUSSELL 2000\b"),
        "R3000": re.compile(r"RUSSELL 3000\b")}
ETF_FUNDNO = {"R2000": ["IWM", "VTWO"], "R1000": ["IWB", "VONE"], "R3000": ["IWV", "VTHR"], "SP500": ["SPY", "IVV", "VOO"]}
RUL = {"R2000": "IWM", "R1000": "IWB", "R3000": "IWV", "SP500": "IVV"}


def us_monthly():
    f = pd.read_parquet(DATA / "us_index_funds.parquet")
    t = pd.read_parquet(DATA / "us_index_tna.parquet")
    t["caldt"] = pd.to_datetime(t.caldt)
    t["me"] = t.caldt + pd.offsets.MonthEnd(0)
    nm = f.fund_name.fillna("").str.upper()
    f["fam"] = None
    for fam, p in PATS.items():
        hit = nm.str.contains(p) & ~nm.str.contains(EXCL)
        f.loc[hit & f.fam.isna(), "fam"] = fam
    f.loc[f.ticker.isin(sum(ETF_FUNDNO.values(), [])), "fam"] = f.loc[f.ticker.isin(sum(ETF_FUNDNO.values(), [])), "ticker"].map(
        {tk: fam for fam, tks in ETF_FUNDNO.items() for tk in tks})
    f[f.fam.notna()][["fundno", "portno", "fund_name", "ticker", "et_flag", "fam"]].to_csv(DATA / "us_index_fund_map.csv", index=False)
    t = t.merge(f[["fundno", "fam", "ticker"]], on="fundno").dropna(subset=["fam"])
    t = t[t.mtna > 0]
    allf = t.groupby(["fam", "me"]).mtna.sum().rename("tna_all") * 1e6          # mtna is $ millions
    etf = t[t.ticker.isin(sum(ETF_FUNDNO.values(), []))]
    etf = etf[[tk in ETF_FUNDNO[fm] for tk, fm in zip(etf.ticker, etf.fam)]]
    etf = etf.groupby(["fam", "me"]).mtna.sum().rename("tna_etf") * 1e6
    # holdings-implied index MV from the iShares ruler fund
    h = pd.read_parquet(DATA / "us_holdings.parquet", columns=["fund", "report_dt", "permno", "nbr_shares", "market_val"])
    h["report_dt"] = pd.to_datetime(h.report_dt)
    h = h[h.fund.isin(RUL.values()) & h.permno.notna() & (h.nbr_shares > 0)]
    c = duckdb.connect()
    so = c.execute(f"""SELECT permno, dlycaldt, shrout, prc FROM read_parquet('{RUS}/_cache/dsf/*.parquet')
                       WHERE dlycaldt >= '2010-06-01'""").df()
    so["me"] = so.dlycaldt + pd.offsets.MonthEnd(0)
    so = so.sort_values("dlycaldt").groupby(["permno", "me"]).tail(1)[["permno", "me", "shrout", "prc"]]
    h["me"] = h.report_dt + pd.offsets.MonthEnd(0)
    h = h.merge(so, on=["permno", "me"], how="left")
    h["own"] = h.nbr_shares / (h.shrout * 1000)
    h["val"] = h.nbr_shares * h.prc.abs()
    g = h.groupby(["fund", "me"]).agg(f90=("own", lambda s: s.quantile(0.90)), val=("val", "sum"), n=("own", "size"))
    g["mv_implied"] = g.val / g.f90
    rows = []
    inv = {v: k for k, v in RUL.items()}
    for (fund, me), r in g.iterrows():
        rows.append(dict(family=inv[fund], date=me, mv=r.mv_implied, f_ruler=r.f90, ruler_val=r.val, n_hold=r.n))
    M = pd.DataFrame(rows)
    # S&P 500: use the official Datastream MV at month-end instead of the implied one (kept for the check)
    mv = pd.read_parquet(DATA / "ds_index_mv_extra.parquet")
    mv = mv[mv.family == "SP500"].dropna(subset=["mv"])
    mv["valuedate"] = pd.to_datetime(mv.valuedate)
    mv["me"] = mv.valuedate + pd.offsets.MonthEnd(0)
    mv = mv.sort_values("valuedate").groupby("me").tail(1).set_index("me").mv * 1e6
    M["mv_official"] = np.where(M.family == "SP500", M.date.map(mv), np.nan)
    M["mv_used"] = np.where(M.family == "SP500", M.mv_official, M.mv)
    M = M.merge(allf.reset_index().rename(columns={"fam": "family", "me": "date"}), on=["family", "date"], how="left")
    M = M.merge(etf.reset_index().rename(columns={"fam": "family", "me": "date"}), on=["family", "date"], how="left")
    M["pi_etf"] = M.tna_etf / M.mv_used
    M["pi_allfunds"] = M.tna_all / M.mv_used
    M["src"] = np.where(M.family == "SP500", "CRSP MF TNA / Datastream MV 41620",
                        "CRSP MF TNA / MV implied from " + M.family.map(RUL) + " holdings (p90 of held/shrout)")
    M.to_parquet(DATA / "pi_us_monthly.parquet", index=False)
    return M


if __name__ == "__main__":
    x = daily()
    pd.set_option("display.width", 250)
    x["y"] = x.date.dt.year
    print(x.groupby(["family", "y"]).pi_visible.median().unstack(0).round(5).to_string())
    print(x.groupby(["family", "y"]).stale_share.median().unstack(0).round(2).to_string())
    M = us_monthly()
    M["y"] = M.date.dt.year
    print(M.groupby(["family", "y"])[["pi_etf", "pi_allfunds", "f_ruler"]].median().unstack(0).round(4).to_string())
