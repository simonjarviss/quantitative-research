"""Step 5: visible pi as a dated series per family (quarter-ends from s03, extended past the last s12 quarter with
Datastream ETF share counts), used by the demand calculation (s06) and written to pi_calibration.csv (s07).

Families (holders' pi: every visible fund that holds the segment's names; see s03):
  MSCI_CANADA      Standard + IMI trackers holding MSCI Canada Standard names
  MSCI_CANADA_SC   Small Cap + IMI trackers holding MSCI Canada Small Cap names
  MSCI_EUROPE      Standard + IMI trackers of DM-Europe Standard names (ruler-weight-weighted mean over countries)
  MSCI_EUROPE_SC   Small Cap + IMI trackers of DM-Europe Small Cap names
  MSCI_EUROPE_<CC> Standard holders of one country's names (adds the country and EMU funds): GB FR DE CH NL SE ES IT DK
                   BE NO AT XE (XE = other DM Europe in CINS: Ireland, Finland, Portugal)
Extension: for each date t after 2025-12-31, k_f(t) = k_f(2025Q4) x numshrs_f(t) / numshrs_f(2025-12-31) for the 21
iShares ETFs with Datastream share counts (s02 dsetf); the other funds move with the k-weighted mean growth of those
ETFs. The Canadian Small Cap series starts 2019-03 (no ruler before): earlier dates take the 2019-03 value (flagged).
Output: data/pi_series.parquet (family, date, pi_visible, pi_imi_part, n_funds, note).
"""
import numpy as np
import pandas as pd

from common import DATA

CTRY = ["GB", "FR", "DE", "CH", "NL", "SE", "ES", "IT", "DK", "BE", "NO", "AT", "XE"]
REGIONAL = {2074, 165547, 55899, 161071, 91849, 167255, 149579}     # EWC, CSCA, EZU, CSEMU, IEUR, IMEA, IMEU


def growth():
    a = pd.read_parquet(DATA / "ds_etf.parquet")
    a["marketdate"] = pd.to_datetime(a.marketdate)
    a = a[(a.isprimqt == 1) & a.numshrs.notna() & (a.etf != "SSAC")].sort_values("marketdate")
    base = a[a.marketdate <= "2025-12-31"].groupby("fundno").numshrs.last()
    return a, base


def main():
    F = pd.read_parquet(DATA / "pi_funds.parquet")
    P = pd.read_parquet(DATA / "pi_quarterly.parquet")
    a, base = growth()
    last_q = P.q.max()
    ext_dates = sorted(set(pd.to_datetime(["2026-02-27", "2026-03-31", "2026-05-29", "2026-06-30"])) | {a.marketdate.max()})
    rows = []

    def fam_rows(seg, mkt, ctries, fam):
        Pm = P[(P.seg == seg) & (P.mkt == mkt) & P.ctry.isin(ctries)]
        for q, g in Pm.groupby("q"):
            w = g.W / g.W.sum()
            rows.append(dict(family=fam, date=q, pi_visible=float((g.pi * w).sum()),
                             pi_imi_part=float((g.pi_imi * w).sum()), n_funds=int(g.n_funds.max()), note=""))
        # extension
        Fl = F[(F.q == last_q) & (F.seg == seg) & (F.mkt == mkt) & F.ctry.isin(ctries)]
        gl = Pm[Pm.q == last_q]
        if Fl.empty:
            return
        wc = (gl.set_index("ctry").W / gl.W.sum())
        Fl = Fl.assign(wk=Fl.k * Fl.ctry.map(wc))
        for d in ext_dates:
            ns = a[a.marketdate <= d].groupby("fundno").numshrs.last()
            gr = (ns / base).dropna()
            cov = Fl[Fl.fundno.isin(gr.index)]
            if cov.empty:
                continue
            broad = cov[~cov.fundno.isin(REGIONAL)]          # mutual funds follow the broad (global/EAFE) ETFs
            g_mean = float((broad.wk * broad.fundno.map(gr)).sum() / broad.wk.sum()) if len(broad) else 1.0
            k_new = np.where(Fl.fundno.isin(gr.index), Fl.fundno.map(gr), g_mean) * Fl.wk
            imi = (Fl.ftype == "IMI").values
            rows.append(dict(family=fam, date=d, pi_visible=float(k_new.sum()), pi_imi_part=float(k_new[imi].sum()),
                             n_funds=int(Fl.fundno.nunique()),
                             note=f"extended from {last_q.date()} by Datastream ETF share counts ({cov.wk.sum() / Fl.wk.sum():.0%} of pi; others at the mean ETF flow {g_mean - 1:+.1%})"))

    fam_rows("STD", "CA", ["CA"], "MSCI_CANADA")
    fam_rows("SC", "CA", ["CA"], "MSCI_CANADA_SC")
    fam_rows("STD", "EU", CTRY, "MSCI_EUROPE")
    fam_rows("SC", "EU", CTRY, "MSCI_EUROPE_SC")
    for c in CTRY:
        fam_rows("STD", "EU", [c], f"MSCI_EUROPE_{c}")
    S = pd.DataFrame(rows).sort_values(["family", "date"])
    # Canadian Small Cap before the first ruler quarter: back-fill with the first value (flagged)
    sc = S[S.family == "MSCI_CANADA_SC"]
    first = sc.dropna(subset=["pi_visible"]).iloc[0]
    fill = []
    for q in sorted(P.q.unique()):
        if q < first.date:
            fill.append(dict(family="MSCI_CANADA_SC", date=q, pi_visible=first.pi_visible, pi_imi_part=first.pi_imi_part,
                             n_funds=first.n_funds, note=f"no Small Cap ruler before {first.date.date()}: first value carried back"))
    S = pd.concat([S, pd.DataFrame(fill)], ignore_index=True).sort_values(["family", "date"])
    # quarters where a ruler was missing (e.g. Canada Small Cap 2022-03: no WSML filing) -> interpolate
    S["pi_visible"] = S.groupby("family").pi_visible.transform(lambda s: s.interpolate(limit_area="inside"))
    S.to_parquet(DATA / "pi_series.parquet", index=False)
    pd.set_option("display.width", 250)
    t = S.pivot_table(index="date", columns="family", values="pi_visible") * 100
    print(t[["MSCI_CANADA", "MSCI_CANADA_SC", "MSCI_EUROPE", "MSCI_EUROPE_SC", "MSCI_EUROPE_GB", "MSCI_EUROPE_DE",
             "MSCI_EUROPE_FR", "MSCI_EUROPE_CH"]].round(3).to_string())


def pi_at(S, fam, d):
    """Linear interpolation of a family's visible pi on date d (last value after the series ends)."""
    s = S[S.family == fam].dropna(subset=["pi_visible"]).set_index("date").pi_visible.sort_index()
    d = pd.Timestamp(d)
    if d <= s.index[0]:
        return float(s.iloc[0])
    if d >= s.index[-1]:
        return float(s.iloc[-1])
    i = s.index.searchsorted(d)
    d0, d1 = s.index[i - 1], s.index[i]
    w = (d - d0) / (d1 - d0)
    return float(s.iloc[i - 1] * (1 - w) + s.iloc[i] * w)


if __name__ == "__main__":
    main()
