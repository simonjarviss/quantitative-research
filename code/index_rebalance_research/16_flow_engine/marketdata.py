"""Market data for the engine: price, FX, ADV (consolidated where possible) and the normal closing-auction volume.

Output frame (one row per line = key x country): key, country, ccy, price, adv_sh, adv_basis, close_base_sh,
close_basis, close_n_days, eur_per_ccy, cad_per_ccy, [v_T for history].

History (R2, validation): data/r2/vol_<tag>.parquet from r2_pull.py vol
  Canada  ADV on the Canadian Composite line (Datastream ds2scdqtprc exch 295 = consolidated); TSX-only if missing
  Europe  primary quote + lit MTF books (Cboe DXE/CXE/BXE, Turquoise, Aquis, Equiduct) = "lit"; primary if missing
  ADV = median over sessions T-25..T-6 of the line, event days removed (MSCI 29 May 2026 and the line's own zeros)
Live (going forward):
  Canada  TSX volume: tsxintra (intraday-trading-stats 20-day average continuous volume) + MOC from the archive, else
          the TSX shadow's yfinance cache; times the name's consolidated/TSX ratio from the shadow screen (R2, to
          2026-06-24; 1.72 median when missing)
  Europe  yfinance primary-exchange volume (20-day median) x 1.25 for the lit MTFs (close_dislocation: primary
          understates lit volume by ~25%); flagged
  Close baseline: Canada = geometric mean of the per-name MOC volume in ~/index_archive (normal days only); else
          ADV x the market's normal close share (common.COUNTRY; Canada from the archive's median per-name share)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import COUNTRY, MKT, MOC_ARCHIVE, R2DIR, SHADOW_TSX, region_of_country

EVENT_DAYS_2026 = {pd.Timestamp("2026-05-29")}      # MSCI May review close; add others when windows cross them
QUARTER_ENDS = {pd.Timestamp(d) for d in ["2026-09-30", "2026-12-31", "2026-06-30", "2026-03-31"]}
YAHOO_SUFFIX = {"GB": ".L", "DE": ".DE", "FR": ".PA", "NL": ".AS", "BE": ".BR", "PT": ".LS", "IE": ".IR",
                "IT": ".MI", "ES": ".MC", "CH": ".SW", "SE": ".ST", "DK": ".CO", "FI": ".HE", "NO": ".OL",
                "AT": ".VI", "PL": ".WA", "CA": ".TO", "LU": ".AS", "GR": ".AT"}
LIT_UPLIFT_EU = 1.25


def yahoo_symbol(ticker: str, country: str) -> str:
    """iShares/exchange ticker -> Yahoo symbol: LSE 'BA.' -> 'BA.L', 'BT.A' -> 'BT-A.L'; 'AMBU B' -> 'AMBU-B.CO'."""
    t = str(ticker).strip().rstrip(".").replace(" ", "-").replace(".", "-").replace("/", "-")
    return t + YAHOO_SUFFIX.get(country, "")


# -------------------------------------------------------------------------------------------------- FX
def fx_r2(date) -> dict:
    """EUR per unit and CAD per unit of each currency on the last date <= date (Compustat g_exrt_dly, GBP base)."""
    f = pd.read_parquet(R2DIR / "fx_gbp_base_2026.parquet")
    f["datadate"] = pd.to_datetime(f.datadate)
    f = f[(f.fromcurd == "GBP") & (f.datadate <= pd.Timestamp(date))]
    last = f.sort_values("datadate").groupby("tocurd").tail(1).set_index("tocurd").exratd   # X per GBP
    per_gbp = last.to_dict()
    per_gbp["GBP"] = 1.0
    eur = {c: per_gbp["EUR"] / v for c, v in per_gbp.items()}
    cad = {c: per_gbp["CAD"] / v for c, v in per_gbp.items()}
    return dict(eur=eur, cad=cad, date=str(f.datadate.max().date()), source="Compustat g_exrt_dly (R2)")


def fx_live() -> dict:
    """EUR and CAD per unit via yfinance (cached per day in data/market/fx_<date>.csv)."""
    p = MKT / f"fx_{pd.Timestamp.today():%Y%m%d}.csv"
    if not p.exists():
        import yfinance as yf
        ccys = ["CAD", "GBP", "USD", "CHF", "SEK", "DKK", "NOK", "PLN"]
        tick = [f"{c}EUR=X" for c in ccys]
        x = yf.download(tick, period="10d", auto_adjust=False, progress=False)["Close"].ffill().iloc[-1]
        pd.DataFrame({"ccy": ccys, "eur_per": [float(x[f"{c}EUR=X"]) for c in ccys]}).to_csv(p, index=False)
    t = pd.read_csv(p)
    eur = dict(zip(t.ccy, t.eur_per))
    eur["EUR"] = 1.0
    cad = {c: v / eur["CAD"] for c, v in eur.items()}
    return dict(eur=eur, cad=cad, date=p.stem[3:], source="yfinance <CCY>EUR=X")


# ---------------------------------------------------------------------------------------- TSX archive
def tsx_moc_baseline() -> pd.DataFrame:
    """Per-symbol normal-day MOC volume (geometric mean over archived normal days) and MOC share of TSX volume."""
    rows = []
    d = MOC_ARCHIVE / "moc_volume"
    for p in sorted(d.glob("moc-volume_*.csv")) if d.exists() else []:
        day = pd.Timestamp(re.search(r"(\d{8})", p.name).group(1))
        if day in QUARTER_ENDS or day.is_month_end:
            continue
        txt = p.read_text(errors="ignore").splitlines()
        h = next(i for i, l in enumerate(txt) if l.startswith("Exchange,Symbol"))
        x = pd.read_csv(p, skiprows=h)
        x = x[x.Exchange == "TSX"].rename(columns={"Symbol": "symbol", "MOC Volume": "moc"})
        x["date"] = day
        rows.append(x[["symbol", "date", "moc"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "moc_gm", "moc_days", "moc_share", "tsx_cont20"])
    m = pd.concat(rows)
    g = m.groupby("symbol").moc.agg(moc_gm=lambda v: float(np.exp(np.log(np.maximum(v, 100)).mean())),
                                    moc_days="size").reset_index()
    # MOC share of TSX volume and the 20-day continuous average from the intraday stats of the same days
    st = []
    di = MOC_ARCHIVE / "intraday_stats"
    for p in sorted(di.glob("intraday-trading-stats_*_1600.csv")) if di.exists() else []:
        day = pd.Timestamp(re.search(r"_(\d{8})_", p.name).group(1))
        txt = p.read_text(errors="ignore").splitlines()
        h = next(i for i, l in enumerate(txt) if l.startswith("Symbol,"))
        s = pd.read_csv(p, skiprows=h)
        s["date"] = day
        st.append(s.rename(columns={"Symbol": "symbol", "Total Vol": "cont", "20 Day Avg Vol": "cont20"})
                  [["symbol", "date", "cont", "cont20"]])
    if st:
        s = pd.concat(st)
        j = m.merge(s, on=["symbol", "date"])
        j["share"] = j.moc / (j.moc + j.cont)
        sh = j.groupby("symbol").share.median()
        c20 = s.sort_values("date").groupby("symbol").cont20.last()
        g["moc_share"] = g.symbol.map(sh)
        g["tsx_cont20"] = g.symbol.map(c20)
    return g


def canada_close_share_fallback() -> float:
    g = tsx_moc_baseline()
    if "moc_share" in g and g.moc_share.notna().sum() > 50:
        return float(g.moc_share.median())
    return COUNTRY["CA"][1]


# ------------------------------------------------------------------------------------------- history (R2)
def _sister_panel(name: str):
    """Read-only access to a sister study's daily panel: (dates, lines, dict of matrices)."""
    from common import STUDIES
    d = STUDIES / name / "data"
    z = np.load(d / "panel.npz")
    return pd.DatetimeIndex(z["dates"]), pd.read_parquet(d / "panel_lines.parquet"), z


def md_r2(tag: str, T, lines: pd.DataFrame, fx: dict | None = None) -> pd.DataFrame:
    """lines: key, isin, country, ccy, price. Returns the md frame with realised v_T for validation.

    Canada: consolidated volume VC of the S&P/TSX study panel (Datastream exch 295 on the Canadian quote lines,
    split-adjusted; read-only), TSX-only VT if VC is missing; else the Datastream pull.
    Europe: Datastream primary quote + lit MTFs when Datastream's primary quote is on the listing market; otherwise
    (AT&S -> Xetra, SES -> Luxembourg, STM -> Paris for a Milan line, ...) the home line's primary volume from the
    STOXX/DAX study panel (Compustat g_secd) + the same MTF lines."""
    v = pd.read_parquet(R2DIR / f"vol_{tag}.parquet")
    mp = pd.read_parquet(R2DIR / f"vol_{tag}_map.parquet")
    v["marketdate"] = pd.to_datetime(v.marketdate)
    v = v.merge(mp[["isin", "infocode", "prim_country"]], on="infocode")
    T = pd.Timestamp(T)
    fx = fx or fx_r2(T)
    tc, tl, tz = _sister_panel("tsx_composite")
    ec, el, ez = _sister_panel("stoxx600_dax")
    t_ix = {i: k for k, i in enumerate(tl["isin"]) if isinstance(i, str)}
    e_home = el[el.home_line]
    e_ix = {i: k for k, i in zip(e_home.index, e_home["isin"]) if isinstance(i, str)}
    out = []
    ca_share = canada_close_share_fallback()

    def window(dates, vol):
        x = pd.DataFrame({"d": dates, "v": vol}).dropna()
        x = x[~x.d.isin(EVENT_DAYS_2026)]
        pre = x[x.d < T]
        win = pre.iloc[-25:-5] if len(pre) >= 25 else pre.iloc[:-5]
        win = win[win.v > 0]
        vt = x.loc[x.d == T, "v"]
        return (float(win.v.median()) if len(win) >= 8 else np.nan, len(win),
                float(vt.iloc[0]) if len(vt) else np.nan)

    for r in lines.drop_duplicates(["key", "country"]).itertuples():
        reg = region_of_country(r.country)
        adv, n, vt, basis = np.nan, 0, np.nan, "no volume"
        x = v[v["isin"] == r.isin].sort_values("marketdate")
        if reg == "CA" and r.isin in t_ix:
            c = t_ix[r.isin]
            aj = np.where(np.isfinite(tz["AJ"][:, c]) & (tz["AJ"][:, c] > 0), tz["AJ"][:, c], 1.0)
            vc = tz["VC"][:, c] / aj                     # panel volumes are split-adjusted (x ajexdi): back to raw
            if np.isfinite(vc[tc.get_indexer([T])[0]]) if T in tc else False:
                adv, n, vt = window(tc, vc)
                basis = "CA consolidated (Datastream 295 via S&P/TSX study panel)"
            else:
                adv, n, vt = window(tc, tz["VT"][:, c] / aj)
                basis = "TSX only (S&P/TSX study panel; no consolidated line)"
        elif reg == "CA" and len(x):
            vol = x.ca_consol_vol.where(x.ca_consol_vol > 0)
            basis = "CA consolidated (Datastream 295)"
            if vol.notna().sum() < 10:
                vol, basis = x.volume, "primary quote only"
            adv, n, vt = window(x.marketdate.values, vol.values)
        elif reg != "CA":
            mtf = x.set_index("marketdate").mtf_vol if len(x) else pd.Series(dtype=float)
            on_home = len(x) and x.prim_country.iloc[0] == r.country
            if on_home:
                vol = x.volume + x.mtf_vol.fillna(0)
                basis = "primary + lit MTF (Datastream)" if x.mtf_vol.notna().sum() >= 10 else "primary only"
                adv, n, vt = window(x.marketdate.values, vol.values)
            elif r.isin in e_ix:
                c = e_ix[r.isin]
                vp = pd.Series(ez["VP"][:, c] / ez["AJ"][:, c], index=ec)
                lit = vp + mtf.reindex(ec).fillna(0)
                adv, n, vt = window(ec, lit.values)
                basis = "home-line primary (Compustat) + lit MTF (Datastream)"
            elif len(x):
                adv, n, vt = window(x.marketdate.values, (x.volume + x.mtf_vol.fillna(0)).values)
                basis = f"off-market Datastream quote ({x.prim_country.iloc[0]}) + MTF: unreliable"
        share = ca_share if reg == "CA" else COUNTRY.get(r.country, ("", 0.2))[1]
        out.append(dict(key=r.key, country=r.country, ccy=r.ccy, price=r.price, adv_sh=adv, adv_basis=basis,
                        adv_n=n, v_T=vt, close_base_sh=adv * share if np.isfinite(adv) else np.nan,
                        close_basis=f"ADV x normal close share {share:.1%} ({COUNTRY.get(r.country, ('', 0, '', 'assumed'))[3]})",
                        close_n_days=0, eur_per_ccy=fx["eur"].get(r.ccy, np.nan),
                        cad_per_ccy=fx["cad"].get(r.ccy, np.nan)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------- live
def _shadow_screen() -> pd.DataFrame:
    runs = sorted((SHADOW_TSX / "runs").glob("*/*/screen_all.csv"))
    runs = [r for r in runs if "BACKFILL" not in r.parent.name]
    s = pd.read_csv(runs[-1])
    return s


def md_live(lines: pd.DataFrame, fx: dict | None = None) -> pd.DataFrame:
    """lines: key, isin, ticker, country, ccy, price. Canada from tsxintra/MOC archive + shadow; EU from yfinance."""
    fx = fx or fx_live()
    out = []
    lines = lines.drop_duplicates(["key", "country"])
    ca = lines[lines.country == "CA"]
    eu = lines[(lines.country != "CA") & (lines.country != "US")]
    if len(ca):
        moc = tsx_moc_baseline().set_index("symbol")
        scr = _shadow_screen().drop_duplicates("symbol").set_index("symbol")
        hist = None
        hp = sorted((SHADOW_TSX / "cache").glob("yf_hist_*.parquet"))
        if hp:
            hist = pd.read_parquet(hp[-1])
        ca_share = canada_close_share_fallback()
        for r in ca.itertuples():
            sym = r.ticker
            ratio = float(scr.cons_ratio.get(sym, np.nan)) if "cons_ratio" in scr else np.nan
            if not np.isfinite(ratio):
                ratio = 1.72
            tsx_adv, basis = np.nan, ""
            if sym in moc.index and np.isfinite(moc.tsx_cont20.get(sym, np.nan)):
                tsx_adv = float(moc.tsx_cont20[sym]) + float(moc.moc_gm[sym])
                basis = f"tsxintra 20d continuous + MOC, x cons ratio {ratio:.2f}"
            elif hist is not None:
                ysym = sym.replace(".", "-") + ".TO"
                try:
                    vv = hist["Volume"][ysym].dropna().iloc[-20:]
                    tsx_adv = float(vv.median())
                    basis = f"yfinance TSX 20d median (shadow cache), x cons ratio {ratio:.2f}"
                except KeyError:
                    pass
            adv = tsx_adv * ratio if np.isfinite(tsx_adv) else np.nan
            if sym in moc.index:
                cb, cbb, nd = float(moc.moc_gm[sym]), "TSX MOC archive geometric mean", int(moc.moc_days[sym])
            else:
                cb = adv / ratio * ca_share if np.isfinite(adv) else np.nan
                cbb, nd = f"TSX ADV x archive median MOC share {ca_share:.1%}", 0
            out.append(dict(key=r.key, country="CA", ccy="CAD", price=r.price, adv_sh=adv, adv_basis=basis,
                            close_base_sh=cb, close_basis=cbb, close_n_days=nd, eur_per_ccy=fx["eur"]["CAD"],
                            cad_per_ccy=1.0))
    if len(eu):
        ysyms = {r.key: yahoo_symbol(r.ticker, r.country) for r in eu.itertuples()}
        p = MKT / f"yf_eu_{pd.Timestamp.today():%Y%m%d}.parquet"
        have = pd.read_parquet(p) if p.exists() else None
        need = sorted(set(ysyms.values()) - (set(have.columns.get_level_values(1)) if have is not None else set()))
        if need:
            import yfinance as yf
            x = yf.download(need, period="3mo", auto_adjust=False, progress=False, group_by="column")
            if not isinstance(x.columns, pd.MultiIndex):
                x = pd.concat({k: x[[k]].rename(columns={k: need[0]}) for k in ["Close", "Volume"]}, axis=1)
            x = x[["Close", "Volume"]]
            have = x if have is None else pd.concat([have, x], axis=1)
            have = have.loc[:, ~have.columns.duplicated()]
            have.to_parquet(p)
        for r in eu.itertuples():
            ys = ysyms[r.key]
            adv = np.nan
            try:
                vv = have["Volume"][ys].dropna()
                vv = vv[vv > 0].iloc[-25:-5] if len(vv) > 30 else vv[vv > 0]
                adv = float(vv.median()) * LIT_UPLIFT_EU
            except (KeyError, TypeError):
                pass
            share = COUNTRY.get(r.country, ("", 0.2))[1]
            ccy = r.ccy
            out.append(dict(key=r.key, country=r.country, ccy=ccy, price=r.price, adv_sh=adv,
                            adv_basis=f"yfinance {ys} primary 20d median x {LIT_UPLIFT_EU} (lit MTF uplift)",
                            close_base_sh=adv * share if np.isfinite(adv) else np.nan,
                            close_basis=f"ADV x normal close share {share:.1%}", close_n_days=0,
                            eur_per_ccy=fx["eur"].get(ccy, np.nan), cad_per_ccy=fx["cad"].get(ccy, np.nan)))
    return pd.DataFrame(out)
