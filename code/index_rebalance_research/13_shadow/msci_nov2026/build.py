"""Company tables for the MSCI shadow: one row per company in the Canada market and in the DM Europe market, with
segment (STD / SC / NON = not in the IMI), full company market cap in USD, float cap, FIF proxy, liquidity and
volatility, as of a holdings date A.

Sources (all read-only):
  iShares UCITS holdings (CH JSON, fund-currency prices): CSCA (MSCI Canada), IMEA (MSCI Europe), WSML (MSCI World
      Small Cap; samples ~90% of the index, so the smallest Small Cap names can be missing), SWDA (MSCI World;
      used for the USD/EUR rate and to spot companies MSCI classifies in another country)
  R2 panels (12_studies/tsx_composite, 12_studies/stoxx600_dax; to 2026-06-26 / 2026-06-30): prices, listed shares,
      consolidated (CA) / primary (EU) volume, Datastream/Worldscope float proxies
  R2 Datastream ds2mktval: company-level shares across ALL classes, incl. unlisted ones (r2data.py)
  Live Canada: the S&P/TSX shadow's 9 Oct 2026 screen (MiG 31-Aug shares, Yahoo closes) and its Yahoo cache
  Live Europe, non-members: Yahoo (yf.py), only for the few large non-IMI candidates

Full company cap (gimi §3.1 "Full company market capitalization", p.50): sum over listed classes of shares x price
plus unlisted classes x the reference line's price (conversion factor 1 assumed, I). Shares are R2's (June 2026) or
MiG's (31 Aug 2026), not MSCI's NOS as of the price cutoff date (data error, see config.SIGMA_DATA).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config as C
import fetch as F
import r2data

EU_LOC = {"United Kingdom": "GB", "France": "FR", "Germany": "DE", "Switzerland": "CH", "Netherlands": "NL",
          "Sweden": "SE", "Italy": "IT", "Spain": "ES", "Denmark": "DK", "Belgium": "BE", "Finland": "FI",
          "Norway": "NO", "Austria": "AT", "Ireland": "IE", "Portugal": "PT"}
ISO3 = {"GBR": "GB", "FRA": "FR", "DEU": "DE", "CHE": "CH", "NLD": "NL", "SWE": "SE", "ITA": "IT", "ESP": "ES",
        "DNK": "DK", "BEL": "BE", "FIN": "FI", "NOR": "NO", "AUT": "AT", "IRL": "IE", "PRT": "PT", "LUX": "LU",
        "JEY": "GB", "GGY": "GB", "IMN": "GB", "BMU": "BM", "CAN": "CA", "USA": "US"}


# ---------------------------------------------------------------------------------------------- panels
@lru_cache(maxsize=2)
def panel(mkt: str) -> dict:
    d = C.CA_PANEL if mkt == "CA" else C.EU_PANEL
    z = np.load(d / "panel.npz")
    L = pd.read_parquet(d / "panel_lines.parquet")
    out = dict(L=L, D=pd.DatetimeIndex(z["dates"]), P=z["P"], SH=z["SH"], AJ=z["AJ"], RET=z["RET"])
    out["V"] = z["VC"] if mkt == "CA" else z["VP"]
    if mkt == "CA":
        out["VT"] = z["VT"]
        vc = z["VC"].copy()
        miss = ~np.isfinite(vc)
        vc[miss] = z["VT"][miss]                       # TSX-only volume where the consolidated line is missing
        out["V"] = vc
    out["FX"] = z["FX"] if mkt == "EU" else None
    fm = pd.read_parquet(d / "float_monthly.parquet")
    fm["month"] = pd.to_datetime(fm.month)
    out["FM"] = fm
    return out


def pidx(D: pd.DatetimeIndex, date) -> int:
    return int(D.searchsorted(pd.Timestamp(date), side="right") - 1)


def isin_cols(L: pd.DataFrame) -> dict:
    """ISIN -> panel column (home / main-exchange line preferred)."""
    x = L.reset_index().rename(columns={"index": "col"})
    x = x[x["isin"].notna()]
    if "main_exch" in x:
        x = x.sort_values(["main_exch", "n_px"], ascending=False)
    return dict(zip(x["isin"], x["col"]))


def float_proxy(mkt: str, cols, date) -> pd.Series:
    """Latest Datastream (else Worldscope) free-float fraction per panel column at or before date."""
    fm = panel(mkt)["FM"]
    f = fm[(fm.month <= pd.Timestamp(date)) & fm.line.isin(list(cols))].sort_values("month").groupby("line").last()
    v = f.ff_ds.where(f.ff_ds.notna() & (f.ff_ds < 1.0), f.ff_ws)
    v = v.fillna(f.ff_ds)
    return v.clip(0.0, 1.0)


def liquidity(mkt: str, col: int, end, ff: float) -> dict:
    """ATVR (12m, 3m) and 3m frequency of trading from R2 daily volume up to `end` (MSCI-LIQ §1.1 as summarised in
    gimi §2.2.5): monthly median traded value (traded days) x days traded / month-end float cap, x 12, averaged."""
    p = panel(mkt)
    D, P, V, SH, AJ = p["D"], p["P"], p["V"], p["SH"], p["AJ"]
    end = pd.Timestamp(end)
    i1 = pidx(D, end)
    i0 = pidx(D, end - pd.DateOffset(months=12))
    if i1 < 30 or col is None or col < 0:
        return dict(atvr12=np.nan, atvr3=np.nan, fot3=np.nan)
    dd = D[i0 + 1:i1 + 1]
    px, vol = P[i0 + 1:i1 + 1, col], V[i0 + 1:i1 + 1, col] / AJ[i0 + 1:i1 + 1, col]
    sh = SH[i0 + 1:i1 + 1, col] / AJ[i0 + 1:i1 + 1, col]
    df = pd.DataFrame(dict(m=dd.to_period("M"), tv=vol * px, cap=px * sh * max(ff, 0.01)))
    rows = []
    if not np.isfinite(vol).any():
        return dict(atvr12=np.nan, atvr3=np.nan, fot3=np.nan)
    for m, g in df.groupby("m"):
        if not np.isfinite(g.tv).any():
            continue
        traded = g.tv > 0
        mmtv = np.nanmedian(g.tv[traded]) * traded.sum() if traded.any() else 0.0
        cap = g.cap.dropna().iloc[-1] if g.cap.notna().any() else np.nan
        rows.append(dict(m=m, r=mmtv / cap if cap and cap > 0 else np.nan, fot=traded.mean()))
    r = pd.DataFrame(rows)
    if len(r) < 3:
        return dict(atvr12=np.nan, atvr3=np.nan, fot3=np.nan)
    return dict(atvr12=float(r.r.mean() * 12), atvr3=float(r.r.tail(3).mean() * 12), fot3=float(r.fot.tail(3).mean()))


def rel_vol(mkt: str, col: int, end, mkt_ret: np.ndarray, n: int = C.REL_VOL_WINDOW) -> float:
    p = panel(mkt)
    i1 = pidx(p["D"], end)
    r = p["RET"][i1 - n + 1:i1 + 1, col]
    x = r - mkt_ret[i1 - n + 1:i1 + 1]
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) >= 20 else np.nan


def market_ret(mkt: str, cols: list[int]) -> np.ndarray:
    """Cross-sectional median daily return of the market's Standard members (panel columns) = market factor."""
    R = panel(mkt)["RET"][:, cols]
    return np.nanmedian(R, axis=1)


# ---------------------------------------------------------------------------------------------- holdings
def k_of(h: pd.Series, so: pd.Series, q: float = 0.85) -> tuple[float, int]:
    """Fund's share of each member's float, k = H_i / F_i. For FIF = 1 names H/SO = k, so k is a high quantile of
    H/SO (net_flow s03 uses the 90th percentile; 85th here because MSCI caps FIF at 1 and share counts lag)."""
    r = (h / so).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[(r > 0)]
    return float(r.quantile(q)), int(len(r))


def holdings(path, fund) -> pd.DataFrame:
    h = F.parse_ch(path, fund)
    return h


# ---------------------------------------------------------------------------------------------- R2 company shares
def unlisted_extra(isins: list[str], ref_date, tag: str) -> pd.DataFrame:
    """Datastream company-level consolidated shares (all classes) per ISIN, nearest valdate <= ref_date."""
    n = r2data.ds_names(isins, tag)
    n = n.sort_values(["isprimqt"], ascending=False).drop_duplicates("isin")
    mv = r2data.ds_mktval(n.ic.tolist(), tag)
    mv = mv[mv.valdate <= pd.Timestamp(ref_date)].sort_values("valdate").groupby("ic").last().reset_index()
    out = n[["isin", "ic", "cmp", "dscmpyname", "qccy"]].merge(mv, on="ic", how="left")
    out["cns_sh"] = out.cns * 1000.0
    return out


# ---------------------------------------------------------------------------------------------- line tables
def ca_lines_live(date=None, yf_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Canadian equity lines (TSX) from the S&P/TSX shadow's logged 9 Oct 2026 screen (read-only): MiG 31-Aug
    shares, Yahoo close, S&P float proxy, listing date. Only lines above C$0.5bn are in that screen. Prices are
    taken on `date` from a Yahoo close table (columns = Yahoo symbols) when given: the TSX shadow's cache
    (to 9 Oct 2026) plus this shadow's own refresh."""
    S = pd.read_csv(C.TSX_SHADOW / "runs/2026-12/2026-10-09_H-10w/screen_all.csv", keep_default_na=False,
                    na_values={c: [""] for c in ("so", "close_asof", "float_sh", "cons_ratio", "listing_date")})
    S["symbol"], S["root"] = S.symbol.astype(str), S.root.replace("", np.nan)
    for c in ("so", "close_asof", "float_sh", "cons_ratio"):
        S[c] = pd.to_numeric(S[c], errors="coerce")
    if date is not None and yf_close is not None:
        cl = yf_close[yf_close.index <= pd.Timestamp(date)].ffill()
        if len(cl):
            last = cl.iloc[-1]
            S["close_asof"] = S.yf.map(last).where(S.yf.map(last).notna(), S.close_asof)
    L = panel("CA")["L"]
    isin = dict(zip(L.tic.astype(str).str.rstrip("."), L["isin"]))
    x = pd.DataFrame(dict(symbol=S.symbol, company=S.root.fillna(S.symbol), name=S.issuer, so=S.so,
                          px=S.close_asof, ff=(S.float_sh / S.so).clip(upper=1.0), listing_date=pd.to_datetime(S.listing_date),
                          yf=S.yf, cons_ratio=S.cons_ratio, flags=S["flags"].fillna("")))
    x["isin"] = x.symbol.map(isin)
    x["ccy"] = "CAD"
    return x


def ca_lines_r2(date) -> pd.DataFrame:
    """Canadian lines from the R2 panel on `date` (point in time; for the replay)."""
    p = panel("CA")
    L, D = p["L"], p["D"]
    i = pidx(D, date)
    px, sh = p["P"][i], p["SH"][i] / p["AJ"][i]
    ok = np.isfinite(px) & np.isfinite(sh) & (sh > 0) & (L.exchg.astype(str) == "7").values
    x = pd.DataFrame(dict(col=np.where(ok)[0]))
    x["symbol"] = L.tic.astype(str).str.rstrip(".").values[ok]
    x["company"] = L.gvkey.values[ok]
    x["name"] = L.conm.values[ok]
    x["isin"] = L["isin"].values[ok]
    x["so"], x["px"] = sh[ok], px[ok]
    ff = float_proxy("CA", x.col, date)
    x["ff"] = x.col.map(ff)
    x["listing_date"] = pd.to_datetime(L.ipodate.values[ok], errors="coerce")
    x["ccy"] = "CAD"
    x["flags"] = ""
    return x


def eu_lines_r2(date) -> pd.DataFrame:
    """European home-exchange lines from the R2 panel on `date` (local prices, EUR FX)."""
    p = panel("EU")
    L, D = p["L"], p["D"]
    i = pidx(D, date)
    px, sh, fx = p["P"][i], p["SH"][i] / p["AJ"][i], p["FX"][i]
    ok = np.isfinite(px) & np.isfinite(sh) & (sh > 0) & L.main_exch.fillna(False).values.astype(bool) & L["isin"].notna().values
    x = pd.DataFrame(dict(col=np.where(ok)[0]))
    for c in ("gvkey", "conm", "isin", "fic", "excntry", "tpci", "first_px", "ipodate", "gsector", "home_line"):
        x[c] = L[c].values[ok]
    x = x.rename(columns={"gvkey": "company", "conm": "name"})
    x["so"], x["px"], x["eur_per"] = sh[ok], px[ok], fx[ok]
    ff = float_proxy("EU", x.col, date)
    x["ff"] = x.col.map(ff)
    x["listing_date"] = pd.to_datetime(x.ipodate, errors="coerce").fillna(pd.to_datetime(x.first_px, errors="coerce"))
    x["flags"] = ""
    # one line per ISIN (cross-listings of the same security on several home exchanges)
    x = x.sort_values("home_line", ascending=False).drop_duplicates("isin")
    return x


# ---------------------------------------------------------------------------------------------- company table
def company_table(mkt: str, std_h: pd.DataFrame, sc_h: pd.DataFrame, lines: pd.DataFrame, usd_per_fund: dict,
                  usd_per_local, ds_ref, tag: str, other_country_isins: set) -> pd.DataFrame:
    """One row per company of market `mkt` ('CA' or 'EU').

    std_h / sc_h: holdings of the Standard fund (CSCA / IMEA) and the Small Cap fund lines of this market (WSML),
      prices in fund currency; usd_per_fund: {'CSCA': 1.0, 'IMEA': usd_per_eur, 'WSML': 1.0}.
    lines: line table (isin, symbol, company, name, so, px local at the reference date, ff, listing_date, ...).
    usd_per_local: function(line_row) -> USD per unit of local price at the reference date (for non-held lines).
    ds_ref: date for Datastream company shares (unlisted classes). other_country_isins: ISINs MSCI holds as another
      country (SWDA / WSML location outside this market) -> companies classified elsewhere.
    """
    lines = lines.copy()
    lines["isin"] = lines["isin"].astype("string")
    # ---- holdings -> lines (ISIN; Canada also by ticker)
    H = []
    for seg, h in (("STD", std_h), ("SC", sc_h)):
        if h is None or h.empty:
            continue
        x = h.copy()
        x["seg_line"] = seg
        x["usd"] = x.fund.map(usd_per_fund)
        x["px_usd_h"] = x.price * x.usd
        x["mv_usd"] = x.mv * x.usd
        H.append(x[["isin", "ticker", "name", "shares", "px_usd_h", "mv_usd", "seg_line", "fund", "sector", "exchange",
                    "mkt_ccy", "location"]])
    H = pd.concat(H, ignore_index=True)
    H = H.sort_values("seg_line", key=lambda s: s.map({"STD": 0, "SC": 1})).drop_duplicates("isin")
    li = lines.drop_duplicates("isin").set_index("isin")
    H["in_lines"] = H["isin"].isin(li.index)
    if mkt == "CA":
        sym = lines.drop_duplicates("symbol").set_index("symbol")
        miss = ~H.in_lines & H.ticker.isin(sym.index)
        H.loc[miss, "isin_line"] = H.loc[miss, "ticker"].map(sym["isin"])
        H.loc[miss & H.isin_line.notna(), "in_lines"] = True
    H["isin_line"] = H.get("isin_line", pd.Series(index=H.index, dtype="object")).fillna(H["isin"])
    # ---- merge
    lines = lines.merge(H.rename(columns={"isin": "isin_h", "name": "name_h"}), left_on="isin", right_on="isin_line",
                        how="outer", indicator=True)
    lines["isin"] = lines["isin"].fillna(lines.isin_h)
    lines["name"] = lines["name"].fillna(lines.name_h)
    unm = lines._merge == "right_only"
    lines.loc[unm, "company"] = "H:" + lines.loc[unm, "isin"].astype(str)
    lines.loc[unm, "flags"] = "unmapped_holding"
    lines["seg_line"] = lines.seg_line.fillna("NON")
    # ---- equivalent company shares (in units of each line), incl. unlisted classes from Datastream
    ex = unlisted_extra(lines["isin"].dropna().tolist(), ds_ref, tag)
    cns = dict(zip(ex["isin"], ex.cns_sh))
    lines["cns_sh"] = lines["isin"].map(cns)
    lines["usd_loc"] = [usd_per_local(r) for r in lines.itertuples()]
    lines["cap_loc_usd"] = lines.so * lines.px * lines.usd_loc                    # listed line, reference date
    # price in USD at A: holdings price when held, else the line price at the reference date
    lines["px_usd"] = lines.px_usd_h.where(lines.px_usd_h.notna(), lines.px * lines.usd_loc)
    lines["ret_ref_to_A"] = (lines.px_usd / (lines.px * lines.usd_loc)).where(lines.px.notna(), 1.0)
    out = []
    for comp, g in lines.groupby("company", dropna=False):
        g = g.copy()
        seg = "STD" if (g.seg_line == "STD").any() else ("SC" if (g.seg_line == "SC").any() else "NON")
        main = g.sort_values(["seg_line", "cap_loc_usd"], key=lambda s: s.map({"STD": 0, "SC": 1, "NON": 2})
                             if s.name == "seg_line" else -s.fillna(0)).iloc[0]
        listed_usd_ref = g.cap_loc_usd.sum(min_count=1)
        flags = set(f for f in ";".join(g["flags"].fillna("").astype(str)).split(";") if f)
        extra_usd_ref = 0.0
        if np.isfinite(main.cns_sh) and np.isfinite(main.so) and main.so > 0:
            listed_sh_main_units = listed_usd_ref / (main.px * main.usd_loc) if np.isfinite(listed_usd_ref) else np.nan
            extra = main.cns_sh - listed_sh_main_units
            if np.isfinite(extra) and extra > 0.03 * listed_sh_main_units:
                extra_usd_ref = extra * main.px * main.usd_loc
                flags.add(f"unlisted_or_other_classes+{extra / listed_sh_main_units:.0%}")
        if not np.isfinite(listed_usd_ref):           # unmapped holding: Datastream shares x holdings price
            full_A = main.cns_sh * main.px_usd if np.isfinite(main.cns_sh) else np.nan
            flags.add("full_cap_from_DS_shares" if np.isfinite(full_A) else "no_full_cap")
        else:
            # roll each line from the reference date to A by its own price move (held lines) or the main line's
            r_main = main.ret_ref_to_A if np.isfinite(main.ret_ref_to_A) else 1.0
            rr = g.ret_ref_to_A.where(g.px_usd_h.notna(), r_main).fillna(r_main)
            full_A = float((g.cap_loc_usd.fillna(0) * rr).sum() + extra_usd_ref * r_main)
        # float cap of the main security (index line): holdings-implied for members, proxy otherwise
        out.append(dict(company=comp, name=str(main["name"]), seg=seg, n_lines=len(g),
                        isins=";".join(sorted(g["isin"].dropna().astype(str))),
                        tickers=";".join(sorted(set(g.ticker.dropna().astype(str)) | set(g.get("symbol", pd.Series(dtype=str)).dropna().astype(str)))),
                        main_isin=main["isin"], main_ticker=main.ticker if isinstance(main.ticker, str) else main.get("symbol"),
                        main_so=main.so, main_px_usd=main.px_usd, main_h=main.shares, main_mv_usd=main.mv_usd,
                        main_ff_proxy=main.ff, main_fund=main.fund, col=main.get("col", np.nan),
                        full_cap_usd=full_A, listing_date=g.listing_date.min(),
                        other_country=bool(g["isin"].isin(other_country_isins).any()),
                        flags=";".join(sorted(flags)), sector=main.sector,
                        exchange=main.exchange if isinstance(main.exchange, str) else
                        ("Toronto Stock Exchange" if mkt == "CA" else None),
                        mkt_ccy=main.mkt_ccy if isinstance(main.mkt_ccy, str) else main.get("ccy"),
                        location=main.location if isinstance(main.location, str) else None,
                        **{k: main.get(k) for k in ("fic", "excntry", "tpci", "gsector", "yf", "cons_ratio", "symbol")
                           if k in g.columns}))
    T = pd.DataFrame(out)
    return T, lines


# ---------------------------------------------------------------------------------------------- enrichment
def enrich(T: pd.DataFrame, mkt: str, std_h: pd.DataFrame, sc_h: pd.DataFrame, lines: pd.DataFrame,
           liq_end, vol_end, seasoning_cut) -> tuple[pd.DataFrame, dict]:
    """FIF and float cap, ATVR / FoT, relative volatility and seasoning per company (main security).

    Standard members: FIF = (H / k) / shares, k = the Standard fund's share of float (k_of on its holdings).
    Small Cap and non-members: Datastream / Worldscope free-float proxy (R2 float_monthly), else the WSML-implied
    FIF (WSML samples, so its implied FIF is noisy), else 1 (flagged)."""
    T = T.copy()
    so_by_isin = lines.dropna(subset=["isin"]).drop_duplicates("isin").set_index("isin").so
    hs = std_h.assign(so=std_h["isin"].map(so_by_isin))
    k_std, n_std = k_of(hs.shares, hs.so)
    ws = sc_h.assign(so=sc_h["isin"].map(so_by_isin)) if sc_h is not None and len(sc_h) else None
    k_sc, n_sc = k_of(ws.shares, ws.so) if ws is not None else (np.nan, 0)
    sec_cap = T.main_so * T.main_px_usd
    fif_h = np.where(T.seg == "STD", (T.main_h / k_std) / T.main_so,
                     np.where(T.seg == "SC", (T.main_h / k_sc) / T.main_so, np.nan))
    fif_h = pd.Series(fif_h, index=T.index).clip(upper=1.0)
    proxy = T.main_ff_proxy
    T["fif_src"] = np.where(T.seg == "STD", "fund-implied",
                            np.where(proxy.notna(), "R2/screen proxy", np.where(np.isfinite(fif_h), "WSML-implied", "none=1")))
    T["fif"] = np.where(T.seg == "STD", fif_h, proxy.where(proxy.notna(), fif_h)).astype(float)
    T["fif"] = T.fif.where(np.isfinite(T.fif), 1.0).clip(0.0, 1.0)
    T["float_usd"] = np.where(T.seg == "STD", T.main_mv_usd / k_std, T.fif * sec_cap)
    T["sec_cap_usd"] = sec_cap
    # panel column of the main line (liquidity, volatility)
    cols = isin_cols(panel(mkt)["L"])
    T["col"] = T.main_isin.map(cols)
    std_cols = [int(c) for c in T.loc[T.seg == "STD", "col"].dropna()]
    mret = market_ret(mkt, std_cols)
    liq, vol = [], []
    for r in T.itertuples():
        if pd.notna(r.col):
            liq.append(liquidity(mkt, int(r.col), liq_end, r.fif if np.isfinite(r.fif) else 1.0))
            vol.append(rel_vol(mkt, int(r.col), vol_end, mret))
        else:
            liq.append(dict(atvr12=np.nan, atvr3=np.nan, fot3=np.nan))
            vol.append(np.nan)
    T = pd.concat([T, pd.DataFrame(liq, index=T.index)], axis=1)
    T["sig_rel"] = vol
    T["seasoned"] = ~(pd.to_datetime(T.listing_date, errors="coerce") > pd.Timestamp(seasoning_cut))
    meta = dict(k_std=k_std, n_k_std=n_std, k_sc=k_sc, n_k_sc=n_sc, liq_end=str(pd.Timestamp(liq_end).date()),
                vol_end=str(pd.Timestamp(vol_end).date()))
    return T, meta
