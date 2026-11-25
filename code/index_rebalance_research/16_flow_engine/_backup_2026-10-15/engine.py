"""Net-flow engine: per stock, the net passive demand from every index that trades a given close.

    python3 engine.py --close 2026-12-18 [--status best|predicted|announced|final] [--include-us] [--md live|none]

Reads inputs/<close>/*.csv (schema in inputs.py / README.md), pi per index (pi.py: pi_overrides.csv >
../12_studies/net_flow/pi_calibration.csv > flagged fallback), market data (marketdata.py) and the close-pricing table
(../12_studies/close_dislocation/table_pricing.csv, RESULTS.md §9). Writes:
  sheets/<close>_<status>.csv            one row per stock (netting key = ISIN; country:ticker when no ISIN)
  sheets/<close>_<status>_contrib.csv    one row per stock x index (the contributions)
  sheets/<close>_<status>.md             summary: largest net imbalances, offsets, notional per market, pi used

Demand per index j and stock i (flow_sizing.md §3):
  dF_ij   = new_index_shares - old_index_shares            (or (w_new - w_old) * index_mcap / price)
  D_vis   = pi_visible_j * dF_ij                            visible trackers only (lower bound)
  D       = pi_true_j * dF_ij,  pi_true = R_j * pi_visible  central estimate (volume-implied: unseen passive +
                                                            benchmarked active + LP round trips; upper-ish bound)
  E       = prob * D                                        probability-weighted (predicted rows)
Net = sum over indices trading this close; gross = sum of |D|. US index rows are used only with --include-us
(interlisted Canadian names: same ISIN, so US-line demand nets against the TSX line at the same 16:00 ET close).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import inputs as I
from common import INDEX, PHI, PRICING_TABLE, SHEETS, US_INDICES, fmt_bn, region_of_country
from pi import pi_for


# ------------------------------------------------------------------------------------------- selection
def select_status(rows: pd.DataFrame, status: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["group"] = np.where(rows.change_type.isin(I.MEMBERSHIP), "membership", "weights")
    rows["srank"] = rows.status.map(I.STATUS_RANK)
    if status == "best":
        top = rows.groupby(["index", "group"]).srank.transform("max")
        return rows[rows.srank == top].drop(columns=["srank"])
    return rows[rows.status == status].drop(columns=["srank"])


def key_of(r) -> str:
    if isinstance(r.isin, str) and len(r.isin) >= 12:
        return r.isin
    return f"{r.country}:{r.ticker}"


# ---------------------------------------------------------------------------------------------- pricing
def pricing_table() -> pd.DataFrame:
    t = pd.read_csv(PRICING_TABLE)
    return t[(t.period == "2016+") & (t.flow_tercile == "all") & (t.crowding == "all")]


def pricing_row(tab: pd.DataFrame, region: str, country: str, side: int, dom_index: str, dom_type: str,
                price_usd: float) -> dict:
    """Close-pricing row (close_dislocation RESULTS §9) for a net buy (side=+1) or sell (-1)."""
    fam = INDEX.get(dom_index, {}).get("pricing")
    if region == "CA":
        fam = "CA S&P/TSX"
    if fam is None:
        fam = "EU STOXX 600" if region == "EU" else "CA S&P/TSX"
    if fam == "EU DAX family":
        grp = "net buy" if side > 0 else "net sell"
    elif fam == "US Russell":
        grp = ("add" if side > 0 else "delete") + (" | price >= $5" if price_usd >= 5 else " | price < $5")
    else:
        grp = "add" if side > 0 else "delete"
    x = tab[(tab.family == fam) & (tab.group == grp)]
    if x.empty:
        return {}
    x = x.iloc[0]
    exact = dom_type in ("add", "delete", "migration_in", "migration_out") and (
        (fam == "CA S&P/TSX" and dom_index == "SPTSX_COMP") or (fam == "EU STOXX 600" and dom_index == "STOXX600")
        or (fam == "EU DAX family" and dom_index in ("DAX", "MDAX", "SDAX", "TECDAX")) or fam == "US Russell")
    note = "" if exact else f"proxy row: §9 has no row for {dom_index} {dom_type}"
    if country == "GB":
        note = (note + "; " if note else "") + "UK: STOXX UK adds -132 bp / deletes -120 bp on T (RESULTS §1, §7)"
    return dict(pricing_row=f"{fam} / {grp.replace(' | ', ', ')} (2016+)", pricing_n=int(x.n), E_T_dislocation_bp=x.d0_mean_bp,
                E_reversal_T1open_bp=x.rev_on1_mean_bp, sd_T1open_name_bp=x.rev_on1_sd_name_bp,
                quote_hold_T1open_bp=x.quote_name_bp_hold_open, quote_hold_T5_bp=x.quote_name_bp_hold5,
                pricing_note=note)


# ------------------------------------------------------------------------------------------------ core
def contributions(rows: pd.DataFrame, close_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    w_form = rows.old_index_shares.isna() | rows.new_index_shares.isna()
    dF = rows.new_index_shares - rows.old_index_shares
    dF_w = (rows.new_weight - rows.old_weight) * rows.index_mcap / rows.price
    rows["dF"] = np.where(w_form, dF_w, dF)
    rows["form"] = np.where(w_form, "weights", "shares")
    pis = pd.DataFrame([pi_for(j, close_date) for j in sorted(rows["index"].unique())])
    rows = rows.merge(pis, on="index", how="left")
    rows["D_vis"] = rows.pi_visible * rows.dF
    rows["D"] = rows.pi_true * rows.dF
    rows["E"] = rows.D * rows.prob
    rows["key"] = [key_of(r) for r in rows.itertuples()]
    return rows, pis


def aggregate(c: pd.DataFrame, md: pd.DataFrame | None, close_date) -> pd.DataFrame:
    c = c[np.isfinite(c.D)].copy()
    c["region"] = c.country.map(region_of_country)
    out = []
    tab = pricing_table()
    for key, g in c.groupby("key"):
        non_us = g[g.region != "US"]
        lead = (non_us if len(non_us) else g).iloc[0]
        net, gross = g.D.sum(), g.D.abs().sum()
        by_idx = g.groupby("index").D.sum()
        dom = by_idx.abs().idxmax()
        dom_type = g.loc[g["index"] == dom].sort_values("D", key=abs).change_type.iloc[-1]
        contrib = "; ".join(f"{j} {v:+,.0f}" for j, v in by_idx.sort_values(key=abs, ascending=False).items())
        types = "; ".join(f"{j}:{t}" for j, t in g.groupby("index").change_type.agg(
            lambda s: "/".join(sorted(set(s)))).items())
        rec = dict(key=key, isin=lead.isin if isinstance(lead.isin, str) else "", ticker=lead.ticker,
                   name=lead["name"], country=lead.country, region=lead.region, ccy=lead.currency,
                   price=g.price.dropna().iloc[0] if g.price.notna().any() else np.nan,
                   net_sh=net, gross_sh=gross, net_sh_visible=g.D_vis.sum(), exp_net_sh=g.E.sum(),
                   min_prob=g.prob.min(), n_indices=g["index"].nunique(),
                   offset=bool((g.D > 0).any() and (g.D < 0).any()),
                   offset_ratio=abs(net) / gross if gross > 0 else np.nan,
                   ca_sh=g.loc[g.region == "CA", "D"].sum(), us_sh=g.loc[g.region == "US", "D"].sum(),
                   eu_sh=g.loc[g.region == "EU", "D"].sum(),
                   contributions=contrib, change_types=types, dominant_index=dom, dominant_type=dom_type,
                   statuses="/".join(sorted(set(g.status))),
                   pi_fallback=bool(g.fallback.any()), R_assumed=bool(g.R_assumed.any()))
        out.append(rec)
    s = pd.DataFrame(out)
    if s.empty:
        return s
    if md is not None and len(md):
        m = md.copy()
        agg = m.groupby("key").agg(adv_sh=("adv_sh", lambda v: v.sum(min_count=1)),
                                   close_base_sh=("close_base_sh", lambda v: v.sum(min_count=1)),
                                   adv_basis=("adv_basis", "first"), close_basis=("close_basis", "first"),
                                   close_n_days=("close_n_days", "max"),
                                   eur_per_ccy=("eur_per_ccy", "first"), cad_per_ccy=("cad_per_ccy", "first"),
                                   md_price=("price", "first"))
        if "v_T" in m:
            agg["v_T"] = m.groupby("key").v_T.sum(min_count=1)
        s = s.merge(agg.reset_index(), on="key", how="left")
        s["price"] = s.price.fillna(s.md_price)
        s = s.drop(columns=["md_price"])
    else:
        for col in ["adv_sh", "close_base_sh", "eur_per_ccy", "cad_per_ccy", "close_n_days"]:
            s[col] = np.nan
        s["adv_basis"] = s["close_basis"] = "no market data"
    s["notional_local"] = s.net_sh * s.price
    s["gross_notional_local"] = s.gross_sh * s.price
    s["notional_eur"] = s.notional_local * s.eur_per_ccy
    s["notional_cad"] = s.notional_local * s.cad_per_ccy
    s["gross_notional_eur"] = s.gross_notional_local * s.eur_per_ccy
    s["exp_notional_eur"] = s.exp_net_sh * s.price * s.eur_per_ccy
    s["pct_adv"] = s.net_sh.abs() / s.adv_sh
    s["pct_adv_visible"] = s.net_sh_visible.abs() / s.adv_sh
    s["close_mult"] = s.net_sh.abs() / s.close_base_sh
    s["phi"] = s.region.map(PHI)
    s["exp_close_print_mult"] = (s.close_base_sh + s.phi * s.net_sh.abs()) / s.close_base_sh
    pr = []
    for r in s.itertuples():
        side = 1 if r.net_sh > 0 else -1
        pr.append(pricing_row(tab, r.region, r.country, side, r.dominant_index, r.dominant_type,
                              r.price if r.region == "US" else np.nan))
    s = pd.concat([s, pd.DataFrame(pr, index=s.index)], axis=1)
    s["side"] = np.where(s.net_sh > 0, "BUY", np.where(s.net_sh < 0, "SELL", "FLAT"))
    s = s.sort_values("notional_eur", key=lambda v: v.abs(), ascending=False).reset_index(drop=True)
    return s


SHEET_COLS = ["key", "isin", "ticker", "name", "country", "ccy", "side", "price", "net_sh", "gross_sh",
              "net_sh_visible", "exp_net_sh", "min_prob", "notional_local", "notional_eur", "notional_cad",
              "gross_notional_eur", "exp_notional_eur", "adv_sh", "pct_adv", "pct_adv_visible", "close_base_sh",
              "close_mult", "exp_close_print_mult", "n_indices", "offset", "offset_ratio", "contributions",
              "change_types", "ca_sh", "us_sh", "eu_sh", "dominant_index", "statuses", "pi_fallback", "R_assumed",
              "pricing_row", "pricing_n", "E_T_dislocation_bp", "E_reversal_T1open_bp", "sd_T1open_name_bp",
              "quote_hold_T1open_bp", "quote_hold_T5_bp", "pricing_note", "adv_basis", "close_basis",
              "close_n_days", "eur_per_ccy", "cad_per_ccy"]


def run(close_date: str, status: str = "best", include_us: bool = False, md: pd.DataFrame | None = None,
        md_source: str = "none", label: str | None = None, extra_notes: list[str] | None = None,
        write: bool = True) -> dict:
    close_date = str(pd.Timestamp(close_date).date())
    rows = I.load(close_date)
    if rows.empty:
        raise SystemExit(f"no inputs in inputs/{close_date}/")
    if not include_us:
        rows = rows[~rows["index"].isin(US_INDICES)]
    rows = select_status(rows, status)
    c, pis = contributions(rows, close_date)
    dropped = c[~np.isfinite(c.D)].groupby("index").size()
    extra_notes = list(extra_notes or [])
    for j, n in dropped.items():
        extra_notes.append(f"**{j}: {n} rows dropped - no pi for this index on or before {close_date}** "
                           "(add a row to pi_overrides.csv)")
    if md is None and md_source == "live":
        import marketdata as M
        lines = c.drop_duplicates(["key", "country"])[["key", "isin", "ticker", "country", "currency", "price"]]
        md = M.md_live(lines.rename(columns={"currency": "ccy"}))
    s = aggregate(c, md, close_date)
    label = label or status
    res = dict(sheet=s, contrib=c, pis=pis, close_date=close_date, status=label)
    if write:
        base = SHEETS / f"{close_date}_{label}"
        cols = [x for x in SHEET_COLS if x in s] + [x for x in s.columns if x not in SHEET_COLS]
        s[cols].to_csv(f"{base}.csv", index=False, float_format="%.6g")
        cc = c[["key", "isin", "ticker", "name", "country", "index", "change_type", "status", "prob", "dF", "form",
                "pi_visible", "R", "pi_true", "D_vis", "D", "E", "price", "currency", "fallback", "R_assumed",
                "pi_source", "source", "source_ts", "notes"]]
        cc.to_csv(f"{base}_contrib.csv", index=False, float_format="%.6g")
        (SHEETS / f"{close_date}_{label}.md").write_text(summary_md(res, extra_notes))
        res["paths"] = [f"{base}.csv", f"{base}_contrib.csv", f"{base}.md"]
    return res


# --------------------------------------------------------------------------------------------- summary
def _tbl(df: pd.DataFrame, cols: list[str], fmt: dict) -> str:
    if df.empty:
        return "_none_\n"
    h = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        d = r._asdict()
        h += "| " + " | ".join(fmt.get(c, lambda v: str(v))(d[c]) for c in cols) + " |\n"
    return h


def summary_md(res: dict, notes: list[str]) -> str:
    s, c, pis = res["sheet"], res["contrib"], res["pis"]
    f0 = lambda v: f"{v:,.0f}" if np.isfinite(v) else "n/a"   # noqa: E731
    fm = lambda v: fmt_bn(v)                                    # noqa: E731
    fx = lambda v: f"{v:,.1f}x" if np.isfinite(v) else "n/a"    # noqa: E731
    fp = lambda v: f"{v:.0%}" if np.isfinite(v) else "n/a"      # noqa: E731
    L = [f"# Net index flow on the {res['close_date']} close ({res['status']})", "",
         f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by `16_flow_engine/engine.py`. "
         "Demand = pi_true x change in index shares, pi_true = R x visible pi (flow_sizing.md §4.4). "
         "Visible-tracker demand (lower bound) is in the CSV (`net_sh_visible`). Gross of crowding: pre-positioned "
         "inventory is not netted (crowding_and_timing.md).", ""]
    L += ["## Inputs used", ""]
    inp = c.groupby(["index", "status"]).agg(rows=("key", "size"), types=("change_type", lambda v: ", ".join(
        f"{k} {n}" for k, n in v.value_counts().items())), source=("source", lambda v: v.iloc[0][:90])).reset_index()
    L.append(_tbl(inp, ["index", "status", "rows", "types", "source"], {}))
    L += ["## Passive shares used", ""]
    p = pis.copy()
    p["pi_visible"] = p.pi_visible.map(lambda v: f"{v:.3%}" if np.isfinite(v) else "MISSING")
    p["pi_true"] = p.pi_true.map(lambda v: f"{v:.3%}" if np.isfinite(v) else "MISSING")
    p["R"] = p.R.map(lambda v: f"{v:.1f}" if np.isfinite(v) else "none")
    p["pi_date"] = p.pi_date.astype(str).str[:10]
    p["flags"] = [("FALLBACK" if a else "") + (" R assumed" if b else "") for a, b in zip(p.fallback, p.R_assumed)]
    L.append(_tbl(p, ["index", "pi_visible", "R", "pi_true", "pi_date", "flags", "pi_source"], {}))
    if pis.fallback.any():
        L.append("**Fallback pi in use** for the flagged indices: `../12_studies/net_flow/pi_calibration.csv` was "
                 "not found or has no row for them on or before this date.\n")
    L += ["## Notional by market (central estimate)", ""]
    if "notional_eur" in s and s.notional_eur.notna().any():
        mk = s.assign(buy=np.where(s.net_sh > 0, s.notional_eur, 0), sell=np.where(s.net_sh < 0, s.notional_eur, 0))
        t = mk.groupby("country").agg(names=("key", "size"), buy_eur=("buy", "sum"), sell_eur=("sell", "sum"),
                                      net_eur=("notional_eur", "sum"), gross_eur=("gross_notional_eur", "sum"),
                                      exp_net_eur=("exp_notional_eur", "sum")).reset_index()
        t = t.sort_values("gross_eur", ascending=False)
        L.append(_tbl(t, ["country", "names", "buy_eur", "sell_eur", "net_eur", "gross_eur", "exp_net_eur"],
                      {k: fm for k in ["buy_eur", "sell_eur", "net_eur", "gross_eur", "exp_net_eur"]}))
        L.append(f"Total: buy €{fmt_bn(t.buy_eur.sum())}, sell €{fmt_bn(t.sell_eur.sum())}, "
                 f"gross €{fmt_bn(t.gross_eur.sum())} (sum over stocks of the gross index trades).\n")
    else:
        L.append("_no prices/FX: notional not computed_\n")
    L += ["## Largest net imbalances", ""]
    top = s.head(20)
    L.append("`notional_eur` assumes every change happens; `exp_notional_eur` weights predicted rows by `prob`. "
             "`pct_adv` = |net| / consolidated ADV (`pct_adv_visible` in the CSV uses visible-tracker demand).\n")
    L.append(_tbl(top, ["ticker", "name", "country", "side", "net_sh", "notional_eur", "exp_notional_eur", "pct_adv",
                        "close_mult", "min_prob", "contributions"],
                  {"net_sh": f0, "notional_eur": fm, "exp_notional_eur": fm, "pct_adv": fx, "close_mult": fx,
                   "min_prob": fp, "name": lambda v: str(v)[:28]}))
    L += ["## Largest close multiples (|net| / normal closing auction)", ""]
    cm = s[np.isfinite(s.close_mult)].sort_values("close_mult", ascending=False).head(12)
    L.append(_tbl(cm, ["ticker", "name", "side", "net_sh", "close_base_sh", "close_mult", "close_basis",
                       "close_n_days"],
                  {"net_sh": f0, "close_base_sh": f0, "close_mult": fx, "name": lambda v: str(v)[:28],
                   "close_n_days": lambda v: f"{v:.0f}" if v == v else ""}))
    L += ["## Names where indices offset", ""]
    off = s[s.offset].sort_values("gross_notional_eur", key=lambda v: v.abs(), ascending=False).head(20)
    L.append(_tbl(off, ["ticker", "name", "net_sh", "gross_sh", "offset_ratio", "contributions"],
                  {"net_sh": f0, "gross_sh": f0, "offset_ratio": fp, "name": lambda v: str(v)[:28]}))
    L += ["## Close pricing attached (close_dislocation RESULTS §9, per name, 2016+)", "",
          "`E_T_dislocation_bp` = expected T-1 close -> T close move in the direction of demand; "
          "`quote_hold_T1open_bp` = that + 0.25 x SD to the T+1 open. Rows marked proxy borrow the nearest family.", ""]
    pt = s.groupby("pricing_row").agg(names=("key", "size"), E_T=("E_T_dislocation_bp", "first"),
                                      rev=("E_reversal_T1open_bp", "first"),
                                      quote=("quote_hold_T1open_bp", "first")).reset_index()
    L.append(_tbl(pt, ["pricing_row", "names", "E_T", "rev", "quote"],
                  {"E_T": lambda v: f"{v:+.0f}", "rev": lambda v: f"{v:+.0f}", "quote": lambda v: f"{v:+.0f}"}))
    if notes:
        L += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--close", required=True)
    ap.add_argument("--status", default="best", choices=["best", "predicted", "announced", "final"])
    ap.add_argument("--include-us", action="store_true")
    ap.add_argument("--md", default="live", help="live (yfinance/TSX archive) or none")
    a = ap.parse_args()
    r = run(a.close, a.status, a.include_us, md_source=a.md)
    print("\n".join(r.get("paths", [])))
    print(r["sheet"][["ticker", "name", "side", "net_sh", "notional_eur", "pct_adv", "close_mult"]].head(15)
          .to_string(index=False))
