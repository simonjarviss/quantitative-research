"""Run the 16_flow_engine for the 18 Dec 2026 close (status best) with polite market data (paper only).

The engine is used unchanged: inputs.load / engine.run(close, 'best', md=...) exactly as `engine.py --close
2026-12-18` does. The only difference is where the market data come from. engine.py --md live calls
yf.download() for every European line and for FX in one threaded burst, which breaks this project's
>= 1.5 s-between-requests rule. Here the same quantities are built from our own throttled Yahoo pull
(raw/<fetch date>/yahoo/prices.jsonl) with the engine's own method (marketdata.md_live):
  EU   adv_sh = median Yahoo primary volume over the last 25..5 sessions x 1.25 (LIT_UPLIFT_EU);
       close_base_sh = adv_sh x the market's normal close share (common.COUNTRY); EUR per unit from our FX pull.
  CA   marketdata.md_live on the Canadian lines only (local TSX archive and shadow cache; no web request).
Writes only the engine's sheet outputs: 16_flow_engine/sheets/2026-12-18_best{.csv,_contrib.csv,.md}.
Usage: python3 run_engine.py --fetch-date 2026-10-12
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config as C
import fetch as F
import lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-date", required=True)
    ap.add_argument("--close", default=str(C.TRADE.date()))
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(C.ENGINE))
    import engine as E
    import inputs as I
    import marketdata as M
    from common import COUNTRY
    out = F.rawdir(a.fetch_date)
    P = F.read_prices(out)
    Wc, ccy = lib.price_panel(P, "close")
    Wv = P.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last").sort_index()
    FX = lib.fx_to(P, "EUR")
    last_session = Wc.index[Wc.notna().sum(axis=1) > 200][-1]       # FX at the equity data's last close, not later
    FX = FX[FX.index <= last_session]
    fx_last = FX.ffill().iloc[-1].to_dict()
    eur = {k: float(v) for k, v in fx_last.items() if np.isfinite(v)}
    eur["EUR"] = 1.0
    cad = {k: v / eur["CAD"] for k, v in eur.items()} if "CAD" in eur else {}
    fx = dict(eur=eur, cad=cad, date=str(FX.index.max().date()), source="yfinance <CCY>EUR=X (throttled pull)")
    rows = I.load(a.close)
    rows = E.select_status(rows, "best")
    c, _ = E.contributions(rows, a.close)
    lines = c.drop_duplicates(["key", "country"])[["key", "isin", "ticker", "country", "currency", "price"]]
    lines = lines.rename(columns={"currency": "ccy"})
    md = []
    ca = lines[lines.country == "CA"]
    if len(ca):
        md.append(M.md_live(ca, fx=fx))
    eu = lines[(lines.country != "CA") & (lines.country != "US")]
    missing = []
    for r in eu.itertuples():
        ys = M.yahoo_symbol(r.ticker, r.country)
        adv = np.nan
        if ys in Wv:
            vv = Wv[ys].dropna()
            vv = vv[vv > 0].iloc[-25:-5] if len(vv) > 30 else vv[vv > 0]
            adv = float(vv.median()) * M.LIT_UPLIFT_EU if len(vv) else np.nan
        if not np.isfinite(adv):
            missing.append(ys)
        share = COUNTRY.get(r.country, ("", 0.2))[1]
        md.append(pd.DataFrame([dict(key=r.key, country=r.country, ccy=r.ccy, price=r.price, adv_sh=adv,
                                     adv_basis=f"yfinance {ys} primary 20d median x {M.LIT_UPLIFT_EU} (lit MTF uplift; throttled pull)",
                                     close_base_sh=adv * share if np.isfinite(adv) else np.nan,
                                     close_basis=f"ADV x normal close share {share:.1%}", close_n_days=0,
                                     eur_per_ccy=eur.get(r.ccy, np.nan), cad_per_ccy=cad.get(r.ccy, np.nan))]))
    md = pd.concat(md, ignore_index=True)
    notes = [f"Run via 13_shadow/europe_dec2026/run_engine.py: engine code unchanged; market data from the shadow's "
             f"throttled Yahoo pull raw/{a.fetch_date}/ (engine method: 20d median primary volume x 1.25; FX {fx['date']}). "
             f"EU lines without volume: {len(missing)}."]
    r = E.run(a.close, "best", md=md, md_source="shadow", label="best", extra_notes=notes)
    print("\n".join(r["paths"]))
    s = r["sheet"]
    print(f"net notional EUR {s.notional_eur.sum() / 1e6:,.1f}m; gross {s.gross_notional_eur.sum() / 1e6:,.1f}m; "
          f"expected net {s.exp_notional_eur.sum() / 1e6:,.1f}m; names {len(s)}")
    print(s[["ticker", "name", "side", "net_sh", "notional_eur", "exp_notional_eur", "pct_adv", "close_mult", "min_prob",
             "contributions"]].head(20).to_string(index=False))
    if missing:
        print("no volume for:", missing)


if __name__ == "__main__":
    main()
