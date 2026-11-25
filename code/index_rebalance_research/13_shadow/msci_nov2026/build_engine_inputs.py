"""Write the net-flow engine inputs for the 30 Nov 2026 MSCI close from a logged shadow run:
    16_flow_engine/inputs/2026-11-30/msci_canada.csv   (index MSCI_CANADA)
    16_flow_engine/inputs/2026-11-30/msci_europe.csv   (index MSCI_EUROPE)

    python3 build_engine_inputs.py [--run predictions/pred_<asof>_<tag>_changes.csv] [--min-prob 0.10] [--dry-run]

Schema: 16_flow_engine/inputs.py (validated with inputs.validate). Status 'predicted', prob = the shadow's p.
Index shares F (float-adjusted shares in the MSCI Standard index):
  Standard members  F = H / k from the iShares UCITS fund (CSCA / IMEA), k = the fund's share of float
  Small Cap / non-members  F = FIF proxy x shares outstanding (R2 / Datastream / Worldscope, flagged in notes)
Migrations: common.INDEX has no MSCI_CANADA_SC / MSCI_EUROPE_SC entry (outside this shadow's write scope), so a
Standard <-> Small Cap migration is ONE row in the Standard file, netted for the Small Cap trackers' opposite trade:
  Std -> Small Cap:  migration_out, F_old = F, F_new = rho x F
  Small Cap -> Std:  migration_in,  F_old = rho x F, F_new = F
with rho = pi_true(<market>_SC) / pi_true(<market>) from 16_flow_engine/pi.py on 2026-11-30 (msci_pi study rows).
Net demand is then (pi_Std - pi_SC) x F, as the engine would compute from two rows; gross is understated.
Direct additions / deletions (into / out of the IMI) use the full F and no rho.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.ENGINE))
import common as EC          # noqa: E402  (16_flow_engine/common.py)
import inputs as EI          # noqa: E402
from pi import pi_for        # noqa: E402

CLOSE = "2026-11-30"
FILES = {"CA": ("MSCI_CANADA", "msci_canada.csv"), "EU": ("MSCI_EUROPE", "msci_europe.csv")}
TYPE = {"MIG_STD_TO_SC": "migration_out", "MIG_SC_TO_STD": "migration_in", "DELETE_FLOAT": "delete",
        "DELETE_LIQ": "delete", "ADD_STD": "add"}
# non-members have no holdings line (no ticker, price or currency): ticker from the Yahoo symbol, currency from the
# listing country, price = the run's own Yahoo close on the as-of date (predict.py yf_close cache)
CCY = {"CA": "CAD", "GB": "GBP", "CH": "CHF", "SE": "SEK", "NO": "NOK", "DK": "DKK", "PL": "PLN"}


def ticker_of(t) -> str | None:
    if isinstance(t.main_ticker, str):
        return t.main_ticker
    return str(t.yf_sym).rsplit(".", 1)[0] if isinstance(t.yf_sym, str) else None


def price_of(t) -> float:
    if np.isfinite(t.price_local) or not isinstance(t.yf_sym, str):
        return t.price_local
    for p in sorted(C.HERE.glob("cache/yf_close_*.parquet"), key=lambda p: p.stem.rsplit("_", 1)[1], reverse=True):
        d = pd.read_parquet(p)
        if t.yf_sym in d.columns:
            s = d[t.yf_sym].dropna()
            s = s[s.index <= pd.Timestamp(t.asof)]
            if len(s):
                return float(s.iloc[-1])
    return np.nan


def rho(mkt: str) -> tuple[float, str]:
    fam = FILES[mkt][0]
    a, b = pi_for(fam, CLOSE), pi_for(fam + "_SC", CLOSE)
    if a.get("fallback") or b.get("fallback") or not np.isfinite(a["pi_true"]) or not np.isfinite(b["pi_true"]) \
            or a["pi_true"] <= 0:
        return 0.0, f"rho=0 FALLBACK (no pi_calibration row for {fam}_SC on/before {CLOSE})"
    r = float(b["pi_true"] / a["pi_true"])
    return r, (f"rho={r:.3f} = pi_true {fam}_SC {b['pi_true']:.4f} / {fam} {a['pi_true']:.4f} "
               f"(pi_calibration {a['pi_date'].date()})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="changes file of a logged run (default: the latest predictions/pred_*_changes.csv)")
    ap.add_argument("--min-prob", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true", help="validate and print, do not write")
    args = ap.parse_args()
    f = Path(args.run) if args.run else max(C.PRED.glob("pred_*_changes.csv"), key=lambda p: (p.name[5:15], p.stat().st_mtime))
    ch = pd.read_csv(f)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    ts = pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC").isoformat(timespec="seconds")
    out = {}
    for mkt, (index, fname) in FILES.items():
        r, rnote = rho(mkt)
        x = ch[(ch.market == mkt) & (ch.p_change >= args.min_prob) & ch.category.isin(TYPE)].copy()
        rows = []
        for t in x.itertuples():
            F = float(t.index_shares_std)
            ct = TYPE[t.category]
            if ct == "migration_out":
                f0, f1, net = F, F * r, f"net of Small Cap trackers' opposite trade, {rnote}; gross understated"
            elif ct == "migration_in":
                f0, f1, net = F * r, F, f"net of Small Cap trackers' opposite trade, {rnote}; gross understated"
            elif ct == "delete":
                f0, f1, net = F, 0.0, "leaves the IMI (float/liquidity test): full F"
            else:
                f0, f1, net = 0.0, F, "enters from outside the IMI: full F"
            fsrc = "F = iShares H/k" if t.seg == "STD" else f"F = FIF proxy ({t.fif_src}) x shares"
            country = EC.EXCH_COUNTRY.get(t.exchange, t.country if isinstance(t.country, str) else None)
            if mkt == "CA":
                country = "CA"
            ccy = {"GBX": "GBP", "GBp": "GBP"}.get(t.mkt_ccy, t.mkt_ccy) if isinstance(t.mkt_ccy, str) \
                else CCY.get(country, "EUR")
            rows.append(dict(close_date=CLOSE, index=index, change_type=ct, isin=t.main_isin, ticker=ticker_of(t),
                             country=country, name=t.name, currency=ccy,
                             old_index_shares=f0, new_index_shares=f1, price=price_of(t), prob=round(float(t.p_change), 4),
                             status="predicted",
                             source=f"13_shadow/msci_nov2026/{f.relative_to(C.HERE) if f.is_relative_to(C.HERE) else f.name} (sha256 {sha[:16]})",
                             source_ts=ts,
                             notes=f"{t.category}; {net}; {fsrc}; {t.why}"[:900]))
        df = EI.validate(pd.DataFrame(rows), f"{index} from {f.name}")
        out[mkt] = df
        print(f"{index}: {len(df)} rows (p >= {args.min_prob}); {rnote}")
        if not args.dry_run:
            d = EC.INPUTS / CLOSE
            d.mkdir(parents=True, exist_ok=True)
            p = d / fname
            df.to_csv(p, index=False)
            print(f"  wrote {p} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    return out


if __name__ == "__main__":
    main()
