"""First live sheet for the Fri 18 Dec 2026 close, from what exists today; re-run whenever an input arrives.

    python3 build_live_2026_12_18.py            # refresh predicted inputs + run the engine (status best)

Predicted inputs written here (status 'predicted'; announced/final rows dropped in later win automatically):
  SPTSX_COMP  adds/deletes from the latest logged TSX shadow run (13_shadow/tsx_dec2026/runs/2026-12/...)
              + quarterly share updates from TMX MiG O/S changes once a second MiG month is held (live.py)
  capping     every capped index with a free holdings file, from today's weights: S&P/TSX Capped Composite 10%
              (XIC), DAX / MDAX / TecDAX 15% (EXS1, EXS3, EXS2). Only indices with a name above the cap get rows.
Not predictable today (TODO.md lists the drop-in dates): STOXX 600, FTSE 100/250, DAX-family membership, S&P/TSX 60,
FTSE All-World, US indices for interlisted names (Nasdaq-100 annual recon is announced 11 Dec).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import engine as E
import inputs as I
import live as LV
from common import INDEX, INPUTS, SHADOW_TSX

CLOSE = "2026-12-18"


def main():
    notes = []
    # 1. S&P/TSX shadow prediction (latest non-backfill run for the 2026-12 review)
    runs = sorted((SHADOW_TSX / "runs" / "2026-12").glob("*/predictions.csv"))
    run_dir = runs[-1].parent
    tsx = I.from_tsx_shadow(run_dir, CLOSE)
    su, su_note = LV.tsx_share_updates()
    rows = pd.concat([tsx, su], ignore_index=True) if len(su) else tsx
    I.write(rows, CLOSE, "SPTSX_COMP")
    notes.append(f"SPTSX_COMP: {len(tsx)} predicted adds/deletes from shadow run {run_dir.name}; share updates: {su_note}")
    # 2. capping from today's weights
    for idx in ["SPTSX_COMP", "DAX", "MDAX", "TECDAX"]:
        cap = INDEX[idx]["cap"]
        h, M, src = LV.members_today(idx)
        c = I.capping_rows(h, idx, cap, M, source=f"iShares {INDEX[idx]['etf']} weights {h["asof"].iloc[0].date()}; "
                                                   f"index cap {M / 1e9:,.0f}bn ({src})")
        mx = h.weight.max()
        if len(c):
            if idx == "SPTSX_COMP":
                I.write(pd.concat([rows, c], ignore_index=True), CLOSE, idx)
            else:
                I.write(c, CLOSE, idx)
            notes.append(f"{idx}: capping at {cap:.0%} binds (max weight {mx:.2%}): {len(c)} rows incl. funding buys")
        else:
            notes.append(f"{idx}: no capping flow (max weight {mx:.2%} < cap {cap:.0%})")
    notes.append("No US-index rows (optional input; --include-us once NDX/S&P lists for interlisted names are in).")
    r = E.run(CLOSE, "best", md_source="live", label="predicted", extra_notes=notes)
    print("\n".join(notes))
    print("\n".join(r["paths"]))
    print(r["sheet"][["ticker", "name", "side", "net_sh", "notional_eur", "pct_adv", "close_mult", "min_prob",
                      "contributions"]].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
