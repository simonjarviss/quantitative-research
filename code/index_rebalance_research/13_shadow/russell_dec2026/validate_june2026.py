"""Calibration: replay this free-data pipeline on the June 2026 recon and score it against FTSE's final lists.

Pre-June membership is reconstructed as today's iShares members - June 2026 R3000 adds + June 2026 deletes, then
the ranking is rebuilt on the 30 Apr 2026 rank-day close with the shares fetched for the live run.
Biases (all stated in the output):
  - look-ahead in shares (today's counts) and in the universe (names delisted since June are missing, which
    mostly hurts deletion recall);
  - NO known-excluded refinement (it would need the Dec 2025 / June 2025 lists), so add precision is a lower
    bound for the live run, which does use it;
  - pre-June R1000/R2000 split is today's split (migrations are not scored).
Usage: python3 validate_june2026.py --fetch-date 2026-10-09
"""
from __future__ import annotations
import argparse
import types

import numpy as np
import pandas as pd

import config as C
import predict as P
import score as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-date", required=True)
    a = ap.parse_args()
    args = types.SimpleNamespace(fetch_date=a.fetch_date, no_fetch=True, info_from=None, workers=1, info_minutes=0,
                                 asof=None, tag="")
    B = P.build(args)
    Co = B["Co"].copy()
    LA = S.match_to_companies(S.read_list(C.RAW / "ftse_2026-06-26" / "ru3000-additions-20260626.pdf"), Co)
    LD = S.match_to_companies(S.read_list(C.RAW / "ftse_2026-06-26" / "ru3000-deletions-20260626.pdf"), Co)
    actA, actD = set(LA.cid.dropna()), set(LD.cid.dropna())
    cur = Co.cur.copy()
    cur[Co.cid.isin(actA)] = "out"
    cur[Co.cid.isin(actD) & (cur == "out")] = "R2000"
    Co["cur"], Co["member"] = cur, cur != "out"
    Co["known_excl"] = False
    reason = P.screens(Co, "jun")
    E, br = P.rank_and_band(Co[reason == ""].copy(), "cap_jun")
    predA = set(E.loc[(E.cur == "out") & (E.pred != "out"), "cid"])
    predD = set(E.loc[(E.cur != "out") & (E.pred == "out"), "cid"]) | set(Co.loc[Co.member & (reason != "") &
                                                                              (reason != "no_price_or_shares"), "cid"])
    rows = []
    for g, Pset, Aset in [("R3000 adds", predA, actA), ("R3000 deletes", predD, actD)]:
        rows.append(dict(group=g, **S.prf(Pset, Aset)))
    out = pd.DataFrame(rows)
    print(f"June 2026 replay: #1000 cap ${br['cap1000'] / 1e9:.2f}bn, #3000 cap ${br['cap3000'] / 1e6:.0f}m "
          f"(FTSE: breakpoint $5.7bn, smallest R2000 $146.4m)")
    print(f"list names matched to today's universe: adds {len(actA)}/{len(LA)}, deletes {len(actD)}/{len(LD)}")
    print(out.round(3).to_string(index=False))
    # largest false adds (what the live known-excluded step is for)
    fa = E[E.cid.isin(predA - actA)].sort_values("rank")[["pv", "name", "rank", "cap_jun", "country", "industry", "flags"]]
    print("\nlargest false adds:\n", fa.head(15).to_string(index=False))
    fd = Co[Co.cid.isin(predD - actD)][["pv", "name", "cur", "cap_jun", "px_jun"]]
    print("\nfalse deletes:\n", fd.head(15).to_string(index=False))
    md = A_md(out, br, len(actA), len(LA), len(actD), len(LD))
    p = C.SCORES / "validate_june2026.md"
    p.write_text(md)
    print(f"\nwritten {p}")


def A_md(out, br, ma, na, md_, nd):
    L = ["# June 2026 replay of the free-data pipeline (calibration, not a live prediction)", "",
         f"Rank-day rebuild on the 30 Apr 2026 close with shares fetched for the live run. #1000 cap "
         f"${br['cap1000'] / 1e9:.2f}bn and #3000 cap ${br['cap3000'] / 1e6:.0f}m (FTSE published $5.7bn and $146.4m).",
         f"FTSE list names matched to today's universe: adds {ma}/{na}, deletes {md_}/{nd}.", "",
         S.md_table(out), "",
         "Biases: today's shares (look-ahead); universe = names still listed today; no known-excluded step "
         "(the live run has one), so add precision is a lower bound; migrations not scored."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
