"""Step 2: the MSCI Canada / MSCI Europe Standard-index events, their sign, and the drop rules -> events_msci.csv.

Source: ../msci_pi/demand.csv (holdings-diff events of CSCA / IMEA / WSML / SWDA per line and review, with visible
net passive demand D_vis = pi_std x dF_std + pi_sc x dF_sc in shares; adv60, vol_T from the sister panels).
Events kept: ADD_STD, DEL_STD, MIG_UP (Small Cap -> Standard), MIG_DOWN (Standard -> Small Cap), reviews
Feb 2018 - May 2026 (T <= 2026-06-24, the end of Datastream prices).
Side = sign(D_vis): +1 = net visible passive buying, -1 = net selling (as close_dislocation signs the DAX family).
Drop rules, applied in this order (the first that applies is recorded in `drop`):
  1 no_panel_line     no line in the sister panel (non-European / non-TSX primary listing: Spotify, Nebius,
                      Ferguson after its NYSE move, EDF, Endeavour Mining)
  2 not_material      |D_vis| < 0.10% of shares outstanding (no material net visible demand)
  3 reclassification  a deletion that MSCI World (SWDA) still holds after T, or an addition SWDA already held
                      before T: a move between MSCI country indices (CRH, Flutter, CNH, Amrize to the US). World,
                      ACWI and EAFE-wide demand nets to ~0 and the trade moves to another line, so D_vis is wrong.
  4 isin_change       the same name is added and deleted at the same review (a new ISIN after a share
                      consolidation or split: Melrose 2021-08, JD Sports 2021-11)
  5 uk_holiday        UK line at a review whose T was a UK bank holiday (31 Aug 2020): implemented a day later
Events with no Datastream close at T-1 or T drop out in s03 (reported there).
Output: events_msci.csv (every candidate event with its drop reason; drop == '' are the study events).
"""
import numpy as np
import pandas as pd

from common import DS_END, MATERIAL, MPI, OUT, STD_EVENTS, TSX_REDESIGN, UK_HOLIDAY_T


def main():
    d = pd.read_csv(MPI / "demand.csv", parse_dates=["T"])
    ev = d[d.event.isin(STD_EVENTS) & (d["T"] <= DS_END)].copy()
    ev["direction"] = np.where(ev.event.isin(["ADD_STD", "MIG_UP"]), 1, -1)
    ev["side"] = np.sign(ev.D_vis).astype(int)
    ev["dvis_pct_so"] = ev.D_vis / ev.shares_out * 100
    ev["dvis_adv"] = ev.D_vis / ev.adv60
    ev["etype"] = np.where(ev.event.str.startswith("MIG"), "migration", "direct")
    ev["drop"] = ""
    ev.loc[ev.col.isna() | ev.shares_out.isna() | ev.price_Tm1.isna(), "drop"] = "no_panel_line"
    m = (ev["drop"] == "") & ~(ev.D_vis.abs() >= MATERIAL * ev.shares_out)
    ev.loc[m, "drop"] = "not_material"
    recl = ((ev.event == "DEL_STD") & (ev.h1_wld > 0)) | ((ev.event == "ADD_STD") & (ev.h0_wld > 0))
    ev.loc[(ev["drop"] == "") & recl, "drop"] = "reclassification"
    key = ev.review + "|" + ev.mkt + "|" + ev.name.str.upper().str.strip()
    both = key.map(ev.groupby(key).event.agg(lambda s: {"ADD_STD", "DEL_STD"} <= set(s)))
    ev.loc[(ev["drop"] == "") & both, "drop"] = "isin_change"
    ev.loc[(ev["drop"] == "") & ev["T"].isin(UK_HOLIDAY_T) & (ev.ctry == "GB"), "drop"] = "uk_holiday"
    ev["sign_vs_direction"] = np.where(ev.side == ev.direction, "same", "opposite")
    ev["group"] = np.where(ev.side > 0, "net buy", "net sell")
    ev["uk"] = ev.ctry.eq("GB")
    ev["period_ca"] = np.where(ev["T"] >= TSX_REDESIGN, "Oct2021+", "pre-Oct2021")
    keep = ["review", "T", "mkt", "isin", "ticker", "name", "ctry", "uk", "event", "etype", "direction", "side", "group",
            "sign_vs_direction", "D_vis", "D_std", "D_sc", "dvis_pct_so", "dvis_adv", "pi_std", "pi_sc", "dF_std",
            "dF_sc", "h0_wld", "h1_wld", "col", "adv60", "med20", "vol_T", "price_Tm1", "shares_out", "fx_eur",
            "period_ca", "drop"]
    ev = ev[keep].sort_values(["mkt", "T", "event", "name"]).reset_index(drop=True)
    ev.to_csv(OUT / "events_msci.csv", index=False)
    pd.set_option("display.width", 250)
    print(ev.groupby(["mkt", "drop"]).size().unstack(0).fillna(0).astype(int))
    k = ev[ev["drop"] == ""]
    print(k.groupby(["mkt", "event", "group"]).size().unstack(-1).fillna(0).astype(int))
    print(k.groupby(["mkt", "period_ca"]).size())
    print("sign opposite to the event direction (kept):", (k.sign_vs_direction == "opposite").sum())
    print(ev[ev["drop"] != ""][["review", "mkt", "name", "event", "dvis_pct_so", "drop"]].to_string())


if __name__ == "__main__":
    main()
