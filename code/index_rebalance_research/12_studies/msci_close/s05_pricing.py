"""Step 5: close-pricing rows for MSCI review closes -> table_pricing_msci.csv.

Same columns, column order and definitions as ../close_dislocation/table_pricing.csv (s09_economics.py):
  quote (bp) = E[trade-day dislocation | bucket] + lambda x SD(reversal to the hold | bucket),  lambda = 0.25
  d0 = signed matched-control T-1 close -> T close; rev_<leg> = -(signed abnormal return from the T close to the exit:
  on1 = T+1 open, r1 = T+1 close, r1_5 = T+5 close, r1_20 = T+20 close); *_sd_name = SD across names; *_sd_basket =
  SD of the review-date mean; t clustered by review (CR1); flow_adv_median = exp(median log(|D_vis| / ADV60)), with
  D_vis the msci_pi visible net passive demand (ADV60: Canada consolidated, Europe primary venue).
Families: 'CA MSCI', 'EU MSCI', 'EU MSCI UK', 'EU MSCI ex-UK'. Groups: 'net buy' / 'net sell' (sign of D_vis).
Periods: 'all' and '2018+' (identical: the event sample starts with the Feb 2018 review); 'Oct2021+' (reviews from
Nov 2021): for Canada the first close under the TSX MOC redesign (18 Oct 2021); for Europe the same cut is written as
well because European net sells differ before and after it (RESULTS §3).
Buckets: flow_tercile 'all'; flow/ADV terciles only for cells with >= 90 events (>= 30 per tercile). crowding = 'all'
throughout (no run-up / short-interest variables are built for MSCI events).
Added 2026-10-15 (review fixes; the close_dislocation columns and their values are unchanged):
  * Event-type sub-rows for families 'CA MSCI' and 'EU MSCI': groups 'direct add' / 'up-migration' (subsets of
    'net buy') and 'direct delete' / 'down-migration' (subsets of 'net sell'), same periods and bucket rules.
  * Four columns after the close_dislocation ones, the expected-loss add-back applied to every row and hold:
      quote_adj_<hold> = quote_<hold> + max(0, -rev_<exit>_mean_bp)
    (hold_open <- rev_on1, hold1 <- rev_r1, hold5 and basket hold5 <- rev_r1_5). A negative expected reversal is a
    loss for the desk holding the other side to that exit; it is always added, whatever its t, and a positive one is
    never credited (s09 shows it separately as the desk's expected gain).
"""
import numpy as np
import pandas as pd

from common import CD, OUT, clt
from s04_tables import load

LAMBDA = 0.25
MIN_TERCILES = 90
ADJ = [("quote_adj_name_bp_hold_open", "quote_name_bp_hold_open", "rev_on1_mean_bp"),
       ("quote_adj_name_bp_hold1", "quote_name_bp_hold1", "rev_r1_mean_bp"),
       ("quote_adj_name_bp_hold5", "quote_name_bp_hold5", "rev_r1_5_mean_bp"),
       ("quote_adj_basket_bp_hold5", "quote_basket_bp_hold5", "rev_r1_5_mean_bp")]
ETYPE_GROUP = {"ADD_STD": "direct add", "MIG_UP": "up-migration", "DEL_STD": "direct delete", "MIG_DOWN": "down-migration"}


def pricing_rows(d, per, fam, grp, min_terciles=MIN_TERCILES):
    """s09_economics.py pricing block (flow buckets only; crowding = 'all')."""
    d = d.dropna(subset=["lflow", "s_d0"]).copy()
    if d.empty:
        return []
    d["allb"] = "all"
    groups = list(d.groupby(["allb"]))
    if len(d) >= min_terciles:
        d["flow_t"] = pd.qcut(d.lflow, 3, labels=["low", "mid", "high"])
        groups += list(d.groupby(["flow_t"], observed=True))
    rows = []
    for keys, b in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = dict(period=per, family=fam, group=grp, flow_tercile=keys[0], crowding="all",
                 n=len(b), flow_adv_median=np.exp(b.lflow.median()))
        m, _, t = clt(b.s_d0.values, b.review.values)
        r.update(d0_mean_bp=m * 1e4, d0_t=t, d0_median_bp=b.s_d0.median() * 1e4)
        for leg in ["on1", "r1", "r1_5", "r1_20"]:
            rv = -b[f"s_{leg}"]
            m, _, t = clt(rv.values, b.review.values)
            bas = b.assign(rv=rv).groupby("review").rv.mean()
            r.update({f"rev_{leg}_mean_bp": m * 1e4, f"rev_{leg}_t": t, f"rev_{leg}_sd_name_bp": rv.std() * 1e4,
                      f"rev_{leg}_sd_basket_bp": bas.std(ddof=1) * 1e4 if len(bas) > 2 else np.nan})
        r["quote_name_bp_hold5"] = r["d0_mean_bp"] + LAMBDA * r["rev_r1_5_sd_name_bp"]
        r["quote_basket_bp_hold5"] = r["d0_mean_bp"] + LAMBDA * r["rev_r1_5_sd_basket_bp"]
        r["quote_name_bp_hold1"] = r["d0_mean_bp"] + LAMBDA * r["rev_r1_sd_name_bp"]
        r["quote_name_bp_hold_open"] = r["d0_mean_bp"] + LAMBDA * r["rev_on1_sd_name_bp"]
        for c, q, rv in ADJ:          # expected-loss add-back (2026-10-15)
            r[c] = r[q] + max(0.0, -r[rv]) if np.isfinite(r[rv]) else np.nan
        rows.append(r)
    return rows


def main():
    X = load()
    cols = pd.read_csv(CD / "table_pricing.csv", nrows=0).columns.tolist()
    fams = [("CA MSCI", X.market == "CA"), ("EU MSCI", X.market == "EU"),
            ("EU MSCI UK", (X.market == "EU") & X.uk), ("EU MSCI ex-UK", (X.market == "EU") & ~X.uk)]
    rows = []
    for fam, m in fams:
        for per in ["all", "2018+", "Oct2021+"]:
            d0 = X[m.values]
            if per == "Oct2021+":
                d0 = d0[d0.event_date >= "2021-10-18"]
            elif per == "2018+":
                d0 = d0[d0.event_date >= "2018-01-01"]
            for grp, d in d0.groupby("grp"):
                rows += pricing_rows(d, per, fam, grp)
            if fam in ("CA MSCI", "EU MSCI"):          # event-type sub-rows (2026-10-15)
                for ev in ["ADD_STD", "MIG_UP", "DEL_STD", "MIG_DOWN"]:
                    d = d0[d0.event == ev]
                    if len(d):
                        rows += pricing_rows(d, per, fam, ETYPE_GROUP[ev])
    pt = pd.DataFrame(rows)[cols + [c for c, _, _ in ADJ]]
    pt.to_csv(OUT / "table_pricing_msci.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(pt[pt.period != "2018+"][["period", "family", "group", "flow_tercile", "n", "flow_adv_median", "d0_mean_bp", "d0_t",
                                    "rev_on1_mean_bp", "rev_on1_t", "rev_r1_mean_bp", "rev_r1_t", "rev_r1_5_mean_bp",
                                    "rev_on1_sd_name_bp", "rev_r1_5_sd_name_bp", "quote_name_bp_hold_open",
                                    "quote_name_bp_hold1", "quote_name_bp_hold5"] + [c for c, _, _ in ADJ[:3]]
             ].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
