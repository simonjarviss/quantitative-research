"""Step 16 (added 2026-10-15): close-pricing rows for S&P/TSX review events since the TSX MOC redesign (first close
under the new rules: 18 Oct 2021), with the pricing logic of s09_economics.py, unchanged:
  quote (bp) = E[trade-day dislocation | bucket] + lambda x SD(reversal to the hold | bucket),  lambda = 0.25
  d0 = signed matched-control T-1 close -> T close; rev_<leg> = -(signed abnormal return from the T close to the exit)
  (on1 = T+1 open, r1 = T+1 close, r1_5 = T+5 close, r1_20 = T+20 close); *_sd_name = SD across names,
  *_sd_basket = SD of the review-date mean; t clustered by review (CR1, common.clt); flow = pi x float / ADV60.
Buckets: s09 forms flow/ADV terciles and run-up (crowding) halves only for family x side cells with >= 60 events and
skips smaller cells. Here the 'all' x 'all' row is always written (S&P/TSX deletes have 52 events since Oct 2021),
and terciles / halves follow s09's >= 60 rule (adds only).
Reads data/legs_*.parquet through s06_tables.load(); writes ONLY table_pricing_post2021.csv (same columns as
table_pricing.csv; period 'Oct2021+'; family 'CA S&P/TSX'). table_pricing.csv and every other output are untouched.
Check printed at the end: the same function on the 2016+ S&P/TSX cells reproduces table_pricing.csv.
"""
import numpy as np
import pandas as pd
from common import OUT, clt
from s06_tables import load

LAMBDA = 0.25
REDESIGN = pd.Timestamp("2021-10-18")
MIN_BUCKETS = 60


def pricing_rows(d, per, fam, grp, min_buckets=MIN_BUCKETS):
    """s09_economics.py pricing block for one family x side cell; returns a list of row dicts."""
    d = d.dropna(subset=["lflow", "s_d0"]).copy()
    if d.empty:
        return []
    d["allb"] = "all"
    groups = list(d.groupby(["allb"]))
    if len(d) >= min_buckets:
        d["flow_t"] = pd.qcut(d.lflow, 3, labels=["low", "mid", "high"])
        med = d.runup3_s.median()
        d["crowd"] = np.where(d.runup3_s.isna(), "no run-up data",
                              np.where(d.runup3_s > med, "run-up above median", "run-up below median"))
        groups += (list(d.groupby(["flow_t"], observed=True)) + list(d.groupby(["allb", "crowd"]))
                   + list(d.groupby(["flow_t", "crowd"], observed=True)))
    rows = []
    for keys, b in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = dict(period=per, family=fam, group=grp, flow_tercile=keys[0], crowding=keys[1] if len(keys) > 1 else "all",
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
        rows.append(r)
    return rows


def main():
    df = load()
    ca = df[df.fam == "CA S&P/TSX"].copy()
    ca["event_date"] = pd.to_datetime(ca.event_date)
    cols = pd.read_csv(OUT / "table_pricing.csv", nrows=0).columns.tolist()
    post = ca[ca.event_date >= REDESIGN]
    rows = []
    for grp, d in post.groupby("grp"):
        rows += pricing_rows(d, "Oct2021+", "CA S&P/TSX", grp)
    pt = pd.DataFrame(rows)[cols]
    pt.to_csv(OUT / "table_pricing_post2021.csv", index=False)
    pd.set_option("display.width", 250)
    print(pt[pt.crowding == "all"][["period", "family", "group", "flow_tercile", "n", "flow_adv_median", "d0_mean_bp", "d0_t",
                                     "rev_on1_mean_bp", "rev_r1_mean_bp", "rev_r1_t", "rev_on1_sd_name_bp",
                                     "quote_name_bp_hold_open", "quote_name_bp_hold1", "quote_name_bp_hold5"]].round(1).to_string(index=False))
    # reproduction check against s09 (2016+ S&P/TSX cells of table_pricing.csv); nothing is written
    ref = pd.read_csv(OUT / "table_pricing.csv")
    ref = ref[(ref.family == "CA S&P/TSX") & (ref.period == "2016+")]
    chk = []
    for grp, d in ca[ca.period == "2016+"].groupby("grp"):
        chk += pricing_rows(d, "2016+", "CA S&P/TSX", grp)
    chk = pd.DataFrame(chk)[cols]
    k = ["period", "family", "group", "flow_tercile", "crowding"]
    m = ref.merge(chk, on=k, suffixes=("_ref", "_new"))
    num = [c for c in cols if c not in k]
    diff = max(np.nanmax(np.abs(m[c + "_ref"].astype(float) - m[c + "_new"].astype(float))) for c in num)
    print(f"check vs table_pricing.csv (CA S&P/TSX 2016+): {len(m)} of {len(ref)} rows matched, max abs diff {diff:.2e}")


if __name__ == "__main__":
    main()
