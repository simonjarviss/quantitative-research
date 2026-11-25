"""Step 2: build the S&P/TSX Composite change events 2006-2026 from Compustat idxcst_his (gvkeyx 000010).

Output: events.csv (one row per add/delete line), reviews.csv (reconstructed timeline per review).
Classification:
  review   = effective at the open after the 3rd-Friday close of Mar/Jun/Sep/Dec (Compustat date up to one
             day after the effective Monday is accepted) AND (deletes) the
             line keeps trading on the TSX after the effective date (>= 5 sessions in the next 30 days)
  offcycle = any other date, or a review-date delete whose line stops trading (M&A / delisting)
  linechg  = same gvkey leaves and re-enters within 5 days (share-line switch, not an index change)
Trade day T = last TSX session before the effective date; the TSX-only volume multiple on T vs the
median of T-20..T-1 is stored to confirm it (volume spike).
"""
import numpy as np, pandas as pd
from common import DATA, OUT, tsx_calendar, review_dates

cal = tsx_calendar()
rev = review_dates(cal)
rev.to_csv(OUT / "reviews.csv", index=False)

m = pd.read_parquet(DATA / "members.parquet")
m = m[m.gvkeyx == "000010"].copy()
m["fromdate"] = pd.to_datetime(m.fromdate); m["thru"] = pd.to_datetime(m.thru)
lines = pd.read_parquet(DATA / "lines.parquet")
nm = lines.set_index(["gvkey", "iid"])[["tic", "conm", "fic", "loc", "gsubind", "tpci", "exchg"]]

adds = m[(m.fromdate >= "2005-12-01")][["gvkey", "iid", "fromdate"]].rename(columns={"fromdate": "eff_date"})
adds["side"] = "add"
dels = m[m.thru.notna() & (m.thru >= "2005-12-01")][["gvkey", "iid", "thru"]].copy()
dels["eff_date"] = dels.thru + pd.Timedelta(days=1)
dels = dels.drop(columns="thru"); dels["side"] = "delete"
ev = pd.concat([adds, dels], ignore_index=True)

# line switches: same gvkey deleted and added within 5 days
a = ev[ev.side == "add"]; d = ev[ev.side == "delete"]
pair = a.merge(d, on="gvkey", suffixes=("_a", "_d"))
pair = pair[(pair.eff_date_a - pair.eff_date_d).abs() <= pd.Timedelta(days=5)]
sw = set(zip(pair.gvkey, pair.iid_a, pair.eff_date_a, ["add"] * len(pair))) | \
     set(zip(pair.gvkey, pair.iid_d, pair.eff_date_d, ["delete"] * len(pair)))
ev["linechg"] = [(g, i, e, s) in sw for g, i, e, s in zip(ev.gvkey, ev.iid, ev.eff_date, ev.side)]

# map to review: effective date in (T, eff + 1 day] of a review. Compustat dates most review deletions
# thru = Friday/Sunday, but 39 (mostly 2019-2022) carry thru = the effective Monday, i.e. eff_date = Tuesday.
ev["review"] = None
for r in rev.itertuples():
    hit = (ev.eff_date > r.T) & (ev.eff_date <= r.eff + pd.Timedelta(days=1))
    ev.loc[hit, "review"] = r.review

# trading history of each line (TSX prices)
s = pd.read_parquet(DATA / "secd.parquet", columns=["gvkey", "iid", "datadate", "prccd", "cshtrd"])
s = s[s.iid.str.endswith("C")]
key = set(zip(ev.gvkey, ev.iid))
s = s[[k in key for k in zip(s.gvkey, s.iid)]].sort_values(["gvkey", "iid", "datadate"])
first_px = s.groupby(["gvkey", "iid"]).datadate.min()
last_px = s.groupby(["gvkey", "iid"]).datadate.max()

out = []
for e in ev.itertuples():
    x = s[(s.gvkey == e.gvkey) & (s.iid == e.iid)]
    # trade day: the review's 3rd-Friday close for review-date events, else the last session before the effective date
    T = rev.loc[rev.review == e.review, "T"].iloc[0] if e.review is not None else cal[cal.searchsorted(e.eff_date) - 1]
    pre = x[(x.datadate < T) & (x.datadate >= T - pd.Timedelta(days=45))].tail(20)
    vT = x.loc[x.datadate == T, "cshtrd"]
    post = x[(x.datadate >= e.eff_date) & (x.datadate <= e.eff_date + pd.Timedelta(days=30))]
    out.append(dict(T=T, vol_T=vT.iloc[0] if len(vT) else np.nan,
                    adv20_med=pre.cshtrd.median() if len(pre) >= 10 else np.nan,
                    n_post30=len(post),
                    first_px=first_px.get((e.gvkey, e.iid), pd.NaT), last_px=last_px.get((e.gvkey, e.iid), pd.NaT)))
ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(out)], axis=1)
ev["mult_T_tsx"] = ev.vol_T / ev.adv20_med
ev["hist_days_before_T"] = (ev["T"] - ev.first_px).dt.days

def classify(r):
    if r.linechg:
        return "linechg"
    if r.review is None:
        return "offcycle"
    if r.side == "delete" and r.n_post30 < 5:
        return "offcycle_at_review"      # deleted on a review date but stopped trading (M&A/delisting)
    return "review"
ev["etype"] = ev.apply(classify, axis=1)
ev["new_listing"] = ev.hist_days_before_T < 183      # < ~6 months of TSX price history at T
ev = ev.join(nm, on=["gvkey", "iid"])
ev = ev[(ev.eff_date >= "2006-01-01") & (ev.eff_date <= "2026-06-30")].sort_values(["eff_date", "side", "tic"])
ev.to_csv(OUT / "events.csv", index=False)

print(ev.groupby(["etype", "side"]).size().unstack())
r = ev[ev.etype == "review"]
print("review events with new_listing:", r.groupby("side").new_listing.sum().to_dict())
print("median TSX volume multiple on T (review):", r.groupby("side").mult_T_tsx.median().round(2).to_dict())
print("share of review events with multiple >= 2:", r.groupby("side").mult_T_tsx.apply(lambda z: (z >= 2).mean()).round(2).to_dict())
print(r.groupby(r.eff_date.dt.year).side.value_counts().unstack().fillna(0).astype(int).T)
