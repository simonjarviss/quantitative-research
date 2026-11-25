"""Step 14 (added 2026-10-14): S&P/TSX adds and deletes split at the TSX MOC redesign (first close under the new
rules: 18 Oct 2021). Uses events_legs.csv only. Signed abnormal returns (+ = with index demand), clustered by review.
Legs: d0 = T-1 close -> T close, d0_id = T open -> T close, on1 = T close -> T+1 open, id1 = T+1 open -> close,
r1 = T close -> T+1 close, r1_5 = T close -> T+5 close.
Output: table_tsx_redesign.csv
"""
import numpy as np
import pandas as pd
from common import OUT, summ

E = pd.read_csv(OUT / "events_legs.csv", low_memory=False)
c = E[E.market == "CA"].copy()
c["event_date"] = pd.to_datetime(c.event_date)
c["period"] = np.select([c.event_date < "2016-01-01", c.event_date < "2021-10-18"], ["2006-2015", "2016 - Sep 2021"],
                        "Oct 2021 - Jun 2026")
rows = []
for (g, p), d in c.groupby(["group", "period"]):
    for leg in ["s_d0", "s_d0_id", "s_on1", "s_id1", "s_r1", "s_r1_5"]:
        x = d[leg].dropna()
        s = summ(x.values, d.loc[x.index, "event_date"].values)
        rows.append(dict(group=g, period=p, leg=leg[2:], **s))
    dd = d.groupby("event_date").s_r1.mean()
    rows.append(dict(group=g, period=p, leg="r1: reviews with mean > 0", n=len(dd), mean_bp=(dd > 0).mean()))
t = pd.DataFrame(rows)
t.to_csv(OUT / "table_tsx_redesign.csv", index=False)
pd.set_option("display.width", 250)
print(t[["group", "period", "leg", "n", "n_dates", "mean_bp", "t", "median_bp", "pos_share"]].round(2).to_string(index=False))
