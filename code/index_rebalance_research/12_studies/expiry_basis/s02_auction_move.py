"""Per index and day: the cash index's move after the futures settlement (estimated closing-auction move, bp).

b_t = ln F_t - ln S_t, with F the daily settlement of the front contract (nearest last trading day after t, so on
an expiry day the next contract) and S the index total-return close. Carry is smooth, so
  a_t = -(b_t - interp(b_{t-1}, b_{t+1}))    (linear in calendar days, same contract on all three days)
is the part of the index close that the futures settlement did not see. Where the futures settle before the cash
closing auction (Eurex, ICE, MEFF), a_t estimates the index-level auction move; where they settle on the close,
a_t is ~0. Also stores the futures return from settle t to settle t+1 (same contract), for the permanence test.
"""
import numpy as np
import pandas as pd
from common import DATA, PAIRS, MSCI, START, third_friday

np.seterr(invalid="ignore", divide="ignore")

ci = pd.read_parquet(DATA / "contracts.parquet"); ci["ltd"] = pd.to_datetime(ci.lasttrddate)
px = pd.read_parquet(DATA / "fut_px.parquet"); px["date_"] = pd.to_datetime(px.date_)
ix = pd.read_parquet(DATA / "idx.parquet"); ix["valuedate"] = pd.to_datetime(ix.valuedate)
eu_qexp = set(ci[ci.contrcode == 801].ltd.dropna())      # FESX quarterly last trading days = European Q-expiry

rows = []
for tk, cc, ic, nm in PAIRS:
    c = ci[ci.contrcode == cc][["futcode", "ltd"]].dropna().sort_values("ltd")
    piv = px[px.futcode.isin(c.futcode) & (px.settlement > 0)].pivot_table(index="date_", columns="futcode",
                                                                          values="settlement")
    s = ix[ix.dsindexcode == ic].set_index("valuedate")["ri"].dropna()
    s = s[s > 0]
    dl = list(s.index)
    for i in range(1, len(dl) - 1):
        t = dl[i]
        if t < pd.Timestamp(START):
            continue
        later = c[c.ltd > t]
        if later.empty:
            continue
        fc = later.futcode.iloc[0]
        tm, tp = dl[i - 1], dl[i + 1]
        try:
            F = [piv.at[d, fc] for d in (tm, t, tp)]
        except KeyError:
            continue
        if any(pd.isna(F)):
            continue
        b = [np.log(F[k]) - np.log(s[d]) for k, d in enumerate((tm, t, tp))]
        w = (t - tm).days / (tp - tm).days
        rows.append((tk, nm, t, -(b[1] - ((1 - w) * b[0] + w * b[2])) * 1e4, np.log(F[2] / F[1]) * 1e4))

d = pd.DataFrame(rows, columns=["fut", "idx", "date", "a_bp", "fret_next_bp"])
cad = d.idx.str.startswith("S&P/TSX")
tf = d.date.apply(lambda x: third_friday(x.year, x.month))
qx = np.where(cad, (d.date == tf) & d.date.dt.month.isin([3, 6, 9, 12]), d.date.isin(eu_qexp))
alld = pd.Series(sorted(ix.valuedate.unique()))
month_end = set(alld.groupby(alld.dt.to_period("M")).max())
q_month = d.date.dt.month.isin([3, 6, 9, 12])
d["type"] = np.select([qx, d.date.isin(MSCI), d.date.isin(month_end) & q_month, d.date.isin(month_end), d.date == tf],
                      ["1 Q-expiry/rebal", "2 MSCI review", "3 quarter-end", "4 other month-end", "5 monthly expiry"],
                      "6 other")
d.to_csv(DATA / "auction_move_daily.csv", index=False)

# Roll: on each quarterly expiry day the hedge must go in the next contract. Volume in that contract on the
# expiry day, and in the 5 days before, relative to the expiring contract's ADV over days -30..-11.
roll = []
for tk, cc, ic, nm in PAIRS:
    c = ci[ci.contrcode == cc][["futcode", "ltd"]].dropna().sort_values("ltd")
    cq = c[c.ltd.dt.month.isin([3, 6, 9, 12])].reset_index(drop=True)
    v = px[px.futcode.isin(c.futcode)].pivot_table(index="date_", columns="futcode", values="volume")
    for k in range(len(cq) - 1):
        E, f0, f1 = cq.ltd[k], cq.futcode[k], cq.futcode[k + 1]
        if E < pd.Timestamp(START) or E > v.index.max() or f0 not in v or f1 not in v:
            continue
        days = v.index[v.index <= E]
        if len(days) < 31 or days[-1] != E:
            continue
        base = v.loc[days[-30:-10], f0].mean()
        if not base or np.isnan(base):
            continue
        roll.append((tk, nm, E, v.at[E, f1] / base, v.loc[days[-6:-1], f1].sum() / v.loc[days[-6:-1], [f0, f1]].sum().sum(),
                     v.at[E, f0] / base if E in v.index else np.nan))
r = pd.DataFrame(roll, columns=["fut", "idx", "expiry", "next_vol_on_E_x_front_adv", "next_share_E-5..E-1",
                                "expiring_vol_on_E_x_adv"])
r.to_csv(DATA / "roll.csv", index=False)
print(len(d), "index-days;", len(r), "expiries")
