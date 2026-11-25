"""Step 11 (descriptive only): IBKR indicative borrow around the 2025-26 trade days, from the iBorrowDesk one-year file
(index_rebalance_research/14_borrow_data/iborrowdesk/, window 2025-10-13 -> 2026-10-09).

Covered events in this study's sample: S&P/TSX reviews Dec 2025, Mar 2026, Jun 2026 (TSX symbols) and STOXX 600
changes on UK, Swedish and Italian lines at the same reviews (ISIN match). Russell June 2026 is in the borrow file but
not in this study (CRSP ends 2025-12-31). n is tiny: no inference.
Per name: indicative fee (% p.a.) and IBKR availability on the last borrow date on or before T-7 calendar days, on T,
and on or before T+7; plus the name's signed trade-day and reversal returns from events_legs.csv.
Output: table_borrow_2025_26.csv
"""
import json
import numpy as np
import pandas as pd
from common import OUT, TSX, EU, STUDIES

IBD = STUDIES.parent / "14_borrow_data" / "iborrowdesk"
u = pd.read_csv(IBD / "universe.csv")
man = pd.read_csv(IBD / "manifest.csv")
bad = set(man.loc[man["match"].astype(str).str.contains("MISMATCH"), "symbol"])
E = pd.read_csv(OUT / "events_legs.csv", low_memory=False)
E["event_date"] = pd.to_datetime(E.event_date)
E = E[(E.event_date >= "2025-10-20") & E.market.isin(["CA", "EU"])]
ev = pd.read_csv(TSX / "events.csv")
ev["lid"] = ev.gvkey.astype(str).str.zfill(6) + "_" + ev.iid
tic = ev.drop_duplicates("lid").set_index("lid").tic.astype(str).str.rstrip(".")
Lp = pd.read_parquet(EU / "data" / "panel_lines.parquet").set_index("lid")
tsx_map = u[u.group == "tsx"].drop_duplicates("src_ticker").set_index("src_ticker").ibd_symbol
isin_map = u[u.group == "stoxx"].drop_duplicates("isin").set_index("isin").ibd_symbol


def load(sym):
    f = IBD / "raw" / f"{sym}.json"
    if not f.exists() or sym in bad:
        return None
    d = pd.DataFrame(json.loads(f.read_text()).get("daily", []))
    if d.empty:
        return None
    d["date"] = pd.to_datetime(d.date)
    return d.sort_values("date").set_index("date")


rows = []
for r in E.itertuples():
    sym = tsx_map.get(tic.get(r.id, "")) if r.market == "CA" else isin_map.get(Lp["isin"].get(r.id, ""), None)
    if not isinstance(sym, str):
        continue
    d = load(sym)
    if d is None:
        continue
    T = r.event_date

    def at(day):
        x = d[d.index <= day]
        return (x.fee.iloc[-1], x.available.iloc[-1]) if len(x) else (np.nan, np.nan)
    fb, ab = at(T - pd.Timedelta(days=7)); f0, a0 = at(T); fa, aa = at(T + pd.Timedelta(days=7))
    rows.append(dict(family=r.fam, review=r.review, group=r.grp, name=r.name, ibd_symbol=sym, fee_Tm7=fb, fee_T=f0, fee_Tp7=fa,
                     avail_Tm7=ab, avail_T=a0, avail_Tp7=aa, s_d0_bp=r.s_d0 * 1e4, s_r1_bp=r.s_r1 * 1e4,
                     s_r1_5_bp=r.s_r1_5 * 1e4))
b = pd.DataFrame(rows)
b.to_csv(OUT / "table_borrow_2025_26.csv", index=False)
pd.set_option("display.width", 250)
print(b.groupby(["family", "group"]).agg(n=("name", "size"), fee_Tm7_med=("fee_Tm7", "median"), fee_T_med=("fee_T", "median"),
                                         fee_Tp7_med=("fee_Tp7", "median"), fee_T_max=("fee_T", "max"),
                                         share_fee_gt_5=("fee_T", lambda v: (v > 5).mean()), d0_mean=("s_d0_bp", "mean"),
                                         r1_mean=("s_r1_bp", "mean")).round(2).to_string())
