"""Step 12 (optional US refinement, one recon only): the closing-auction deviation on the Russell recon day 27 Jun 2025.

Last continuous price = close of the 15:59 ET one-minute SIP bar (Alpaca, feed=sip, asof=2025-06-27 symbol mapping),
pulled through the Alpaca MCP on 2026-10-11 for the 405 events of 2025 plus IWM, and saved as
data/alpaca_1559_2025.csv (symbol, bar close, bar volume; transcribed from the MCP response; VRM returned no bar).
Official close = CRSP dsf_v2 dlyclose on T (data/opens_us.parquet).
auction deviation = official close / 15:59 bar close - 1, signed by side (+ = in the direction of index demand),
minus IWM's own deviation (market move in the last minute). Also: share of the trade-day abnormal move (d0) that sits
in the auction, and the next-open reversal (on1) by the size of the deviation.
Why only 2025: each name-day needs one MCP call slot and the result has to be copied out of the tool response; the
REST /v2/stocks/auctions route needs the user's keys outside the MCP. Extending it to 2016-2024 is a REST job.
Output: table_auction_2025.csv, auction_2025_names.csv
"""
import numpy as np
import pandas as pd
from common import OUT, DATA, RUS, summ

a = pd.read_csv(DATA / "alpaca_1559_2025.csv")
# symbols requested from Alpaca = CRSP ticker at the June 2025 month-end, class suffix ' ' -> '.'
m = pd.read_parquet(RUS / "_cache" / "msf.parquet", columns=["permno", "mthcaldt", "ticker"])
m = m[(m.mthcaldt.dt.year == 2025) & (m.mthcaldt.dt.month == 6)]
s = m.assign(sym=m.ticker.str.replace(" ", ".", regex=False))[["permno", "sym"]].drop_duplicates("permno")
o = pd.read_parquet(DATA / "opens_us.parquet")
o = o[pd.to_datetime(o.dlycaldt) == "2025-06-27"].set_index("permno")
E = pd.read_csv(OUT / "events_legs.csv", low_memory=False)
E = E[(E.market == "US") & (E.year == 2025)].copy()
E["permno"] = E["id"].astype(int)
x = E.merge(s[["permno", "sym"]], on="permno", how="left").merge(a, on="sym", how="left")
x["close_off"] = x.permno.map(o.dlyclose)
iwm_dev = o.dlyclose.get(88222) / a.set_index("sym").c1559["IWM"] - 1
x["dev"] = x.close_off / x.c1559 - 1
x["dev_s"] = x.side * (x.dev - iwm_dev)
x["flag_big"] = x.dev.abs() > 0.2
x["px5"] = np.where(x.price >= 5, "price >= $5", "price < $5")
x[["permno", "sym", "grp", "side", "price", "c1559", "v1559", "close_off", "dev", "dev_s", "s_d0", "s_on1", "s_r1", "flag_big"]].to_csv(
    OUT / "auction_2025_names.csv", index=False)
rows = []
y = x[~x.flag_big & x.dev.notna()]
for (g, p), d in pd.concat([y.assign(px5="all"), y]).groupby(["grp", "px5"]):
    r = dict(group=g, price=p, **{f"dev_{k}": v for k, v in summ(d.dev_s.values, np.arange(len(d))).items()})
    r["d0_mean_bp"] = d.s_d0.mean() * 1e4
    r["auction_share_of_d0"] = d.dev_s.mean() / d.s_d0.mean() if d.s_d0.mean() != 0 else np.nan
    r["on1_mean_bp"] = d.s_on1.mean() * 1e4
    r["corr_dev_on1"] = d[["dev_s", "s_on1"]].corr().iloc[0, 1]
    rows.append(r)
t = pd.DataFrame(rows)
t.attrs["iwm_dev_bp"] = iwm_dev * 1e4
t.to_csv(OUT / "table_auction_2025.csv", index=False)
pd.set_option("display.width", 250)
print("IWM deviation bp", round(iwm_dev * 1e4, 1), "| matched", x.c1559.notna().sum(), "of", len(x), "| |dev|>20%:", int(x.flag_big.sum()))
print(t[["group", "price", "dev_n", "dev_mean_bp", "dev_t", "dev_median_bp", "dev_trim_bp", "d0_mean_bp", "auction_share_of_d0",
         "on1_mean_bp", "corr_dev_on1"]].round(2).to_string(index=False))
