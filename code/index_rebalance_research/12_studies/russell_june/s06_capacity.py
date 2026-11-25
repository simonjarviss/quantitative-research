"""Step 6: rough capacity of the capturable (predicted) portfolios, 2016-2025.

Per predicted name at horizon h:
  ADV$  = mean daily dollar volume (CRSP prc x vol) over the 20 trading days ending at h
  spread = closing quoted spread (ask-bid)/mid from CRSP dsf_v2 at h (last quote within 4 days)
  position at 10% participation built over D days = 0.10 x ADV$ x D
     D = 10 (build in the first two weeks, hold to T)    -> "build10"
     D = trading days from h+1 to T-1 (literal 10% every day of the window; upper bound) -> "fullwin"
  gross P&L = position x the stock's own abnormal return h->T (= dollar-weighted CAR), signed by side
Exit at the recon close is not the binding constraint (recon-day volume is ~10x normal for these names, s05).
Output: output/capacity_by_year.csv, output/capacity_summary.csv
"""
import numpy as np
import pandas as pd
from common import CACHE, OUT, MAIN_YEARS, SIDE
import lib

S = pd.read_csv(OUT / "car_stock_level.csv.gz")
Q = pd.read_parquet(CACHE / "quotes.parquet")
Q = Q[(Q.bid > 0) & (Q.ask >= Q.bid)]
Q["spr"] = (Q.ask - Q.bid) / ((Q.ask + Q.bid) / 2)
Q["dlycaldt"] = pd.to_datetime(Q.dlycaldt)
Q = Q.sort_values("dlycaldt")
rows = []
for y in MAIN_YEARS:
    hd = lib.horizon_dates(y)
    for h in ["H6", "H3", "HR"]:
        f = hd[h]
        e = S[(S.portfolio == f"PRED-{h}") & (S.year == y) & (S.window == f"{h}_T")].copy()
        d = lib.daily(lib.tshift(f, -19), f, "permno, dlycaldt, prc, vol", f"AND permno IN ({','.join(map(str, e.permno.astype(int)))})")
        adv = (d.prc.abs() * d.vol).groupby(d.permno).mean()
        q = Q[(Q.dlycaldt <= f) & (Q.dlycaldt >= f - pd.Timedelta(days=6))].drop_duplicates("permno", keep="last").set_index("permno").spr
        e["adv"] = e.permno.map(adv)
        e["spread"] = e.permno.map(q)
        D_full = int(((lib.tdays() > f) & (lib.tdays() < hd["T"])).sum())
        e["side"] = e.group.map(SIDE)
        for lab, D in [("build10", 10), ("fullwin", D_full)]:
            e[f"pos_{lab}"] = 0.10 * e.adv * D
            e[f"pnl_{lab}"] = e[f"pos_{lab}"] * e.ar * e.side
        e["horizon"], e["D_full"] = h, D_full
        rows.append(e)
E = pd.concat(rows, ignore_index=True)
by = E.groupby(["horizon", "year", "group"]).agg(n=("permno", "size"), median_adv=("adv", "median"), median_spread=("spread", "median"),
                                                  mean_car=("ar", "mean"), D_full=("D_full", "first"),
                                                  pos_build10=("pos_build10", "sum"), pnl_build10=("pnl_build10", "sum"),
                                                  pos_fullwin=("pos_fullwin", "sum"), pnl_fullwin=("pnl_fullwin", "sum")).reset_index()
by.to_csv(OUT / "capacity_by_year.csv", index=False)
E2 = pd.concat([E, E.assign(group=np.where(E.side > 0, "BUY_all", "SELL_all")), E.assign(group="LONG_SHORT")], ignore_index=True)
out = []
for (h, g), d in E2.groupby(["horizon", "group"]):
    yr = d.groupby("year").agg(pos=("pos_build10", "sum"), pnl=("pnl_build10", "sum"))
    signed_car = (d.ar * d.side).mean()
    out.append(dict(horizon=h, group=g, names_per_recon=len(d) / d.year.nunique(), median_adv_musd=d.adv.median() / 1e6,
                    median_spread_bp=d.spread.median() * 1e4, ew_signed_car_pct=signed_car * 100,
                    deploy_build10_musd=yr.pos.mean() / 1e6,
                    pnl_build10_at_ew_car_musd=yr.pos.mean() * signed_car / 1e6,
                    pnl_build10_dollar_weighted_musd=yr.pnl.mean() / 1e6,
                    pnl_build10_years_positive=int((yr.pnl > 0).sum()), n_years=len(yr),
                    deploy_fullwin_musd=d.groupby("year").pos_fullwin.sum().mean() / 1e6,
                    pnl_fullwin_dollar_weighted_musd=d.groupby("year").pnl_fullwin.sum().mean() / 1e6,
                    median_name_build10_musd=d.pos_build10.median() / 1e6))
s = pd.DataFrame(out)
s.to_csv(OUT / "capacity_summary.csv", index=False)
pd.set_option("display.width", 250)
print(s.round(2).to_string(index=False))
