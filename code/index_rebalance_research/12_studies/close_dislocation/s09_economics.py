"""Step 9: economics of supplying liquidity at the trade-day close, capacity, and a desk pricing rule.

Liquidity supplier (LP) = takes the OTHER side of index demand at the T close (sells index buys, buys index sells) and
exits at the T+1 open, T+1 close, T+5 close or T+20 close. Per unit supplied, gross P&L = -(signed return from the T
close to the exit). Three hedges: none (raw), index (index-adjusted) and matched controls (abnormal; the study's
main basis). Gross of commissions, spread, impact, borrow (needed to short index buys) and taxes (UK 0.5% / FTT on the
LP's buys of index sells unless it is an exempt intermediary).
Per event date (one Russell recon, one S&P/TSX or STOXX/DAX review): the LP supplies s = 10% of the predicted demand
D = pi x float shares in every name of the group, priced at the T-1 close (USD / CAD / EUR). P&L per event date =
sum over names of s x D x (-signed abnormal return). t across event dates.
Pricing rule for a risk (guaranteed-close-style) trade quoted against the pre-close (T-1 close) reference:
  quote (bp) = E[trade-day dislocation | bucket] + lambda x SD(reversal to the hold | bucket)
reported by family x side x flow/ADV tercile x run-up (crowding) half, 2016+ events, lambda = 0.25 (per name) and the
basket version (SD of the event-date mean). E[reversal] is shown separately: it is the desk's expected gain on top.
Outputs: table_lp_pnl.csv, table_pricing.csv
"""
import numpy as np
import pandas as pd
from common import OUT, summ, clt
from s06_tables import load

EXITS = [("on1", "T+1 open"), ("r1", "T+1 close"), ("r1_5", "T+5 close"), ("r1_20", "T+20 close")]
S = 0.10
LAMBDA = 0.25

df = load()
df["D_ccy"] = df.demand_ccy
rows = []
us = df[df.market == "US"]
px = us.assign(grp=us.grp + np.where(us.price >= 5, " | price >= $5", " | price < $5"))
for (fam, grp, per), d in pd.concat([df, df.assign(period="all"), px, px.assign(period="all")]).groupby(["fam", "grp", "period"]):
    for leg, lab in EXITS:
        for basis, pre in [("matched control", "s_"), ("index-hedged", "si_"), ("unhedged", "sr_")]:
            pnl = -d[pre + leg]
            s = summ(pnl.values, d.review.values)
            x = d.assign(p=pnl).dropna(subset=["p"])
            ed = x.groupby("review").agg(ew=("p", "mean"), cap=("D_ccy", lambda v: S * np.nansum(v)))
            ed["pnl_ccy"] = x.groupby("review").apply(lambda g: S * np.nansum(g.D_ccy * g.p), include_groups=False)
            t_pnl = ed.pnl_ccy.mean() / (ed.pnl_ccy.std(ddof=1) / np.sqrt(len(ed))) if len(ed) > 2 else np.nan
            rows.append(dict(family=fam, group=grp, period=per, exit=lab, leg=leg, basis=basis, ccy=d.ccy.iloc[0],
                             **{f"name_{k}": v for k, v in s.items()},
                             n_event_dates=len(ed), basket_mean_bp=ed.ew.mean() * 1e4,
                             basket_t=ed.ew.mean() / (ed.ew.std(ddof=1) / np.sqrt(len(ed))) if len(ed) > 2 else np.nan,
                             basket_share_pos=(ed.ew > 0).mean(), basket_sd_bp=ed.ew.std(ddof=1) * 1e4,
                             cap_per_date_mean_m=ed.cap.mean() / 1e6, cap_per_date_median_m=ed.cap.median() / 1e6,
                             pnl_per_date_mean_m=ed.pnl_ccy.mean() / 1e6, pnl_per_date_median_m=ed.pnl_ccy.median() / 1e6,
                             pnl_per_date_t=t_pnl, dates_pnl_pos=(ed.pnl_ccy > 0).mean(),
                             dollar_weighted_bp=ed.pnl_ccy.sum() / ed.cap.sum() * 1e4 if ed.cap.sum() > 0 else np.nan,
                             median_flow_adv=np.nanmedian(d.flow_adv), median_D_name_m=np.nanmedian(d.D_ccy) / 1e6))
lp = pd.DataFrame(rows)
lp.to_csv(OUT / "table_lp_pnl.csv", index=False)

# ---------------- pricing table (2016+ and all), buckets of flow/ADV tercile x run-up half within family x side
rows = []
pxd = pd.concat([df, px])
for per in ["2016+", "all"]:
    d0 = pxd if per == "all" else pxd[pxd.period == "2016+"]
    for (fam, grp), d in d0.groupby(["fam", "grp"]):
        d = d.dropna(subset=["lflow", "s_d0"]).copy()
        if len(d) < 60:
            continue
        d["flow_t"] = pd.qcut(d.lflow, 3, labels=["low", "mid", "high"])
        med = d.runup3_s.median()
        d["crowd"] = np.where(d.runup3_s.isna(), "no run-up data", np.where(d.runup3_s > med, "run-up above median", "run-up below median"))
        d["allb"] = "all"
        for keys, b in (list(d.groupby(["allb"])) + list(d.groupby(["flow_t"], observed=True))
                        + list(d.groupby(["allb", "crowd"])) + list(d.groupby(["flow_t", "crowd"], observed=True))):
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
pt = pd.DataFrame(rows)
pt.to_csv(OUT / "table_pricing.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
x = lp[(lp.basis == "matched control") & (lp.period != "pre-2016")]
print(x[["family", "group", "period", "exit", "name_n", "name_mean_bp", "name_t", "name_median_bp", "n_event_dates", "basket_mean_bp",
         "basket_t", "basket_share_pos", "cap_per_date_median_m", "pnl_per_date_mean_m", "pnl_per_date_t", "dollar_weighted_bp"]].round(2).to_string(index=False))
print(pt[(pt.crowding == "all")][["period", "family", "group", "flow_tercile", "n", "flow_adv_median", "d0_mean_bp", "d0_t", "rev_r1_mean_bp",
                                  "rev_r1_5_mean_bp", "rev_r1_5_sd_name_bp", "rev_r1_5_sd_basket_bp", "quote_name_bp_hold5"]].round(1).to_string(index=False))
