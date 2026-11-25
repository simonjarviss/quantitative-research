"""Step 10: how big are the weight-change flows (share, IWF and capping updates in existing members) next to the
add/delete flows, per review? Notional, %ADV and number of names; visible and true (R x visible) demand.

Per review and component:
  n             names in the component
  gross_vis     sum |D_vis| x price (local currency; Europe in EUR)
  gross_true    the same x R (family R from table_R.csv, net over families)
  med_adv       median |D_true| / ADV60 across the component's names
  n_over_1adv   names whose |D_true| exceeds one day of ADV60
Components:
  CA  add, delete, tsx60 move, weight: share update, weight: IWF / other, weight: capping, unchanged (funding trade)
      plus the placebo (T-2 -> T-1, a no-event day) gross over continuing members, the measurement-noise floor.
  US  Russell recon: adds, deletes, migrations, weight (|delta| > THR), unchanged; S&P 500 quarterly: weight, unchanged.
  EU  per review, net over STOXX + DAX family + FTSE: add, delete, weight, unchanged.
Outputs: table_sizes_by_review.csv, table_sizes_summary.csv (medians across reviews by market x period).
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, period
from s09_flow_test import Rmap


def comp_stats(d, key, val_true, val_vis, adv):
    g = d.groupby(key)
    return pd.DataFrame({"n": g.size(),
                         "gross_vis": g.apply(lambda x: np.nansum(np.abs(x[val_vis]))),
                         "gross_true": g.apply(lambda x: np.nansum(np.abs(x[val_true]))),
                         "med_adv": g.apply(lambda x: np.nanmedian(np.abs(x["_dt"]) / x[adv])),
                         "n_over_1adv": g.apply(lambda x: int(np.nansum(np.abs(x["_dt"]) / x[adv] > 1)))})


def canada(R):
    D = pd.read_csv(OUT / "ca_demand.csv", parse_dates=["T"])
    px = D.price_Tm1.fillna(D.price_TSX).fillna(D.price_TSX60)
    D["_dt"] = R["TSX"] * D.D_vis_TSX_sh + R["TSX60"] * D.D_vis_TSX60_sh
    D["n_true"] = D._dt * px
    D["n_vis"] = D.D_net_sh * px
    D["comp"] = D.event.replace({"tsx60_add": "tsx60 move", "tsx60_delete": "tsx60 move", "add_offcycle": "add (off-cycle)",
                                 "delete_offcycle": "delete (off-cycle)"})
    D.loc[D.event == "weight", "comp"] = "weight: " + D.wc_type
    D.loc[D.event == "unchanged", "comp"] = "unchanged (funding)"
    s = comp_stats(D, ["review", "comp"], "n_true", "n_vis", "adv60").reset_index()
    s["market"] = "CA"; s["mode"] = s.review.map(D.groupby("review").mode_TSX.first())
    pl = pd.read_parquet(DATA / "ca_placebo.parquet")
    pl = pl[(pl.hb_s > 0) & (pl.ha_s > 0)]
    plg = pl.assign(n=pl.notional_pl.abs() * np.where(pl.etf == "XIC", R["TSX"], R["TSX60"])).groupby("review").n.sum()
    pls = pd.DataFrame({"review": plg.index, "comp": "placebo T-2->T-1 (continuing members)", "n": np.nan,
                        "gross_vis": np.nan, "gross_true": plg.values, "market": "CA", "mode": "daily"})
    return pd.concat([s, pls])


def russell(R):
    D = pd.read_csv(OUT / "us_demand.csv", parse_dates=["T"])
    D["_dt"] = R["R2000"] * D.D_vis_R2000 + R["R1000"] * D.D_vis_R1000 + R["R3000"] * D.D_vis_R3000
    D["n_true"] = D._dt * D.price_Tm1
    D["n_vis"] = D.D_net * D.price_Tm1
    D["comp"] = D.event.replace({"BUY_add": "add", "SELL_del": "delete", "BUY_mig": "R1000->R2000", "SELL_mig": "R2000->R1000",
                                 "unchanged": "unchanged (funding)", "OTHER": "other", "other": "other"})
    D["review"] = D.year.astype(str) + "-06"
    s = comp_stats(D, ["review", "comp"], "n_true", "n_vis", "adv60").reset_index()
    s["market"] = "US Russell"; s["mode"] = "month-end (May -> July)"
    Q = pd.read_csv(OUT / "us_sp500_demand.csv", parse_dates=["T"])
    rsp = R["SP500"]
    Q["_dt"] = rsp * Q.D_vis
    Q["n_true"] = Q._dt * Q.price_Tm1
    Q["n_vis"] = Q.D_vis * Q.price_Tm1
    Q["comp"] = Q.cls.replace({"unchanged": "unchanged (funding)"})
    q = comp_stats(Q, ["quarter", "comp"], "n_true", "n_vis", "adv60").reset_index().rename(columns={"quarter": "review"})
    q["market"] = "US S&P 500"; q["mode"] = "month-end"
    return pd.concat([s, q])


def europe(R):
    L = pd.read_parquet(DATA / "legs_net.parquet")
    E = pd.read_csv(OUT / "eu_demand.csv", parse_dates=["T"])
    fams = ["STOXX", "DAX", "MDAX", "TecDAX", "SDAX", "FTSE100", "FTSE250"]
    E["Dt"] = E.D_vis_sh * E.family.map(R)
    pr = {"add": 4, "delete": 3, "weight": 2, "unchanged": 1, "none": 0}
    E["prio"] = E.cls.map(pr)
    E["key"] = np.where(E.col >= 0, E.col.astype(str), "c8:" + E.c8.astype(str))
    g = E.groupby(["T", "key"]).agg(review=("review", "first"), Dt=("Dt", "sum"), Dv=("D_vis_sh", "sum"),
                                    price=("price_Tm1", "first"), fx=("fx_eur", "first"), adv60=("adv60", "first"),
                                    prio=("prio", "max"), p_usd=("p_usd", "first"))
    g["comp"] = g.prio.map({v: k for k, v in pr.items()}).replace({"unchanged": "unchanged (funding)"})
    # EUR notional; unmapped names (no panel price) use the s12 USD price at EUR 0.9 per USD (approximate, flagged)
    eur = g.price * g.fx
    eur = eur.fillna(g.p_usd * 0.9)
    g["n_true"] = g.Dt * eur
    g["n_vis"] = g.Dv * eur
    g["_dt"] = g.Dt
    s = comp_stats(g.reset_index(), ["review", "comp"], "n_true", "n_vis", "adv60").reset_index()
    s["market"] = "EU"; s["mode"] = "quarter-end"
    return s


if __name__ == "__main__":
    R = Rmap()
    S = pd.concat([canada(R), russell(R), europe(R)], ignore_index=True)
    S["year"] = S.review.str[:4].astype(int)
    S["period"] = S.year.map(period)
    S.to_csv(OUT / "table_sizes_by_review.csv", index=False)
    # summary: per market x period x component, median across reviews; and weight / (add + delete) notional ratio
    # Canada summary uses the daily-mode reviews only (month-end pairs carry a month of off-cycle changes)
    S = S[~((S.market == "CA") & S["mode"].isin(["month-end", "nearest"]))]
    tot = S.pivot_table(index=["market", "review", "period"], columns="comp", values="gross_true", aggfunc="sum")
    wcols = [c for c in tot.columns if str(c).startswith("weight")]
    adcols = [c for c in tot.columns if c in ("add", "delete", "R1000->R2000", "R2000->R1000", "tsx60 move")]
    tot["weight_total"] = tot[wcols].sum(1, min_count=1)
    tot["adddel_total"] = tot[adcols].sum(1, min_count=1)
    tot["weight_over_adddel"] = tot.weight_total / tot.adddel_total
    tot.reset_index().to_csv(OUT / "table_sizes_totals_by_review.csv", index=False)
    summ = S.groupby(["market", "period", "comp"]).agg(reviews=("review", "nunique"), n_med=("n", "median"),
                                                        gross_true_med=("gross_true", "median"),
                                                        gross_vis_med=("gross_vis", "median"),
                                                        med_adv_med=("med_adv", "median"),
                                                        n_over_1adv_med=("n_over_1adv", "median")).reset_index()
    summ.to_csv(OUT / "table_sizes_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(summ.round(3).to_string())
    print(tot.groupby(["market", "period"])[["weight_total", "adddel_total", "weight_over_adddel"]].median().to_string())
