"""Step 6: combine the three markets into events_legs.csv and tabulate every leg by market x side x period.

Signed returns: s_<leg> = side x abnormal (matched-control) return; si_<leg> = side x index-adjusted return;
sr_<leg> = side x raw return. Positive = the price moved in the direction of index demand.
Statistics per cell: n, event dates, mean (bp), t clustered by event date (CR1), median, 2.5%-each-tail trimmed mean,
share > 0, SD. US event dates = one per year, so its t uses 8-10 clusters per period: read with care.
Outputs: events_legs.csv, table_legs.csv (all bases), table_dax_moves.csv
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, LEGS, summ, period

LEGN = [l for l, _ in LEGS]


def load():
    us = pd.read_parquet(DATA / "legs_us.parquet")
    ca = pd.read_parquet(DATA / "legs_ca.parquet")
    eu = pd.read_parquet(DATA / "legs_eu.parquet")
    df = pd.concat([us, ca, eu], ignore_index=True, sort=False)
    df = df.copy()
    df["period"] = df.year.map(period)
    for l in LEGN:
        df[f"s_{l}"] = df.side * df[f"ab_{l}"]
        df[f"si_{l}"] = df.side * df[f"ix_{l}"]
        df[f"sr_{l}"] = df.side * df[f"raw_{l}"]
    df = df.copy()
    df["flow_adv"] = df.demand_sh / df.adv60
    df["lflow"] = np.log(df.flow_adv.where(df.flow_adv > 0))
    df["flow_adv_base"] = df.demand_sh / df.adv_base       # same demand over the months -12..-6 baseline ADV
    df["lflow_b"] = np.log(df.flow_adv_base.where(df.flow_adv_base > 0))
    df["vmult"] = df.vol_T / df.adv60                    # realised T volume / ADV60
    df["vmult20"] = df.vol_T / df.med20                  # sister-study multiple (vs median of T-20..T-1)
    df["lvmult"] = np.log(df.vmult.where(df.vmult > 0))
    df["runup3_s"] = df.side * df.runup_h3
    df["runup6_s"] = df.side * df.runup_h6
    ccy = np.where(df.market == "US", 1.0, np.where(df.market == "CA", 1.0, df.get("fx_eur", pd.Series(1.0, index=df.index)).fillna(1.0)))
    df["demand_ccy"] = df.demand_sh * df.price * ccy     # USD (US), CAD (CA), EUR (EU)
    df["mcap_ccy"] = df.shares * df.price * ccy
    df["lmcap"] = np.log(df.mcap_ccy.where(df.mcap_ccy > 0))
    df["adv60_ccy"] = df.adv60 * df.price * ccy
    df["ccy"] = df.market.map({"US": "USD", "CA": "CAD", "EU": "EUR"})
    lab = {"BUY_add": "add", "SELL_del": "delete", "BUY_mig": "R1000->R2000", "SELL_mig": "R2000->R1000"}
    df["grp"] = df.group.replace(lab)
    dax = df.family == "DAX"
    df.loc[dax, "grp"] = np.where(df.loc[dax, "side"] > 0, "net buy", "net sell")
    df["fam"] = df.family.replace({"R2000": "US Russell", "TSX": "CA S&P/TSX", "STOXX": "EU STOXX 600", "DAX": "EU DAX family"})
    return df


if __name__ == "__main__":
    df = load()
    df.to_csv(OUT / "events_legs.csv", index=False)
    rows = []
    us = df[df.market == "US"]
    px = us.assign(grp=us.grp + np.where(us.price >= 5, " | price >= $5", " | price < $5"))
    for basis, pre in [("matched control", "s_"), ("index-adjusted", "si_"), ("raw", "sr_")]:
        for (fam, grp, per), d in pd.concat([df, df.assign(period="all"), px, px.assign(period="all")]).groupby(["fam", "grp", "period"]):
            for l in LEGN:
                rows.append(dict(basis=basis, family=fam, group=grp, period=per, leg=l, leg_label=dict(LEGS)[l],
                                 **summ(d[pre + l].values, d.review.values)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "table_legs.csv", index=False)
    # DAX family by move type (net passive demand sign in brackets)
    dx = df[df.family == "DAX"]
    rows = []
    for (mv, per), d in pd.concat([dx, dx.assign(period="all")]).groupby(["move", "period"]):
        for l in ["d0", "on1", "r1", "r1_5", "r1_20"]:
            s = summ(d[f"ab_{l}"].values, d.review.values)          # UNSIGNED: + = price rose vs controls
            rows.append(dict(move=mv, period=per, pi_net_pct=d.pi_signed.mean() * 100, leg=l, **s))
    pd.DataFrame(rows).to_csv(OUT / "table_dax_moves.csv", index=False)
    pd.set_option("display.width", 250)
    m = t[(t.basis == "matched control") & t.leg.isin(["d0", "on1", "id1", "r1", "r1_5", "r1_20"])]
    print(m.pivot_table(index=["family", "group", "period"], columns="leg", values="mean_bp").round(0).to_string())
    print(m.pivot_table(index=["family", "group", "period"], columns="leg", values="t").round(1).to_string())
