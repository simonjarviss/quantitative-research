"""Step 4: signed legs by market x side x split -> events_legs_msci.csv, table_legs.csv, table_diff.csv,
table_by_review.csv.

Signed: s_<leg> = side x abnormal (matched control, Datastream); si_<leg> = side x index-adjusted; sr_<leg> = side x
raw; cs_<leg> = side x abnormal on the Compustat panel (close-to-close legs only). + = the price moved in the
direction of net visible passive demand. Statistics per cell (close_dislocation common.summ): n, reviews, mean (bp),
t clustered by review (CR1), median, 2.5%-each-tail trimmed mean, share > 0, SD.
Splits: all; Canada before / from 18 Oct 2021 (TSX MOC redesign); Europe UK / ex-UK, FTT (FR, IT; ES from 2021) /
other ex-UK, 2018-2021 / 2022-2026; both markets direct adds/deletes vs Standard <-> Small Cap migrations; May/Nov
semi-annual vs Feb/Aug quarterly reviews.
table_diff.csv: pooled OLS of each signed leg on [1, split dummy, sell dummy], SEs clustered by review
(close_dislocation common.ols_cluster): UK minus ex-UK; Oct2021+ minus before (Canada); migrations minus direct.
table_flow.csv: each signed leg on z(log |D_vis| / ADV60), z(log cap), sell dummy (+ UK in Europe), 1% winsorised,
SEs clustered by review; with the correlation of log flow/ADV and log turnover (ADV60 x price / cap).
"""
import numpy as np
import pandas as pd

from common import DATA, LEG_SHOW, OUT, _cd, summ

LEGN = [l for l, _ in _cd.LEGS]
CC = ["d0", "pre5", "r1", "r2_5", "r6_20", "r1_5", "r1_20"]          # close-to-close legs (Compustat cross-check)
FTT_START = {"FR": "2012-08-01", "IT": "2013-03-01", "ES": "2021-01-16"}


def load():
    X = pd.read_parquet(DATA / "legs_msci.parquet")
    X["event_date"] = pd.to_datetime(X.event_date)
    for l in LEGN:
        if f"ab_{l}" in X:
            X[f"s_{l}"] = X.side * X[f"ab_{l}"]
            X[f"si_{l}"] = X.side * X[f"ix_{l}"]
            X[f"sr_{l}"] = X.side * X[f"raw_{l}"]
    for l in CC:
        X[f"cs_{l}"] = X.side * X[f"cs_ab_{l}"]
        X[f"csi_{l}"] = X.side * X[f"cs_ix_{l}"]
    X["fam"] = X.market.map({"CA": "CA MSCI", "EU": "EU MSCI"})
    X["grp"] = X.group
    X["period_ca"] = np.where(X.event_date >= "2021-10-18", "Oct2021+", "pre-Oct2021")
    X["window"] = np.where(X.event_date.dt.year <= 2021, "2018-2021", "2022-2026")
    ftt = np.array([c in FTT_START and d >= pd.Timestamp(FTT_START[c]) for c, d in zip(X.ctry, X.event_date)])
    X["regime"] = np.where(X.market == "CA", "CA", np.where(X.uk, "UK", np.where(ftt, "FTT (FR/IT/ES)", "other ex-UK")))
    X["flow_adv"] = X.dvis_adv_used.abs()            # |visible net demand| / ADV60 (s03: consolidated, else TSX-only)
    X["lflow"] = np.log(X.flow_adv.where(X.flow_adv > 0))
    X["vmult20"] = X.vol_T / X.med20
    return X


def splits(X):
    """(split name, label, mask) list"""
    out = [("all", "all", np.ones(len(X), bool))]
    out += [("period", p, (X.period_ca == p).values) for p in ["pre-Oct2021", "Oct2021+"]]
    out += [("window", w, (X.window == w).values) for w in ["2018-2021", "2022-2026"]]
    out += [("etype", t, (X.etype == t).values) for t in ["direct", "migration"]]
    out += [("uk", "UK", ((X.market == "EU") & X.uk).values), ("uk", "ex-UK", ((X.market == "EU") & ~X.uk).values)]
    out += [("regime", r, (X.regime == r).values) for r in ["FTT (FR/IT/ES)", "other ex-UK"]]
    sair = X.review.str[-2:].isin(["05", "11"]).values
    out += [("review_type", "May/Nov (semi-annual)", sair), ("review_type", "Feb/Aug (quarterly)", ~sair)]
    out += [("uk_period", f"{u} {p}", ((X.market == "EU") & (X.uk == (u == "UK")) & (X.period_ca == p)).values)
            for u in ["UK", "ex-UK"] for p in ["pre-Oct2021", "Oct2021+"]]
    return out


def main():
    X = load()
    X.drop(columns=[c for c in X.columns if c.startswith("cs_raw_")]).to_csv(OUT / "events_legs_msci.csv", index=False)
    rows = []
    for split, lab, m in splits(X):
        for (fam, grp), d in X[m].groupby(["fam", "grp"]):
            for basis, pre, legs in [("matched control", "s_", LEGN), ("index-adjusted", "si_", LEGN), ("raw", "sr_", LEGN),
                                     ("matched control, Compustat", "cs_", CC), ("index-adjusted, Compustat", "csi_", CC)]:
                for l in legs:
                    if pre + l not in d:
                        continue
                    rows.append(dict(basis=basis, family=fam, group=grp, split=split, label=lab, leg=l,
                                     leg_label=dict(_cd.LEGS)[l], **summ(d[pre + l].values, d.review.values)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_legs.csv", index=False)
    # difference tests
    rows = []
    tests = [("UK minus ex-UK", X.market == "EU", X.uk.astype(float)),
             ("Oct2021+ minus pre-Oct2021 (Canada)", X.market == "CA", (X.period_ca == "Oct2021+").astype(float)),
             ("Oct2021+ minus pre-Oct2021 (Europe)", X.market == "EU", (X.period_ca == "Oct2021+").astype(float)),
             ("migration minus direct (Europe)", X.market == "EU", (X.etype == "migration").astype(float)),
             ("migration minus direct (Canada)", X.market == "CA", (X.etype == "migration").astype(float)),
             ("Canada minus Europe", X.market.isin(["CA", "EU"]), (X.market == "CA").astype(float))]
    for name, msk, dummy in tests:
        d = X[msk.values]
        for l in ["d0", "d0_id", "on1", "id1", "r1", "r1_5", "r1_20"]:
            Xm = np.column_stack([np.ones(len(d)), dummy[msk.values].values, (d.side < 0).astype(float).values])
            tab, (n, G, r2) = _cd.ols_cluster(d[f"s_{l}"].values * 1e4, Xm, d.review.values, ["const", "split", "sell"])
            if tab is None:
                continue
            t = tab.set_index("term").loc["split"]
            rows.append(dict(test=name, leg=l, diff_bp=t.coef, t=t.t, n=n, reviews=G))
    pd.DataFrame(rows).to_csv(OUT / "table_diff.csv", index=False)
    diff_rows = rows
    # flow/ADV test (close_dislocation s07 style): signed leg (bp) on z(log |D_vis|/ADV60), z(log cap at T-20), sell
    # dummy (+ UK dummy in Europe), winsorised 1%, SEs clustered by review. corr(lflow, log turnover) reported.
    frows = []
    X["lmcap"] = np.log(X.mcap_f.where(X.mcap_f > 0))
    X["lturn"] = np.log((X.adv60_used * X.price_Tm1 * X.fx_eur.fillna(1.0) / X.mcap_f).where(X.adv60_used > 0))
    for mk, d in X.groupby("market"):
        d = d.dropna(subset=["lflow", "lmcap"])
        z = lambda v: (np.clip(v, *np.nanquantile(v, [0.01, 0.99])) - np.nanmean(v)) / np.nanstd(v)
        cols = [np.ones(len(d)), z(d.lflow.values), z(d.lmcap.values), (d.side < 0).astype(float).values]
        names = ["const", "z_flow_adv", "z_log_cap", "sell"]
        if mk == "EU":
            cols.append(d.uk.astype(float).values); names.append("uk")
        Xm = np.column_stack(cols)
        for l in ["d0", "d0_on", "d0_id", "on1", "r1", "r1_5"]:
            y = d[f"s_{l}"].values * 1e4
            lo, hi = np.nanquantile(y, [0.01, 0.99])
            tab, (n, G, r2) = _cd.ols_cluster(np.clip(y, lo, hi), Xm, d.review.values, names)
            if tab is None:
                continue
            t = tab.set_index("term").loc["z_flow_adv"]
            frows.append(dict(market=mk, leg=l, coef_bp_per_sd=t.coef, t=t.t, n=n, reviews=G, r2=r2,
                              corr_lflow_lturnover=np.corrcoef(d.lflow, d.lturn.fillna(d.lturn.median()))[0, 1]))
    pd.DataFrame(frows).to_csv(OUT / "table_flow.csv", index=False)
    # per review
    rv = X.groupby(["fam", "review", "grp"]).agg(n=("s_d0", "size"), d0_bp=("s_d0", "mean"), on1_bp=("s_on1", "mean"),
                                                 r1_bp=("s_r1", "mean"), r1_5_bp=("s_r1_5", "mean"),
                                                 vmult20_median=("vmult20", "median")).reset_index()
    for c in ["d0_bp", "on1_bp", "r1_bp", "r1_5_bp"]:
        rv[c] *= 1e4
    rv.to_csv(OUT / "table_by_review.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    m = T[(T.basis == "matched control") & T.leg.isin(LEG_SHOW)]
    for split in ["all", "period", "uk", "etype", "regime", "window", "uk_period", "review_type"]:
        x = m[m.split == split]
        if x.empty:
            continue
        pv = x.pivot_table(index=["family", "group", "label"], columns="leg", values="mean_bp")[LEG_SHOW].round(0)
        tv = x.pivot_table(index=["family", "group", "label"], columns="leg", values="t")[LEG_SHOW].round(1)
        nn = x[x.leg == "d0"].set_index(["family", "group", "label"])[["n", "n_dates"]]
        print(f"--- {split}: mean bp"); print(pv.join(nn).to_string())
        print(f"--- {split}: t"); print(tv.to_string())
    print(pd.DataFrame(diff_rows).round(2).to_string())
    print(pd.DataFrame(frows).round(2).to_string())


if __name__ == "__main__":
    main()
