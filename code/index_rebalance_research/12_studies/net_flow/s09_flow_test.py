"""Step 9: the flow test. Does bigger event-level passive demand (relative to ADV) move the close more?

Demand: holdings-implied visible demand per line x the family's calibrated R (table_R.csv), summed over the families
trading that close (net): D_true = sum_f R_f D_vis,f. With one R per family this only rescales, except where a line
gets demand from several families (Canada TSX + TSX 60, US R2000 + R1000 + R3000, Europe STOXX + DAX family + FTSE).
Regression (bp per 1 SD of the regressor; regressors winsorised 1%/99% then z-scored within the sample):
    signed leg = a_T + b z(log |D_true| / ADV60) + c1 z(log ADV60 / shares out) + c2 z(log mcap) + price buckets + e
  a_T = event-date fixed effects; SEs clustered by event date. + = the price moved with demand.
  Price buckets: below 5, 5-20, 20+ in local currency. Turnover uses shares outstanding, not our float estimate,
  so log(D/ADV) - log(1/turnover) = log(D/shares out): the test is identified by demand per share outstanding at a
  given turnover (index float/IWF, multi-index membership, size of the share/IWF update).
Samples:
  adds/deletes  the sister study's actual review changes with its matched-control legs (../close_dislocation/
                events_legs.csv), merged to holdings-implied demand: CA by (review, line), US by (year, permno),
                EU by (event date, line).
  weight        existing members whose index float shares changed (|delta| > THR) at the review; legs from s07
                (control = unchanged members of the same size quintile). Canada: daily-mode reviews only (2012-06+).
  tsx60 moves   Composite members moving into / out of the TSX 60 (s07 legs).
Also: "sister flow" = the constant-pi flow/ADV of the sister study on the same events and spec, and a linear test for
weight changes: unsigned abnormal leg (bp) on signed D_true / ADV60 (bp per 1x ADV of demand).
Output: table_flow_test.csv, data/flow_sample.parquet.
"""
import numpy as np
import pandas as pd
from common import DATA, OUT, CD, fe_ols, period

LEGS = ["pre5", "d0", "on1", "id1", "r1", "r1_5", "r1_20"]


def wz(x):
    x = pd.Series(np.asarray(x, float))
    lo, hi = x.quantile([0.01, 0.99])
    x = x.clip(lo, hi)
    return ((x - x.mean()) / x.std()).values


def wins(x, p=0.01):
    x = pd.Series(np.asarray(x, float))
    return x.clip(*x.quantile([p, 1 - p])).values


def Rmap():
    """Full-sample R per family as selected in s08 (table_R_selected.csv)."""
    S = pd.read_csv(OUT / "table_R_selected.csv")
    return {f: (r if np.isfinite(r) and r > 0 else 1.0) for f, r in zip(S.family, S.R)}


def sample():
    R = Rmap()
    L = pd.read_parquet(DATA / "legs_net.parquet")
    E = pd.read_csv(CD / "events_legs.csv", low_memory=False)
    E["event_date"] = pd.to_datetime(E.event_date)
    rows = []
    # ---- Canada
    ca = L[L.market == "CA"].copy()
    ca["D_true"] = R["TSX"] * ca.D_vis_TSX_sh.fillna(0) + R["TSX60"] * ca.D_vis_TSX60_sh.fillna(0)
    ca["grp"] = ca.event.map({"add": "add/delete", "delete": "add/delete", "weight": "weight",
                              "tsx60_add": "tsx60 move", "tsx60_delete": "tsx60 move"})
    e = E[E.market == "CA"].merge(ca[["review", "lid", "D_true", "D_net_sh", "mode_TSX"]], left_on=["review", "id"],
                                 right_on=["review", "lid"], how="inner")
    e = e.assign(grp="add/delete", T=e.event_date, src="sister legs")
    rows.append(e.assign(market="CA"))
    # weight changes and TSX 60 moves: daily-mode reviews only (a month-end pair mixes in changes made on other days)
    w = ca[ca.grp.isin(["weight", "tsx60 move"]) & ca.mode_TSX.astype(str).str.startswith("daily")]
    w = w.assign(src="s07 legs", side=np.sign(w.D_true))
    rows.append(w)
    # ---- US Russell
    us = L[L.market == "US"].copy()
    us["D_true"] = R["R2000"] * us.D_vis_R2000.fillna(0) + R["R1000"] * us.D_vis_R1000.fillna(0) + R["R3000"] * us.D_vis_R3000.fillna(0)
    us["D_net_sh"] = us.D_net
    eu_ = E[E.market == "US"].copy()
    eu_["permno"] = pd.to_numeric(eu_.id, errors="coerce")
    e = eu_.merge(us[["year", "permno", "D_true", "D_net_sh"]], on=["year", "permno"], how="inner")
    rows.append(e.assign(grp="add/delete", T=e.event_date, src="sister legs", market="US"))
    rows.append(us[us.event == "weight"].assign(grp="weight", src="s07 legs", side=lambda d: np.sign(d.D_true)))
    # ---- Europe
    eu = L[L.market == "EU"].copy()
    eu["D_true"] = sum(R[f] * eu[f"D_vis_{f}"].fillna(0) for f in ["STOXX", "DAX", "MDAX", "TecDAX", "SDAX", "FTSE100", "FTSE250"])
    ee = E[E.market == "EU"].merge(eu[["T", "lid", "D_true", "D_net_sh"]], left_on=["event_date", "id"],
                                   right_on=["T", "lid"], how="inner")
    rows.append(ee.assign(grp="add/delete", src="sister legs", market="EU"))
    rows.append(eu[eu.event == "weight"].assign(grp="weight", src="s07 legs", side=lambda d: np.sign(d.D_true)))
    A = pd.concat(rows, ignore_index=True, sort=False)
    A["T"] = pd.to_datetime(A["T"])
    # a name that changes in both the STOXX 600 and the DAX family has two sister rows: keep one per date x line
    A = A.copy()
    kid = A["id"].astype(object).where(A["id"].notna(), A["lid"]) if "id" in A else A["lid"]
    kid = kid.where(kid.notna(), A["permno"].astype("Int64").astype(str) if "permno" in A else np.nan)
    A["key_"] = A.market + "|" + A["T"].astype(str) + "|" + kid.astype(str)
    assert not A.key_.str.endswith("|nan").any() and not A.key_.str.endswith("|<NA>").any()
    A = A.drop_duplicates(["key_", "grp"]).copy()
    A["year_"] = A["T"].dt.year
    A["period"] = A.year_.map(period)
    A["lflow"] = np.log(A.D_true.abs() / A.adv60)
    A["lturn"] = np.log(A.adv60 / A.shares_out)
    A["lmcap_"] = np.log(A.price_Tm1 * A.shares_out) if "price_Tm1" in A else np.nan
    # sister rows carry price / shares under their own names
    A["price_"] = A.price_Tm1.fillna(A.get("price"))
    A["shares_"] = A.shares_out.fillna(A.get("shares"))
    A["lturn"] = np.log(A.adv60 / A.shares_)
    A["lmcap_"] = np.log(A.price_ * A.shares_)
    A["lflow"] = np.log(A.D_true.abs() / A.adv60)
    A["side_"] = np.sign(A.D_true)            # direction of net holdings-implied demand (all groups)
    A["pb_lo"] = (A.price_ < 5).astype(float)
    A["pb_mid"] = ((A.price_ >= 5) & (A.price_ < 20)).astype(float)
    # sign check: holdings demand vs sister side for adds/deletes
    m = A.grp == "add/delete"
    A["sign_agree"] = np.where(m, (np.sign(A.D_true) == A.side).astype(float), np.nan)
    for k in LEGS:
        A[f"y_{k}"] = A[f"ab_{k}"] * A.side_ * 1e4      # + = moved with net holdings-implied demand
    A.to_parquet(DATA / "flow_sample.parquet", index=False)
    return A


def run(A):
    out = []
    mk = {"CA": "CA S&P/TSX", "US": "US Russell", "EU": "EU STOXX 600 + DAX family (+FTSE)"}
    for (mkt, grp), d0 in A.groupby(["market", "grp"]):
        for side_lab, sd in (("all", None), ("buy", 1), ("sell", -1)):
            for per in ("all", "pre-2016", "2016+"):
                d = d0.copy()
                if sd is not None:
                    d = d[d.side_ == sd]
                if per != "all":
                    d = d[d.period == per]
                d = d[np.isfinite(d.lflow) & np.isfinite(d.lturn) & np.isfinite(d.lmcap_)]
                if len(d) < 40 or d["T"].nunique() < 4:
                    continue
                cf = float(np.corrcoef(wins(d.lflow), wins(d.lturn))[0, 1])
                for leg in LEGS:
                    dd = d[np.isfinite(d[f"y_{leg}"])]
                    if len(dd) < 40:
                        continue
                    y = wins(dd[f"y_{leg}"].values)
                    base = {"market": mk[mkt], "group": grp, "side": side_lab, "period": per, "leg": leg}
                    specs = {"flow + turnover + size + price, date FE": ["lflow", "lturn", "lmcap_", "pb_lo", "pb_mid"],
                             "flow + size, date FE (sister spec)": ["lflow", "lmcap_"]}
                    for sname, cols in specs.items():
                        X = np.column_stack([wz(dd[c]) if c in ("lflow", "lturn", "lmcap_") else dd[c].values for c in cols])
                        if side_lab == "all":
                            X = np.column_stack([X, (dd.side_ < 0).astype(float).values])
                        tab, (n, G, r2) = fe_ols(y, X, dd["T"].astype(str).values, dd["T"].astype(str).values,
                                                 cols + (["sell"] if side_lab == "all" else []))
                        if tab is None:
                            continue
                        t = tab.set_index("term")
                        out.append(base | dict(spec=sname, coef_flow=t.loc["lflow", "coef"], t_flow=t.loc["lflow", "t"],
                                               coef_turn=t.loc["lturn", "coef"] if "lturn" in t.index else np.nan,
                                               t_turn=t.loc["lturn", "t"] if "lturn" in t.index else np.nan,
                                               n=n, dates=G, corr_flow_turn=cf, mean_y=float(np.mean(y))))
                    if grp == "add/delete" and "lflow_sis" in dd:
                        cols = ["lflow_sis", "lturn", "lmcap_", "pb_lo", "pb_mid"]
                        ok = np.isfinite(dd.lflow_sis.values)
                        X = np.column_stack([wz(dd[c][ok]) if c in ("lflow_sis", "lturn", "lmcap_") else dd[c][ok].values for c in cols])
                        if side_lab == "all":
                            X = np.column_stack([X, (dd.side_[ok] < 0).astype(float).values])
                        tab, (n, G, r2) = fe_ols(y[ok], X, dd["T"][ok].astype(str).values, dd["T"][ok].astype(str).values,
                                                 cols + (["sell"] if side_lab == "all" else []))
                        if tab is not None:
                            t = tab.set_index("term")
                            out.append(base | dict(spec="SISTER constant-pi flow + turnover + size + price, date FE",
                                                   coef_flow=t.loc["lflow_sis", "coef"], t_flow=t.loc["lflow_sis", "t"],
                                                   coef_turn=t.loc["lturn", "coef"], t_turn=t.loc["lturn", "t"], n=n, dates=G,
                                                   corr_flow_turn=float(np.corrcoef(wins(dd.lflow_sis[ok]), wins(dd.lturn[ok]))[0, 1]),
                                                   mean_y=float(np.mean(y[ok]))))
                    # linear signed test (weight changes): unsigned abnormal leg on signed D_true / ADV60
                    if grp in ("weight", "tsx60 move"):
                        x = wins(dd.D_true.values / dd.adv60.values)
                        yu = wins(dd[f"ab_{leg}"].values * 1e4)
                        X = np.column_stack([x, wz(dd.lturn), wz(dd.lmcap_), dd.pb_lo.values, dd.pb_mid.values])
                        tab, (n, G, r2) = fe_ols(yu, X, dd["T"].astype(str).values, dd["T"].astype(str).values,
                                                 ["D/ADV", "lturn", "lmcap", "pb_lo", "pb_mid"])
                        if tab is not None:
                            t = tab.set_index("term")
                            out.append(base | dict(spec="LINEAR: unsigned leg on signed D_true/ADV60 (bp per 1x ADV)",
                                                   coef_flow=t.loc["D/ADV", "coef"], t_flow=t.loc["D/ADV", "t"], n=n, dates=G,
                                                   corr_flow_turn=cf, mean_y=float(np.mean(yu))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    A = sample()
    if "lflow" in A:
        # sister constant-pi flow on the same rows (adds/deletes only)
        A["lflow_sis"] = np.where(A.grp == "add/delete", np.log(A.demand_sh / A.adv60), np.nan) if "demand_sh" in A else np.nan
    print(A.groupby(["market", "grp"]).size())
    print("sign agreement holdings vs sister side (adds/deletes):", A[A.grp == "add/delete"].groupby("market").sign_agree.mean().round(3).to_dict())
    T = run(A)
    T.to_csv(OUT / "table_flow_test.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    x = T[(T.leg == "d0") & (T.side == "all")]
    print(x[["market", "group", "period", "spec", "coef_flow", "t_flow", "coef_turn", "t_turn", "n", "dates", "corr_flow_turn"]].round(2).to_string())


def terciles(A):
    """Mean signed leg (bp, t clustered by date) by within-date tercile of log |D_true|/ADV60 and by |D_true|/ADV60
    bucket, per market x group. Descriptive companion to the regressions."""
    from common import clt
    out = []
    A = A[np.isfinite(A.lflow)].copy()
    A["terc"] = A.groupby(["market", "grp", "T"]).lflow.transform(
        lambda s: np.ceil(s.rank(pct=True) * 3).clip(1, 3) if len(s) >= 3 else np.nan)
    A["dadv"] = A.D_true.abs() / A.adv60
    A["bucket"] = pd.cut(A.dadv, [0, 0.1, 0.5, 1, 3, np.inf], labels=["<0.1x", "0.1-0.5x", "0.5-1x", "1-3x", ">3x"])
    for (mkt, grp), d in A.groupby(["market", "grp"]):
        for per in ("all", "pre-2016", "2016+"):
            x = d if per == "all" else d[d.period == per]
            for leg in LEGS:
                for lab, col in (("tercile", "terc"), ("D/ADV bucket", "bucket")):
                    for k, g in x.groupby(col, observed=True):
                        y = g[f"y_{leg}"].values
                        ok = np.isfinite(y)
                        if ok.sum() < 20:
                            continue
                        m, se, t = clt(np.clip(y[ok], *np.nanquantile(y[ok], [0.01, 0.99])), g["T"].astype(str).values[ok])
                        out.append(dict(market=mkt, group=grp, period=per, leg=leg, split=lab, bin=str(k), n=int(ok.sum()),
                                        dates=g["T"][ok].nunique(), mean_bp=m, t=t, med_dadv=float(np.nanmedian(g.dadv))))
    return pd.DataFrame(out)


if __name__ == "__main__":
    A = pd.read_parquet(DATA / "flow_sample.parquet")
    TT = terciles(A)
    TT.to_csv(OUT / "table_flow_terciles.csv", index=False)
    x = TT[(TT.leg == "d0") & (TT.period == "all")]
    print(x[["market", "group", "split", "bin", "n", "dates", "mean_bp", "t", "med_dadv"]].round(2).to_string())


def wctype(A):
    """Canada only: weight changes split into share updates and IWF / other (s04 labels), trade-day leg."""
    from common import clt
    out = []
    w = A[(A.market == "CA") & (A.grp == "weight")]
    for wt, d in w.groupby("wc_type"):
        for per in ("all", "pre-2016", "2016+"):
            dd = d if per == "all" else d[d.period == per]
            dd = dd[np.isfinite(dd.lflow) & np.isfinite(dd.lturn) & np.isfinite(dd.lmcap_) & np.isfinite(dd.y_d0)]
            for leg in ("pre5", "d0", "on1", "r1"):
                y = wins(dd[f"y_{leg}"].values)
                X = np.column_stack([wz(dd.lflow), wz(dd.lturn), wz(dd.lmcap_), dd.pb_lo, dd.pb_mid, (dd.side_ < 0).astype(float)])
                tab, (n, G, r2) = fe_ols(y, X, dd["T"].astype(str).values, dd["T"].astype(str).values,
                                         ["lflow", "lturn", "lmcap", "pb_lo", "pb_mid", "sell"])
                m, se, t = clt(y, dd["T"].astype(str).values)
                out.append(dict(wc_type=wt, period=per, leg=leg, n=n, dates=G, mean_bp=m, t_mean=t,
                                coef_flow=tab.set_index("term").loc["lflow", "coef"], t_flow=tab.set_index("term").loc["lflow", "t"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    W = wctype(pd.read_parquet(DATA / "flow_sample.parquet"))
    W.to_csv(OUT / "table_flow_wctype_ca.csv", index=False)
    print(W[W.leg == "d0"].round(2).to_string())
