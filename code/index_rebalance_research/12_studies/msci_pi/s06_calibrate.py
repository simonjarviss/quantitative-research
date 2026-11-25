"""Step 6: visible passive demand per line at every MSCI review, and the true-to-visible multiplier R.

Visible demand (shares, fund basis = T-1 basis):
    D_vis = pi_std(T) x dF_std + pi_sc(T) x dF_sc
pi_std = holders' pi of the line's Standard family (MSCI_CANADA; MSCI_EUROPE_<country>), pi_sc = MSCI_CANADA_SC /
MSCI_EUROPE_SC, both interpolated to T (s05). For a migration the two terms have opposite signs and the IMI funds
cancel; for a direct add/delete only one term is non-zero.
R is the slope of excess trade-day volume on visible demand, both per ADV60 (as net_flow/s08):
    (V_T - ADV60) / ADV60 = a_T + R |D_vis| / ADV60 + b1 log mcap + b2 log(ADV60 / shares out) + price buckets + e
event-date fixed effects a_T, y and x winsorised at 1% / 99%, SEs clustered by event date. Volume: Canada consolidated
(Datastream 295), Europe primary venue only (understates by about 25%, so European R is a lower bound).
Samples: all lines of the review (events, FIF/NOS updates, unchanged members); events only; event-type slopes (direct
Standard adds/deletes, Standard <-> Small Cap migrations, Small Cap adds/deletes, weight changes) in one regression;
windows 2018-2021 / 2022-2026; the audit-style median of (V_T - median V_{T-20..T-1}) / |D_vis| for adds.
R includes liquidity-provider and pre-positioner round trips and benchmarked-active trading, so R x pi_visible is an
UPPER bound on true passive ownership.
Outputs: demand.csv (line level), table_R.csv, table_R_by_type.csv.
"""
import numpy as np
import pandas as pd

from common import DATA, OUT, fe_ols, winsor
from s05_pi_series import pi_at

CC = {"GB": "GB", "FR": "FR", "DE": "DE", "CH": "CH", "NL": "NL", "SE": "SE", "ES": "ES", "IT": "IT", "DK": "DK",
      "BE": "BE", "NO": "NO", "AT": "AT", "IE": "XE", "FI": "XE", "PT": "XE", "LU": "XE"}
EVENTS = ["ADD_STD", "DEL_STD", "MIG_UP", "MIG_DOWN", "ADD_SC", "DEL_SC"]
STD_EVENTS = ["ADD_STD", "DEL_STD", "MIG_UP", "MIG_DOWN"]
TYPE = {"ADD_STD": "direct_std", "DEL_STD": "direct_std", "MIG_UP": "migration", "MIG_DOWN": "migration",
        "ADD_SC": "direct_sc", "DEL_SC": "direct_sc", "W_STD": "weight", "W_SC": "weight", "U_STD": "weight",
        "U_SC": "weight"}
WINDOWS = [(2018, 2021), (2022, 2026)]


def demand():
    E = pd.read_csv(OUT / "events.csv", parse_dates=["T"])
    S = pd.read_parquet(DATA / "pi_series.parquet")
    E["fam_std"] = np.where(E.mkt == "CA", "MSCI_CANADA", "MSCI_EUROPE_" + E.ctry.map(CC).fillna("XE"))
    E["fam_sc"] = np.where(E.mkt == "CA", "MSCI_CANADA_SC", "MSCI_EUROPE_SC")
    cache = {}

    def pa(f, d):
        k = (f, d)
        if k not in cache:
            cache[k] = pi_at(S, f, d)
        return cache[k]
    E["pi_std"] = [pa(f, d) for f, d in zip(E.fam_std, E["T"])]
    E["pi_sc"] = [pa(f, d) for f, d in zip(E.fam_sc, E["T"])]
    E["D_std"] = E.pi_std * E.dF_std.fillna(0.0)
    E["D_sc"] = E.pi_sc * E.dF_sc.fillna(0.0)
    E["D_vis"] = E.D_std + E.D_sc
    E["type"] = E.event.map(TYPE).fillna("none")
    E.to_csv(OUT / "demand.csv", index=False)
    return E


def prep(d, p=0.01):
    d = d[(d.adv60 > 0) & d.vol_T.notna() & (d.shares_out > 0) & (d.price_Tm1 > 0) & (d.type != "none")].copy()
    d["y"] = winsor((d.vol_T - d.adv60) / d.adv60, p)
    x = (d.D_vis.abs() / d.adv60).values
    d["x"] = np.clip(x, 0, np.nanquantile(x, 1 - p))
    fx = d.fx_eur.fillna(1.0) if "fx_eur" in d else 1.0
    d["lmcap"] = np.log(d.price_Tm1 * d.shares_out * fx)
    d["lturn"] = np.log(d.adv60 / d.shares_out)
    d["pb_lo"] = (d.price_Tm1 < 5).astype(float)
    d["pb_mid"] = ((d.price_Tm1 >= 5) & (d.price_Tm1 < 20)).astype(float)
    d["Tk"] = d["T"].dt.strftime("%Y-%m-%d") + d.mkt
    return d


CTRL = ["lmcap", "lturn", "pb_lo", "pb_mid"]


def fit(d, xcols, label, fam, window="", basis=""):
    if d.Tk.nunique() < 3 or len(d) < 30:
        return []
    tab, (n, G, r2w) = fe_ols(d.y.values, d[xcols + CTRL].values, d.Tk.values, d.Tk.values, xcols + CTRL)
    if tab is None:
        return []
    out = []
    for c in xcols:
        t = tab.set_index("term").loc[c]
        out.append(dict(sample=label, family=fam, term=c, R=t.coef, se=t.se, t=t.t, n=n, dates=G, r2_within=r2w,
                        window=window, basis=basis, n_events=int(d.event.isin(EVENTS).sum())))
    return out


def med_ratio(d, label, fam):
    x = d[(d.D_vis.abs() > 0) & d.med20.notna()]
    r = (x.vol_T - x.med20) / x.D_vis.abs()
    if len(r) < 3:
        return []
    return [dict(sample=label, family=fam, term="median (V_T - med20)/|D_vis|", R=float(np.median(r)), se=np.nan,
                 t=np.nan, n=len(r), dates=x["T"].nunique(), r2_within=np.nan, window="", basis="",
                 iqr=f"{np.quantile(r, .25):.2f}-{np.quantile(r, .75):.2f}", n_events=len(r))]


def main():
    E = demand()
    rows, rows_t = [], []
    for mkt, fam, basis in (("CA", "CA", "consolidated (Datastream 295)"), ("EU", "EU", "primary venue only (understates ~25%)")):
        Em = E[(E.mkt == mkt) & (E["T"] <= "2026-06-30")]
        # MAIN: Standard index changes (direct adds/deletes and Standard <-> Small Cap migrations) against the
        # review's unchanged Standard members (the date FE baseline). Weight-change lines are left out (their measured
        # demand is small and noisy) and so are Small Cap events (WSML is sampled, so its 'adds/deletes' mix index
        # changes with sampling trades). Event lines are ~1% of the sample: y and x winsorised at 0.2% / 99.8% only.
        m = prep(Em[Em.event.isin(STD_EVENTS + ["U_STD"])], p=0.002)
        rows += fit(m, ["x"], f"{mkt} MAIN: Standard adds/deletes/migrations vs unchanged members", fam, "2018-2026", basis)
        for a, b in WINDOWS:
            x = m[(m["T"].dt.year >= a) & (m["T"].dt.year <= b)]
            rows += fit(x, ["x"], f"{mkt} MAIN: Standard adds/deletes/migrations vs unchanged members, {a}-{b}", fam, f"{a}-{b}", basis)
        for ty in ["direct_std", "migration"]:
            m["x_" + ty] = np.where(m.type == ty, m.x, 0.0)
        rows_t += fit(m, ["x_direct_std", "x_migration"], f"{mkt} MAIN sample, slope by event type", fam, "2018-2026", basis)
        for a, b in WINDOWS:
            x = m[(m["T"].dt.year >= a) & (m["T"].dt.year <= b)]
            rows_t += fit(x, ["x_direct_std", "x_migration"], f"{mkt} MAIN sample, slope by event type, {a}-{b}", fam, f"{a}-{b}", basis)
        sc = prep(Em[Em.event.isin(["ADD_SC", "DEL_SC", "U_SC"])], p=0.002)
        rows_t += fit(sc, ["x"], f"{mkt} Small Cap adds/deletes vs unchanged Small Cap members (lower bound: WSML sampling)",
                      fam, "2018-2026", basis)
        d = prep(Em)
        rows += fit(d, ["x"], f"{mkt} all lines (events + FIF/NOS updates + unchanged), net visible demand", fam, "2018-2026", basis)
        ev = d[d.event.isin(EVENTS)]
        if len(ev) >= 30:
            rows += fit(ev, ["x"], f"{mkt} adds/deletes/migrations only (no baseline lines)", fam, "2018-2026", basis)
        # event-type slopes in one regression
        for ty in ["direct_std", "migration", "direct_sc", "weight"]:
            d["x_" + ty] = np.where(d.type == ty, d.x, 0.0)
        rows_t += fit(d, ["x_direct_std", "x_migration", "x_direct_sc", "x_weight"],
                      f"{mkt} all lines, slope by event type", fam, "2018-2026", basis)
        for a, b in WINDOWS:
            x = d[(d["T"].dt.year >= a) & (d["T"].dt.year <= b)]
            rows += fit(x, ["x"], f"{mkt} all lines, net visible demand, {a}-{b}", fam, f"{a}-{b}", basis)
            xt = x.copy()
            rows_t += fit(xt, ["x_direct_std", "x_migration", "x_direct_sc", "x_weight"],
                          f"{mkt} all lines, slope by event type, {a}-{b}", fam, f"{a}-{b}", basis)
        rows += med_ratio(d[d.event.isin(["ADD_STD", "MIG_UP"])], f"{mkt} Standard entries (audit-style)", fam)
        rows += med_ratio(d[d.event.isin(["DEL_STD", "MIG_DOWN"])], f"{mkt} Standard exits (audit-style)", fam)
        rows += med_ratio(d[d.event == "ADD_SC"], f"{mkt} Small Cap entries from outside the IMI (audit-style)", fam)
        rows += med_ratio(d[d.event.isin(["MIG_UP", "MIG_DOWN"])], f"{mkt} migrations (audit-style)", fam)
        rows += med_ratio(d[d.event.isin(["ADD_STD", "DEL_STD"])], f"{mkt} direct Standard adds/deletes (audit-style)", fam)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_R.csv", index=False)
    Tt = pd.DataFrame(rows_t)
    Tt.to_csv(OUT / "table_R_by_type.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(T[["sample", "term", "R", "se", "t", "n", "dates", "n_events"] + (["iqr"] if "iqr" in T else [])].round(3).to_string())
    print(Tt[["sample", "term", "R", "se", "t", "n", "dates"]].round(3).to_string())


if __name__ == "__main__":
    main()
