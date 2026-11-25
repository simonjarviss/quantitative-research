"""Step 3: point-in-time Russell rank rebuild at H6 (end-Dec), H3 (end-Mar), HR (rank day) and predictions.

Only data dated on or before the horizon date d is used:
  - price, cap, shares: CRSP dsf_v2 on d (company cap = sum over listed common classes by permco)
  - security flags, SIC, float: msf_v2 at the last month-end <= d (fallback: the month of d for names that
    listed after that month-end; flagged as `attr_fallback`)
  - current membership (for banding / member price rule): Datastream list at the last month-end <= d
  - NOT point-in-time: Compustat HQ `loc` (current header only); CRSP shares/float may be restated.
Screens: common, NYSE/Nasdaq/AMEX/Arca/BATS, US (CRSP usincflg or Compustat loc=USA; current members exempt),
structures (SIC 6726/6770/6792/6733, LP/LLC/trust names; current members exempt), float >= 5% (CRSP float/shrout),
price >= $1 (members: 30-day average >= $1 also passes), company cap >= $30m. Rank by cap; R1000 = #1-1000,
R2000 = #1001-3000; existing R3000 members within +/-2.5% cumulative-cap percentile of #1000 keep their index.

Outputs
  _cache/pred_{year}_{H}.parquet   full ranked cross-section with predicted status
  output/predictions.csv          one row per (year, horizon, permco) predicted change (incl. false positives)
  output/prediction_accuracy.csv  precision / recall by year x horizon x group
  output/rank_day_fit.csv         change-F1 of the rebuild on candidate rank days (validates assumed dates)
Usage: python3 s03_predict.py [years...] [--rankfit]
"""
import sys
import numpy as np
import pandas as pd
from common import CACHE, OUT, CAL, R1000, R2000, MAIN_YEARS, EXT_YEARS, snapshots
import lib

MSF = pd.read_parquet(CACHE / "msf.parquet")
MSF["ym"] = MSF.mthcaldt.dt.to_period("M")
MEM = pd.read_parquet(CACHE / "membership.parquet")
MEM["ym"] = pd.PeriodIndex(MEM.ym, freq="M")
COMP = pd.read_parquet(CACHE / "comp_company.parquet")
CCM = pd.read_parquet(CACHE / "ccm.parquet")
EV = pd.read_csv(OUT / "events_actual.csv")
import os
EXCL_MINRANK = int(os.environ.get("EXCL_MINRANK", "1001"))   # flag only names ranked EXCL_MINRANK..2700 last year
STRUCT_RX = r"ROYALTY TR| RTY TR| TRUST$| TR$| L\s?P$|L\.P\.|PARTNERS LP| LLC$"


def status_at(ym) -> pd.Series:
    m = MEM[(MEM.ym == ym) & MEM.permco.notna()]
    s1 = set(m.loc[m.code == R1000, "permco"].astype(int))
    s2 = set(m.loc[m.code == R2000, "permco"].astype(int)) - s1
    return pd.concat([pd.Series("R1000", index=sorted(s1)), pd.Series("R2000", index=sorted(s2))]).rename_axis("permco")


def hq_loc(d) -> pd.DataFrame:
    l = CCM[(CCM.linkdt <= d) & (CCM.linkenddt.fillna(pd.Timestamp("2099-12-31")) >= d)]
    return l[["permno", "gvkey"]].merge(COMP[["gvkey", "loc"]], on="gvkey").drop_duplicates("permno")[["permno", "loc"]]


def rebuild(d, cur: pd.Series, memory=True, excl=frozenset()) -> pd.DataFrame:
    d = pd.Timestamp(d)
    x = lib.daily(d, d, "permno, permco, prc, cap, shrout")
    x = x[~x.permco.isin(set(excl) - set(cur.index))]     # Russell-judged ineligible at the last recon (PIT)
    x = x.dropna(subset=["cap", "prc"])
    x = x[x.cap > 0]
    ym_attr = (d + pd.offsets.MonthEnd(0)).to_period("M") if d == lib.last_tday_on_or_before(d + pd.offsets.MonthEnd(0)) \
        else (d.to_period("M") - 1)
    a = MSF[MSF.ym == ym_attr][["permno", "sharetype", "securitysubtype", "primaryexch", "usincflg", "siccd", "issuernm", "floatsh", "shrout"]]
    a = a.rename(columns={"shrout": "shrout_m"})
    x = x.merge(a, on="permno", how="left")
    miss = x.sharetype.isna()
    if miss.any():   # listed after the attribute month-end: take the flags from the month of d
        b = MSF[MSF.ym == d.to_period("M")][a.columns.str.replace("shrout_m", "shrout")].rename(columns={"shrout": "shrout_m"})
        fill = x.loc[miss, ["permno"]].merge(b, on="permno", how="left")
        for c in a.columns[1:]:
            x.loc[miss, c] = fill[c].values
    x["attr_fallback"] = miss
    x = x.merge(hq_loc(d), on="permno", how="left")
    x["member"] = x.permco.isin(cur.index)
    # security-level screens
    x = x[(x.securitysubtype == "COM") & x.sharetype.isin(["NS", "SB", "UG"]) & x.primaryexch.isin(["N", "Q", "A", "R", "B"])]
    us = (x.usincflg == "Y") | (x["loc"] == "USA")
    struct_bad = x.siccd.isin([6726, 6770, 6792, 6733]) | x.issuernm.str.contains(STRUCT_RX, regex=True, na=False)
    if memory:
        x = x[(us | x.member) & (~struct_bad | x.member)]
    else:
        x = x[us & ~struct_bad]
    x = x[~(x.floatsh.notna() & (x.floatsh < 0.05 * x.shrout_m))]
    # member price rule: 30-calendar-day average close >= $1
    if memory:
        p = lib.daily(d - pd.Timedelta(days=30), d, "permno, prc", f"AND permco IN ({','.join(map(str, cur.index))})")
        avgp = p.groupby("permno").prc.mean()
        x["prc30"] = x.permno.map(avgp)
    else:
        x["prc30"] = np.nan
    x = x.sort_values("cap", ascending=False)
    comp = x.groupby("permco", sort=False).agg(cap=("cap", "sum"), prc=("prc", "max"), prc30=("prc30", "max"),
                                                permno=("permno", "first"), member=("member", "first")).reset_index()
    ok_px = (comp.prc >= 1) | (comp.member & (comp.prc30 >= 1))
    comp = comp[ok_px & (comp.cap >= 30e6)].sort_values("cap", ascending=False).reset_index(drop=True)
    comp["rank"] = np.arange(1, len(comp) + 1)
    top = comp[comp["rank"] <= 4000]
    comp["cumpct"] = np.where(comp["rank"] <= 4000, comp.cap.cumsum() / top.cap.sum() * 100, np.nan)
    bp = comp.loc[comp["rank"] == 1000, "cumpct"].iloc[0]
    lo, hi = bp - 2.5, bp + 2.5
    comp["cur"] = comp.permco.map(cur).fillna("out")
    comp["pred"] = np.where(comp["rank"] <= 1000, "R1000", np.where(comp["rank"] <= 3000, "R2000", "out"))
    inband = comp.cur.isin(["R1000", "R2000"]) & comp.cumpct.between(lo, hi)
    comp.loc[inband, "pred"] = comp.loc[inband, "cur"]
    # current members missing from the eligible ranking (failed price/cap/float or not common): predicted out
    gone = cur[~cur.index.isin(comp.permco)]
    allx = lib.daily(d, d, "permno, permco, prc, cap").dropna(subset=["prc"]).sort_values("cap", ascending=False)
    allx = allx.drop_duplicates("permco").set_index("permco")
    g = pd.DataFrame({"permco": gone.index, "cur": gone.values, "pred": "out"})
    g["permno"] = g.permco.map(allx.permno)
    g["cap"] = g.permco.map(allx.cap)
    g["prc"] = g.permco.map(allx.prc)
    g = g[g.permno.notna()]            # not trading on d -> cannot be predicted / traded
    g["member"] = True
    out = pd.concat([comp, g], ignore_index=True)
    out["bp_cap"] = comp.loc[comp["rank"] == 1000, "cap"].iloc[0]
    out["r3000_cap"] = comp.loc[comp["rank"] == 3000, "cap"].iloc[0]
    out["band_lo"], out["band_hi"] = lo, hi
    return out


def classify(df, a="cur", b="pred"):
    return np.select([(df[a] == "out") & (df[b] == "R2000"), (df[a] == "R1000") & (df[b] == "R2000"),
                      (df[a] == "R2000") & (df[b] == "out"), (df[a] == "R2000") & (df[b] == "R1000"),
                      df[a] != df[b]], ["BUY_add", "BUY_mig", "SELL_del", "SELL_mig", "OTHER"], "none")


def horizon_snapshot(d):
    """Last month-end membership list on or before d."""
    me = d + pd.offsets.MonthEnd(0)
    return me.to_period("M") if d == lib.last_tday_on_or_before(me) else d.to_period("M") - 1


_EXCL = {}


def known_excluded(y) -> frozenset:
    """Companies Russell left out at the PREVIOUS June recon although our rebuild ranked them comfortably
    inside the R3000 (rank <= 2700) on that rank day and they kept trading: treat as Russell-ineligible
    (country / structure / float / voting).  Uses only the y-1 rank-day data and the y-1 final list (public
    by end-June y-1), so it is point-in-time for every horizon of year y."""
    if y in _EXCL:
        return _EXCL[y]
    if y - 1 not in CAL.index and y - 1 != 2007:
        return frozenset()
    rd = CAL.loc[y - 1, "rank_day"] if y - 1 in CAL.index else pd.Timestamp(2007, 5, 31)
    rd = max(lib.last_tday_on_or_before(rd), lib.tdays()[0])
    prev = known_excluded(y - 1) if y - 1 in CAL.index else frozenset()
    r = rebuild(rd, status_at(horizon_snapshot(rd)), excl=prev)
    post = status_at(snapshots(y - 1)[1].to_period("M")) if y - 1 in CAL.index else status_at(pd.Period("2007-06", "M"))
    alive = set(lib.daily(snapshots(y - 1)[1] if y - 1 in CAL.index else pd.Timestamp(2007, 6, 29),
                          snapshots(y - 1)[1] if y - 1 in CAL.index else pd.Timestamp(2007, 6, 29), "permco").permco)
    ex = set(r.loc[(r["rank"] <= 2700) & (r["rank"] >= EXCL_MINRANK) & ~r.permco.isin(post.index) & r.permco.isin(alive), "permco"].astype(int))
    ex |= {p for p in prev if p not in post.index}      # carry forward names still left out
    ex = frozenset(ex)
    _EXCL[y] = ex
    return ex


def run_year(y, memory=True):
    hd = lib.horizon_dates(y)
    ex = known_excluded(y)
    act = EV[(EV.year == y) & EV.valid & (EV.group != "OTHER")]
    preds, acc = [], []
    for h in ["H6", "H3", "HR"]:
        d = hd[h]
        cur = status_at(horizon_snapshot(d))
        r = rebuild(d, cur, memory, excl=ex)
        r["year"], r["horizon"], r["hdate"] = y, h, d
        r["pgroup"] = classify(r)
        r.to_parquet(CACHE / f"pred_{y}_{h}.parquet", index=False)
        preds.append(r[r.pgroup != "none"])
        for g in ["BUY_add", "BUY_mig", "SELL_del", "SELL_mig"]:
            P = set(r.loc[r.pgroup == g, "permco"]); A = set(act.loc[act.group == g, "permco"])
            vis = A & set(lib.daily(d, d, "permco").permco)
            hit = len(P & A)
            acc.append(dict(year=y, horizon=h, group=g, n_pred=len(P), n_actual=len(A), n_actual_listed_at_h=len(vis),
                            n_hit=hit, precision=hit / len(P) if P else np.nan, recall=hit / len(A) if A else np.nan,
                            recall_listed=len(P & vis) / len(vis) if vis else np.nan))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(acc)


def change_f1(r, y):
    act = EV[(EV.year == y) & EV.valid]
    A = set(zip(act.permco, act.post))
    r = r[r.cur != r.pred]
    P = set(zip(r.permco, r.pred))
    hit = len(A & P)
    return 2 * hit / (len(A) + len(P)) if (A or P) else np.nan, len(P), len(A)


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:] if a.isdigit()] or MAIN_YEARS + EXT_YEARS
    if "--rankfit" in sys.argv:
        out = []
        for y in yrs:
            cands = {"cal": CAL.loc[y, "rank_day"], "endApr": pd.Timestamp(y, 4, 30), "endMay": pd.Timestamp(y, 5, 31)}
            for k in (-10, -5, 5, 10):
                cands[f"cal{k:+d}td"] = lib.tshift(CAL.loc[y, "rank_day"], k)
            for k, dd in cands.items():
                dd = lib.last_tday_on_or_before(dd)
                r = rebuild(dd, status_at(horizon_snapshot(dd)), excl=known_excluded(y))
                f1, npred, nact = change_f1(r, y)
                out.append(dict(year=y, candidate=k, date=dd.date(), change_f1=round(f1, 4), n_pred_changes=npred, n_actual_changes=nact))
                print(out[-1], flush=True)
        pd.DataFrame(out).to_csv(OUT / "rank_day_fit.csv", index=False)
        sys.exit()
    P, A = [], []
    for y in yrs:
        p, a = run_year(y)
        P.append(p); A.append(a)
        print(y, a[["horizon", "group", "n_pred", "n_actual", "n_hit", "precision", "recall"]].round(2).to_string(index=False), flush=True)
    keep = ["year", "horizon", "hdate", "permco", "permno", "cur", "pred", "pgroup", "rank", "cap", "prc", "cumpct", "member"]
    pd.concat(P)[keep].to_csv(OUT / "predictions.csv", index=False)
    pd.concat(A).to_csv(OUT / "prediction_accuracy.csv", index=False)
