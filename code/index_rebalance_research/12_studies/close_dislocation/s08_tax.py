"""Step 8: Europe by tax regime, and the UK close test (STOXX Europe 600 review events).

Regime by listing country (home exchange): UK (0.5% SDRT/stamp duty), FTT markets FR/IT/ES, other taxed (IE 1%,
BE TOB), untaxed (DE, NL, Nordics, CH, AT, PT, PL, ...). FTT start dates (FR Aug 2012, IT Mar 2013, ES Jan 2021) are
applied: an FR/IT/ES event before its tax started is counted as untaxed in `regime_dated`.
UK split (all UK names are LSE main-market lines, Compustat exchg 194):
  UK-incorporated        Compustat fic = GBR -> SDRT applies to every buy, unless the buyer is an s.88A intermediary
  Irish-incorporated     fic = IRL -> no UK SDRT, but Irish 1% stamp duty on Irish-registered shares
  other foreign-inc.     fic in JEY/GGY/IMN/BMU/GIB/NLD/KAZ/RUS/CHE/FRA... -> no UK SDRT unless the shares sit on a UK
                         register (not checked name by name); Compustat fic is the CURRENT incorporation
  AIM                    names on the FTSE AIM All-Share list at the month-end before T (SDRT-exempt since Apr 2014)
Concurrent FTSE UK review: FTSE 100/250 membership (Datastream month-end lists) differs between the month-end before
and after the review month -> the same close carries FTSE tracker flow.
Output: table_tax.csv
"""
import numpy as np
import pandas as pd
from common import OUT, summ
from s06_tables import load

LEGS = ["pre5", "d0", "d0_id", "on1", "id1", "r1", "r1_5", "r1_20"]
FTT_START = {"FRA": "2012-08-01", "ITA": "2013-03-01", "ESP": "2021-01-16"}

df = load()
st = df[df.family == "STOXX"].copy()
st["regime_dated"] = st.regime
for c, d0 in FTT_START.items():
    m = (st.cty == c) & (pd.to_datetime(st.event_date) < pd.Timestamp(d0))
    st.loc[m, "regime_dated"] = "untaxed"
uk = st.cty == "GBR"
st["uk_inc"] = np.where(~uk, "", np.where(st.fic == "GBR", "UK-incorporated (SDRT)",
                                          np.where(st.fic == "IRL", "Irish-incorporated (IE 1% duty)", "other foreign-incorporated (no SDRT)")))
st.loc[uk & (st.ftse_pre == "AIM"), "uk_inc"] = "AIM (SDRT-exempt)"
st["ftse_change"] = np.where(uk, np.where(st.ftse_pre != st.ftse_post, "FTSE 100/250 change same review", "no FTSE change"), "")
st["ftse_move"] = np.where(uk, st.ftse_pre + "->" + st.ftse_post, "")

rows = []


def add(split, sub):
    for (lab, grp, per), d in pd.concat([sub, sub.assign(period="all")]).groupby(["lab", "grp", "period"]):
        for l in LEGS:
            rows.append(dict(split=split, cell=lab, group=grp, period=per, leg=l, **summ(d[f"s_{l}"].values, d.review.values)))


add("regime (listing country, tax start dates applied)", st.assign(lab=st.regime_dated))
add("UK by incorporation", st[uk].assign(lab=st.loc[uk, "uk_inc"]))
add("UK by concurrent FTSE review", st[uk].assign(lab=st.loc[uk, "ftse_change"]))
ukinc = uk & (st.fic == "GBR")
add("UK-incorporated by concurrent FTSE review", st[ukinc].assign(lab=st.loc[ukinc, "ftse_change"]))
# formal differences: UK-incorporated minus other foreign-incorporated (no SDRT), and UK minus untaxed; t clustered by review
from common import ols_cluster
for lab, a, b in [("UK-incorporated minus foreign-incorporated LSE lines", st[uk & (st.fic == "GBR")], st[uk & ~st.fic.isin(["GBR", "IRL"])]),
                  ("UK minus untaxed listing countries", st[uk], st[st.regime_dated == "untaxed"]),
                  ("FTT (FR/IT/ES) minus untaxed", st[st.regime_dated == "FTT (FR/IT/ES)"], st[st.regime_dated == "untaxed"])]:
    for grp in ["add", "delete", "both"]:
        aa = a if grp == "both" else a[a.grp == grp]
        bb = b if grp == "both" else b[b.grp == grp]
        z = pd.concat([aa.assign(dd=1.0), bb.assign(dd=0.0)])
        for l in LEGS:
            X = np.column_stack([np.ones(len(z)), z.dd.values] + ([(z.side < 0).astype(float).values] if grp == "both" else []))
            res, (n, G, r2) = ols_cluster(z[f"s_{l}"].values * 1e4, X, z.review.values,
                                          ["const", "diff"] + (["sell"] if grp == "both" else []))
            if res is None:
                continue
            r = res.set_index("term").loc["diff"]
            rows.append(dict(split="difference test", cell=lab, group=grp, period="all", leg=l, n=n, n_dates=G,
                             mean_bp=r.coef, t=r.t))
t = pd.DataFrame(rows)
t.to_csv(OUT / "table_tax.csv", index=False)
cnt = st[uk].groupby(["grp", "uk_inc"]).size().unstack(0)
print(cnt)
print(st[uk].groupby(["grp", "ftse_move"]).size().unstack(0).fillna(0).astype(int))
pd.set_option("display.width", 250)
x = t[(t.period == "all") & t.leg.isin(["d0", "on1", "r1", "r1_5", "r1_20"])]
x = x.assign(v=x.mean_bp.round(0).astype("Int64").astype(str) + " (" + x.t.round(1).astype(str) + ") n" + x.n.astype(str))
print(x.pivot_table(index=["split", "cell", "group"], columns="leg", values="v", aggfunc="first").to_string())
