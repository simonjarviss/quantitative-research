"""Step 10: checks. Writes table_checks.csv (one row per check) and table_checks_volume_by_date.csv.

1. Headline numbers recomputed from events_legs.csv with independent code (plain pandas; t clustered by event date).
2. Event dates against volume spikes: median V_T / median(V T-20..T-1) per event date; dates below 1.5x are flagged.
3. Opens: share of events with a usable T and T+1 open; T+1 opens equal to the T close; EU source (Datastream vs
   Compustat) and agreement of the two where both exist (2018+).
4. Stale or zero returns: share of exact-zero trade-day returns; headline without them.
5. Outliers: headline trimmed (2.5% each tail), winsorised (1%/99%), and dropping the single most extreme event date.
6. Penny stocks (US): headline without names below $5 at T-1; overnight leg without T+1 opens equal to the T close.
"""
import numpy as np
import pandas as pd
from common import OUT, DATA

E = pd.read_csv(OUT / "events_legs.csv", low_memory=False)
rows = []


def ct(x, g):
    x = pd.Series(np.asarray(x, float)); g = pd.Series(np.asarray(g))
    ok = x.notna(); x, g = x[ok].reset_index(drop=True), g[ok].reset_index(drop=True)
    n, G = len(x), g.nunique()
    if n < 3 or G < 2:
        return x.mean(), np.nan, n
    e = (x - x.mean()).groupby(g).sum()
    se = np.sqrt(G / (G - 1) * (e ** 2).sum()) / n
    return x.mean(), x.mean() / se, n


def add(check, fam, grp, per, leg, val, t=np.nan, n=np.nan, note=""):
    rows.append(dict(check=check, family=fam, group=grp, period=per, leg=leg, value=val, t=t, n=n, note=note))


HEAD = [("US Russell", "delete"), ("US Russell", "add"), ("US Russell", "R2000->R1000"), ("US Russell", "R1000->R2000"),
        ("CA S&P/TSX", "add"), ("CA S&P/TSX", "delete"), ("EU STOXX 600", "add"), ("EU STOXX 600", "delete"),
        ("EU DAX family", "net buy"), ("EU DAX family", "net sell")]
for fam, grp in HEAD:
    for per in ["all", "pre-2016", "2016+"]:
        d = E[(E.fam == fam) & (E.grp == grp)]
        if per != "all":
            d = d[d.period == per]
        for leg in ["d0", "on1", "id1", "r1", "r1_5", "r1_20"]:
            m, t, n = ct(d[f"s_{leg}"] * 1e4, d.review)
            add("1 recomputed mean (bp)", fam, grp, per, leg, m, t, n)
            x = d[f"s_{leg}"].dropna()
            lo, hi = x.quantile([0.01, 0.99])
            m, t, n = ct(x.clip(lo, hi) * 1e4, d.loc[x.index, "review"])
            add("5 winsorised 1/99 mean (bp)", fam, grp, per, leg, m, t, n)
            if leg in ["d0", "on1", "r1_20"]:
                by = d.groupby("review")[f"s_{leg}"].mean()
                if len(by) > 3:
                    worst = (by - by.mean()).abs().idxmax()
                    dd = d[d.review != worst]
                    m, t, n = ct(dd[f"s_{leg}"] * 1e4, dd.review)
                    add("5 drop most extreme event date (bp)", fam, grp, per, leg, m, t, n, note=f"dropped {worst}")
        dz = d[~d.d0_ret_is_zero.astype(bool)]
        m, t, n = ct(dz.s_d0 * 1e4, dz.review)
        add("4 trade day without exact-zero T returns (bp)", fam, grp, per, "d0", m, t, n,
            note=f"zero-return share {d.d0_ret_is_zero.astype(bool).mean():.3f}")
        m, t, n = ct(d.s_d0_on * 1e4 + d.s_d0_id * 1e4, d.review)
        add("3 trade day = overnight + intraday pieces (bp, approx.)", fam, grp, per, "d0_on+d0_id", m, t, n)
        if "open_T1_eqprev" in d:
            dd = d[~d.open_T1_eqprev.fillna(False).astype(bool)]
            m, t, n = ct(dd.s_on1 * 1e4, dd.review)
            add("6 overnight without T+1 open = T close (bp)", fam, grp, per, "on1", m, t, n,
                note=f"flag share {d.open_T1_eqprev.fillna(False).astype(bool).mean():.3f}")
        if fam == "US Russell":
            dd = d[d.price >= 5]
            for leg in ["d0", "on1", "r1_20"]:
                m, t, n = ct(dd[f"s_{leg}"] * 1e4, dd.review)
                add("6 price >= $5 at T-1 (bp)", fam, grp, per, leg, m, t, n)
        add("3 usable T+1 open (share)", fam, grp, per, "on1", d.open_T1_ok.astype(bool).mean(), n=len(d))
        add("3 usable T open (share)", fam, grp, per, "d0_on", d.open_T_ok.astype(bool).mean(), n=len(d))

# 2. volume spike per event date
vb = E.assign(m=E.vol_T / E.med20).groupby(["fam", "review", "event_date"]).agg(n=("m", "size"), median_mult=("m", "median"),
                                                                                share_ge2=("m", lambda v: (v >= 2).mean())).reset_index()
vb["flag_below_1p5"] = vb.median_mult < 1.5
vb.to_csv(OUT / "table_checks_volume_by_date.csv", index=False)
for fam, d in vb.groupby("fam"):
    add("2 event dates with median V_T/median(V_T-20..T-1) < 1.5", fam, "all", "all", "vol", int(d.flag_below_1p5.sum()), n=len(d),
        note="; ".join(f"{r.review}:{r.median_mult:.2f}" for r in d[d.flag_below_1p5].itertuples()))
    add("2 median of event-date median volume multiple", fam, "all", "all", "vol", d.median_mult.median(), n=len(d))

# 3. EU open sources and agreement
eu = E[E.market == "EU"]
add("3 EU T+1 open from Datastream primary quote (share)", "EU", "all", "all", "on1", (eu.open_src_T1 == 1).mean(), n=len(eu))
add("3 EU T+1 open from Compustat g_secd fallback (share)", "EU", "all", "all", "on1", (eu.open_src_T1 == 2).mean(), n=len(eu))
both = eu.dropna(subset=["g_open_T1", "ds_open_T1"])
if len(both):
    r = both.g_open_T1 / both.ds_open_T1
    r = np.where(r > 50, r / 100, np.where(r < 0.02, r * 100, r))          # GBp vs GBP quotes
    add("3 EU Datastream vs Compustat T+1 open agree within 0.5% (share, 2018+)", "EU", "all", "all", "on1",
        float(np.mean(np.abs(r - 1) < 0.005)), n=len(both))
    rc = both.g_close_T1 / both.ds_close_T1
    rc = np.where(rc > 50, rc / 100, np.where(rc < 0.02, rc * 100, rc))
    add("3 EU Datastream vs Compustat T+1 close agree within 0.5% (share, 2018+)", "EU", "all", "all", "id1",
        float(np.mean(np.abs(rc - 1) < 0.005)), n=len(both))
# 7. extremes (listed for reading)
ext = E.assign(a=E.s_d0.abs()).sort_values("a", ascending=False).groupby("fam").head(5)
for r in ext.itertuples():
    add("7 largest |trade-day| moves", r.fam, r.grp, r.period, "d0", r.s_d0 * 1e4, note=f"{r.review} {r.id} {r.name}")

# 8. recomputed means vs table_legs.csv (written by s06 through common.summ)
TL = pd.read_csv(OUT / "table_legs.csv")
TL = TL[TL.basis == "matched control"].set_index(["family", "group", "period", "leg"])
rc = pd.DataFrame([r for r in rows if r["check"].startswith("1 ")])
diffs, tdiffs = [], []
for r in rc.itertuples():
    k = (r.family, r.group, r.period, r.leg)
    if k in TL.index:
        diffs.append(abs(TL.loc[k, "mean_bp"] - r.value)); tdiffs.append(abs(TL.loc[k, "t"] - r.t))
add("8 max |difference| recomputed vs table_legs mean (bp) / t", "all", "all", "all", "all", float(np.nanmax(diffs)),
    float(np.nanmax(tdiffs)), len(diffs))
out = pd.DataFrame(rows)
out.to_csv(OUT / "table_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 120)
print(out[~out.check.str.startswith(("1 ", "5 winsor"))].round(3).to_string(index=False))
