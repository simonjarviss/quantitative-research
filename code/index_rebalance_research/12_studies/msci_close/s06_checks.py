"""Step 6: checks -> table_checks.csv (one row per check: check, cell, value, note).

1  Recomputation: every headline mean and t of table_legs.csv (matched control, split 'all') from
   events_legs_msci.csv with independent numpy code (cluster sums written out).
2  Source agreement: Datastream close at T vs the sister panel's Compustat close (share within 0.5%, after a x100
   pence/pound rescale); Datastream vs Compustat signed d0 / r1 / r1_5 (correlation, mean difference).
3  Leg identity: d0 = d0_on + d0_id (compounded vs summed; max gap in any cell mean).
4  Stale opens (T+1 open equal to the T close) and zero trade-day returns: share, and the overnight / d0 means
   without them. 4b (added 2026-10-15): the same for the controls and the index members (T+1 and T opens), and the
   overnight legs against controls / members without stale opens (s03 ctrl_stale_*, abx_*, ix_stale_*, ixx_*).
5  Outliers: trimmed means and medians vs means; dropping each review in turn (min / max of the cell mean).
6  Delisted within 20 sessions (no Datastream close at T+20: takeovers, mergers): count and the means without them.
7  Volume: median V_T / median(V T-20..T-1) per market x side (events are real trade-day flow).
8  Controls: share of events with a control cell, cell level, median number of controls.
9  Pricing code: s05.pricing_rows on close_dislocation's own S&P/TSX 2016+ events reproduces its table_pricing.csv
   rows (crowding = 'all').
"""
import numpy as np
import pandas as pd

from common import CD, OUT
from s04_tables import load
from s05_pricing import pricing_rows


def mean_t(x, g):
    x = np.asarray(x, float); g = np.asarray(g)
    ok = np.isfinite(x); x, g = x[ok], g[ok]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.sum() / n
    G = np.unique(g)
    s = np.array([(x[g == k] - m).sum() for k in G])
    se = np.sqrt(len(G) / (len(G) - 1) * (s ** 2).sum()) / n
    return m * 1e4, m / se


def main():
    X = load()
    T = pd.read_csv(OUT / "table_legs.csv")
    rows = []
    add = lambda c, cell, v, note="": rows.append(dict(check=c, cell=cell, value=v, note=note))
    # 1 recomputation
    tm = T[(T.basis == "matched control") & (T.split == "all")]
    dmax, tmax = 0.0, 0.0
    for r in tm.itertuples():
        d = X[(X.fam == r.family) & (X.grp == r.group)]
        m, t = mean_t(d[f"s_{r.leg}"], d.review)
        if np.isfinite(m):
            dmax = max(dmax, abs(m - r.mean_bp)); tmax = max(tmax, abs(t - r.t))
    add("1 recomputation", f"{len(tm)} cells", dmax, f"max |mean diff| bp; max |t diff| {tmax:.2e}")
    # 2 source agreement
    for mk, d in X.groupby("market"):
        a, b = d.ds_close_T.values, d.cs_close_T.values
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        q = a[ok] / b[ok]
        q = np.where(np.abs(np.log(q / 100)) < 0.05, q / 100, q)
        add("2 DS vs Compustat close at T", mk, float(np.mean(np.abs(q - 1) < 0.005)), f"share within 0.5% (n {ok.sum()})")
        for l in ["d0", "r1", "r1_5"]:
            u, v = d[f"s_{l}"], d[f"cs_{l}"]
            k = u.notna() & v.notna()
            add("2 DS vs Compustat signed leg", f"{mk} {l}", float(np.corrcoef(u[k], v[k])[0, 1]),
                f"correlation; mean DS {u[k].mean()*1e4:.1f} bp vs Compustat {v[k].mean()*1e4:.1f} bp (n {k.sum()})")
    # 3 leg identity
    gap = []
    for (f, g), d in X.groupby(["fam", "grp"]):
        k = d.s_d0_on.notna() & d.s_d0_id.notna()
        gap.append(abs(d.s_d0[k].mean() - (d.s_d0_on[k] + d.s_d0_id[k]).mean()) * 1e4)
    add("3 d0 = d0_on + d0_id", "all cells", max(gap), "max |cell mean gap| bp (controls' open coverage differs by leg)")
    # 4 stale opens, zero returns
    for (f, g), d in X.groupby(["fam", "grp"]):
        st = d.open_T1_eqprev
        m0, t0 = mean_t(d.s_on1, d.review)
        m1, t1 = mean_t(d.s_on1[~st], d.review[~st])
        add("4 stale T+1 open", f"{f} {g}", float(st.mean()), f"share; on1 {m0:.0f} bp (t {t0:.1f}) -> {m1:.0f} (t {t1:.1f}) without")
        z = d.d0_ret_is_zero
        m0, t0 = mean_t(d.s_d0, d.review)
        m1, t1 = mean_t(d.s_d0[~z], d.review[~z])
        add("4 zero trade-day return", f"{f} {g}", float(z.mean()), f"share; d0 {m0:.0f} bp (t {t0:.1f}) -> {m1:.0f} (t {t1:.1f}) without")
        # 4b (added 2026-10-15) stale opens among the controls and the index members, for the T+1 open (on1) and the
        # T open (d0_on): share, and the leg against controls / index members without stale opens, with and without
        # the event's own stale opens
        for leg, lab in [("on1", "T+1"), ("d0_on", "T")]:
            cs, cn = d[f"ctrl_stale_{leg}"], d[f"ctrl_n_{leg}"]
            ev_st = d[f"stale_{leg}"]
            ok = d[f"s_{leg}"].notna()
            sx, sxi = d.side * d[f"abx_{leg}"], d.side * d[f"ixx_{leg}"]
            m0, t0 = mean_t(d[f"s_{leg}"], d.review)
            m1, t1 = mean_t(sx, d.review)
            m2, t2 = mean_t(sx[~ev_st], d.review[~ev_st])
            add(f"4b stale {lab} open, controls", f"{f} {g}", float(cs.mean()),
                f"mean share per event (pooled {(cs * cn).sum() / cn.sum():.3f}, max {cs.max():.2f}; event's own "
                f"{ev_st[ok].mean():.3f}); {leg} matched {m0:.0f} bp (t {t0:.1f}) -> {m1:.0f} (t {t1:.1f}) vs controls "
                f"without stale opens -> {m2:.0f} (t {t2:.1f}) also without the event's own")
            i0, it0 = mean_t(d[f"si_{leg}"], d.review)
            i1, it1 = mean_t(sxi, d.review)
            i2, it2 = mean_t(sxi[~ev_st], d.review[~ev_st])
            add(f"4b stale {lab} open, index members", f"{f} {g}", float(d[f"ix_stale_{leg}"].mean()),
                f"mean cap-weighted share; {leg} index-adjusted {i0:.0f} bp (t {it0:.1f}) -> {i1:.0f} (t {it1:.1f}) vs "
                f"members without stale opens -> {i2:.0f} (t {it2:.1f}) also without the event's own")
    # 5 outliers / drop one review
    for (f, g), d in X.groupby(["fam", "grp"]):
        for l in ["d0", "on1", "r1", "r1_5"]:
            ms = []
            for rv in d.review.unique():
                e = d[d.review != rv]
                ms.append((mean_t(e[f"s_{l}"], e.review)[0], rv))
            lo, hi = min(ms), max(ms)
            add("5 drop one review", f"{f} {g} {l}", float(np.mean(d[f's_{l}']) * 1e4),
                f"mean; range {lo[0]:.0f} (without {lo[1]}) .. {hi[0]:.0f} (without {hi[1]})")
    # 6 delisted within 20 sessions
    for (f, g), d in X.groupby(["fam", "grp"]):
        dl = ~d.ds_close_T20_ok & (d.event_date < "2026-05-01")
        m0, t0 = mean_t(d.s_d0, d.review)
        m1, t1 = mean_t(d.s_d0[~dl], d.review[~dl])
        r0, rt0 = mean_t(d.s_r1, d.review)
        r1_, rt1 = mean_t(d.s_r1[~dl], d.review[~dl])
        add("6 no close at T+20 (delisted)", f"{f} {g}", int(dl.sum()),
            f"count; d0 {m0:.0f} -> {m1:.0f} (t {t1:.1f}); r1 {r0:.0f} -> {r1_:.0f} (t {rt1:.1f}) without")
    # 7 volume
    for (f, g), d in X.groupby(["fam", "grp"]):
        add("7 V_T / median V(T-20..T-1)", f"{f} {g}", float(np.nanmedian(d.vmult20)),
            f"median; share above 2x {np.nanmean(d.vmult20 > 2):.2f} (CA consolidated, EU primary venue)")
    # 8 controls
    for mk, d in X.groupby("market"):
        add("8 controls", mk, float((d.n_ctrl > 0).mean()),
            f"share with a cell; median n_ctrl {d.n_ctrl[d.n_ctrl > 0].median():.0f}; levels "
            + ", ".join(f"{k} {v}" for k, v in d.ctrl_level.value_counts().items()))
    # 9 pricing code vs close_dislocation
    E = pd.read_csv(CD / "events_legs.csv", low_memory=False)
    E = E[(E.fam == "CA S&P/TSX") & (E.period == "2016+")]
    new = []
    for grp, d in E.groupby("grp"):
        new += pricing_rows(d, "2016+", "CA S&P/TSX", grp, min_terciles=60)
    new = pd.DataFrame(new)
    ref = pd.read_csv(CD / "table_pricing.csv")
    ref = ref[(ref.family == "CA S&P/TSX") & (ref.period == "2016+") & (ref.crowding == "all")]
    k = ["period", "family", "group", "flow_tercile", "crowding"]
    m = ref.merge(new, on=k, suffixes=("_r", "_n"))
    num = [c for c in ref.columns if c not in k]
    diff = max(np.nanmax(np.abs(m[c + "_r"] - m[c + "_n"])) for c in num)
    add("9 pricing code reproduces close_dislocation s09", f"{len(m)} of {len(ref)} rows", diff, "max abs diff")
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "table_checks.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140); pd.set_option("display.max_rows", 300)
    print(C.to_string())


if __name__ == "__main__":
    main()
