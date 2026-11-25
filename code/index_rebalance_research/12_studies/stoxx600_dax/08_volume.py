"""Step 8: volume footprint around T, STOXX 600 and DAX family.

Volume bases (stated in every output row):
  primary   = primary-exchange shares traded (Compustat g_secd.cshtrd = Datastream primary quote; r2_warehouse §5.2)
  de_consol = German exchanges consolidated (Datastream ds2primqtprc.consolvol: Xetra + Frankfurt + regionals +
              Tradegate, no MTFs) - German lines only
  lit_cons  = primary + lit MTF order books (Cboe DXE/CXE/BXE, Turquoise, Aquis, Equiduct; ds2scdqtprc) - names in
              data/mtf_infocodes.parquet, from 2013 (run 01_pull_data.py dsmtf after the prep step below)
turnover(t) = volume(t) / shares outstanding(t). Abnormal turnover in a block = mean daily turnover in the block /
mean daily turnover over the baseline T-252..T-127 (months -12 to -6). Blocks of 21 sessions: m-6 = T-126..T-106 ...
m-1 = T-21..T-1, then T, m+1 = T+1..T+21, m+2 = T+22..T+42. Trade-day multiple = volume(T) / median volume(T-20..T-1).
Groups: actual review changes, predicted changes (rules model) at C and H3, controls (the control universe of
07_returns.py at C: not predicted, not changed).
Usage: python3 08_volume.py prep   -> writes data/mtf_infocodes.parquet (then pull dsmtf)
       python3 08_volume.py        -> volume_footprint.csv, volume_summary.csv
"""
import sys
import numpy as np, pandas as pd
from common import DATA, OUT, eu_calendar, xetra_calendar, review_dates, stoxx_country_ok

cal = eu_calendar()
z = np.load(DATA / "panel.npz")
P, VP, SH, AJ, MC_FX, VAL = z["P"], z["VP"], z["SH"], z["AJ"], z["FX"], z["VAL"]
AJf = pd.DataFrame(AJ).ffill().values
L = pd.read_parquet(DATA / "panel_lines.parquet")
nd, nl = P.shape
gv = L.gvkey.values
ces = pd.read_csv(OUT / "car_events_stoxx.csv", dtype={"gvkey": str}, usecols=["review", "form", "portfolio", "col"])
ced = pd.read_csv(OUT / "car_events_dax.csv", dtype={"gvkey": str}, usecols=["review", "form", "portfolio", "col"])


def groups_for(family):
    ce = ces if family == "stoxx" else ced
    s1, s2 = ("add", "del") if family == "stoxx" else ("up", "down")
    out = {}
    for (rv, form, port), g in ce.groupby(["review", "form", "portfolio"]):
        if form == "P130":
            continue
        if port.startswith("hind") and form == "C":
            out.setdefault(rv, {})["actual_" + port.split("_")[1]] = g.col.values
        if port.startswith("pred_") and form in ("C", "H3"):
            out.setdefault(rv, {})[f"pred{form}_" + port.split("_")[1]] = g.col.values
    return out


def controls(family, q):
    """Control universe at C, as in 07_returns.py (recomputed: one line per company, not predicted/changed)."""
    f = cal.get_loc(q.C)
    live = (~np.isnan(P[max(0, f - 9):f + 1])).any(0)
    op = L.gsector.notna().values & (L.tpci == "0").values & L.home_line.values
    mc = P[f] * SH[f] / AJf[f] * MC_FX[f]
    ce = ces if family == "stoxx" else ced
    excl = set(ce.loc[ce.review == q.review, "col"])
    if family == "stoxx":
        diag = pd.read_csv(OUT / "prediction_diagnostics_stoxx.csv").set_index(["review", "horizon"])
        L550 = diag.loc[(q.review, "C"), "L550_bn"] * 1e9
        ok = np.array([stoxx_country_ok(c, q.review) for c in L.home_cty])
        m = live & op & ok & (mc >= 0.15 * L550) & (mc <= 6 * L550)
    else:
        m = live & op & (L.home_cty.values == "DEU") & (mc >= 1e8)
    m[list(excl)] = False
    adtv = np.nan_to_num(np.nanmean(VAL[max(0, f - 62):f + 1], 0), nan=-1)
    idx = np.where(m)[0]
    df = pd.DataFrame({"g": gv[idx], "c": idx, "a": adtv[idx]}).sort_values(["g", "a"], ascending=[True, False])
    excl_g = set(gv[list(excl)]) if excl else set()
    df = df[~df.g.isin(excl_g)]
    return df.drop_duplicates("g").c.values


# control universe per review (saved for 09_shorts.py)
crow = []
for fam in ["stoxx", "dax"]:
    rev = review_dates(cal if fam == "stoxx" else xetra_calendar(), fam)
    gr = groups_for(fam)
    for q in rev.itertuples():
        if q.review in gr:
            for c in controls(fam, q):
                crow.append((fam, q.review, int(c)))
pd.DataFrame(crow, columns=["family", "review", "col"]).to_parquet(DATA / "controls_C.parquet", index=False)

if len(sys.argv) > 1 and sys.argv[1] == "prep":
    cols = set()
    rng = np.random.default_rng(7)
    for fam in ["stoxx", "dax"]:
        rev = review_dates(cal if fam == "stoxx" else xetra_calendar(), fam)
        gr = groups_for(fam)
        for q in rev[rev["T"] >= "2014-01-01"].itertuples():
            for k, v in gr.get(q.review, {}).items():
                if k.startswith("actual") or k.startswith("predC"):
                    cols |= set(v)
            if q.review in gr:
                c = controls(fam, q)
                cols |= set(rng.choice(c, size=min(40, len(c)), replace=False))
    ic = L.loc[sorted(cols), "ds_pq"].dropna().astype(int).unique()
    pd.DataFrame({"infocode": ic}).to_parquet(DATA / "mtf_infocodes.parquet", index=False)
    print("mtf infocodes:", len(ic))
    sys.exit()

# ---------------- volume matrices on the panel
VDE = np.full((nd, nl), np.nan, np.float32)
dp = pd.read_parquet(DATA / "ds_prim.parquet", columns=["infocode", "marketdate", "volume", "consolvol"])
dp["marketdate"] = pd.to_datetime(dp.marketdate)
dp = dp[dp.marketdate.isin(cal)]
ic2cols = {}
for i, ic in zip(L.index, L.ds_pq):
    if not np.isnan(ic):
        ic2cols.setdefault(int(ic), []).append(i)
de = dp[dp.consolvol.notna() & dp.infocode.isin([ic for ic, cs in ic2cols.items() if any(L.home_cty[c] == "DEU" for c in cs)])]
for ic, g in de.groupby("infocode"):
    r_ = cal.get_indexer(g.marketdate)
    for c in ic2cols[ic]:
        if L.home_cty[c] == "DEU":
            VDE[r_, c] = g.consolvol.values * AJf[r_, c]      # split-adjust like VP
VLIT = np.full((nd, nl), np.nan, np.float32)
try:
    mt = pd.read_parquet(DATA / "ds_mtf.parquet")
    mt["marketdate"] = pd.to_datetime(mt.marketdate)
    mt = mt[mt.marketdate.isin(cal)].groupby(["infocode", "marketdate"]).volume.sum().reset_index()
    pv = dp[dp.infocode.isin(mt.infocode.unique()) & (dp.marketdate >= "2013-01-01")][["infocode", "marketdate", "volume"]]
    mm = pv.merge(mt.rename(columns={"volume": "mtf"}), on=["infocode", "marketdate"], how="left")
    mm["lit"] = mm.volume + mm.mtf.fillna(0)
    for ic, g in mm.groupby("infocode"):
        r_ = cal.get_indexer(g.marketdate)
        for c in ic2cols.get(ic, []):
            VLIT[r_, c] = g.lit.values * AJf[r_, c]
    print("lit-consolidated lines:", int((~np.isnan(VLIT).all(0)).sum()))
except FileNotFoundError:
    print("no ds_mtf.parquet: lit-consolidated basis skipped")

BASES = {"primary": VP, "de_consol": VDE, "lit_cons": VLIT}
CT = {k: g.col.values for k, g in pd.read_parquet(DATA / "controls_C.parquet").groupby(["family", "review"])}
blocks = {f"m-{k}": (-21 * k, -21 * k + 20) for k in range(6, 0, -1)}
blocks.update({"T": (0, 0), "m+1": (1, 21), "m+2": (22, 42)})
rows = []
for fam in ["stoxx", "dax"]:
    rev = review_dates(cal if fam == "stoxx" else xetra_calendar(), fam)
    gr = groups_for(fam)
    for q in rev.itertuples():
        if q.review not in gr:
            continue
        iT = cal.get_loc(q.T)
        if iT + 42 >= nd:
            continue
        grp = dict(gr[q.review]); grp["control"] = CT.get((fam, q.review), np.array([], int))
        for g, cols in grp.items():
            for c in cols:
                for bname, V in BASES.items():
                    if bname == "de_consol" and L.home_cty[c] != "DEU":
                        continue
                    TO = V[:, c] / SH[:, c]
                    base = TO[iT - 252:iT - 126]
                    if np.sum(~np.isnan(base)) < 60 or not np.nanmean(base) > 0:
                        continue
                    b = np.nanmean(base)
                    r = dict(family=fam, review=q.review, year=q.year, group=g, col=c, cty=L.home_cty[c], basis=bname)
                    for k, (a0, a1) in blocks.items():
                        r[f"abto_{k}"] = np.nanmean(TO[iT + a0:iT + a1 + 1]) / b
                    med = np.nanmedian(V[iT - 20:iT, c])
                    r["mult_T"] = V[iT, c] / med if med > 0 else np.nan
                    r["mult_T1"] = V[iT + 1, c] / med if med > 0 else np.nan
                    if bname == "lit_cons":
                        r["prim_share_normal"] = np.nanmedian(VP[iT - 20:iT, c] / VLIT[iT - 20:iT, c])
                        r["prim_share_T"] = VP[iT, c] / VLIT[iT, c] if VLIT[iT, c] > 0 else np.nan
                    rows.append(r)
    print(fam, "done", len(rows), flush=True)
fp = pd.DataFrame(rows)
fp.to_csv(OUT / "volume_footprint.csv", index=False)
order = ["abto_m-6", "abto_m-5", "abto_m-4", "abto_m-3", "abto_m-2", "abto_m-1", "abto_T", "abto_m+1", "abto_m+2",
         "mult_T", "mult_T1", "prim_share_normal", "prim_share_T"]
sm = fp.groupby(["family", "basis", "group"])[[c for c in order if c in fp.columns]].median().round(2)
sm.insert(0, "n_names", fp.groupby(["family", "basis", "group"]).size())
sm.insert(1, "n_reviews", fp.groupby(["family", "basis", "group"]).review.nunique())
sm.to_csv(OUT / "volume_summary.csv")
print(sm.to_string())
