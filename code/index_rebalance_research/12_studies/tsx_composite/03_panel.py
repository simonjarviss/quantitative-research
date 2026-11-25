"""Step 3: daily wide panel (TSX sessions x Canadian lines) + float estimates + index proxy.

Writes data/panel.npz (float32 matrices) and data/panel_lines.parquet, data/float_monthly.parquet,
data/index_proxy.parquet.

Universe lines: Compustat '..C' lines (TSX/Canadian quotes), tpci 0 (common) or 4 (units), that were ever
Composite members, or are not TSX Venture (exchg 9). Eligibility flags (domicile, GICS) are added later.
Matrices:
  P   close (prccd), NaN when no row
  RET daily total return from prccd*trfd/ajexdi (trfd forward-filled; gaps > 5 sessions -> NaN)
  VT  TSX volume (cshtrd) in split-adjusted shares (x ajexdi)
  VC  Canadian consolidated volume (Datastream ds2scdqtprc exch 295) in split-adjusted shares
  VU  US-exchange volume (Compustat US lines of the same gvkey, exchg 11/12/14), split-adjusted
  SH  shares outstanding (cshoc x ajexdi), forward-filled; market cap = P * SH / AJ
"""
import numpy as np, pandas as pd
from common import DATA, tsx_calendar

cal = tsx_calendar()
lines = pd.read_parquet(DATA / "lines.parquet")
mem = pd.read_parquet(DATA / "members.parquet")
comp = mem[mem.gvkeyx == "000010"]
ever = set(zip(comp.gvkey, comp.iid))
ca = lines[lines.iid.str.endswith("C") & lines.tpci.isin(["0", "4"])].copy()
ca["ever_member"] = [k in ever for k in zip(ca.gvkey, ca.iid)]
ca = ca[ca.ever_member | (ca.exchg != 9)]
ca["lid"] = ca.gvkey + "_" + ca.iid

s = pd.read_parquet(DATA / "secd.parquet")
s["lid"] = s.gvkey + "_" + s.iid
sc = s[s.lid.isin(set(ca.lid))].copy()
sc = sc[sc.datadate.isin(cal)]
sc = sc.sort_values(["lid", "datadate"])
sc["trfd"] = sc.groupby("lid").trfd.ffill().fillna(1.0)
sc["tri"] = sc.prccd * sc.trfd / sc.ajexdi
sc["pos"] = cal.searchsorted(sc.datadate)
prev_tri = sc.groupby("lid").tri.shift(1)
prev_pos = sc.groupby("lid").pos.shift(1)
sc["ret"] = np.where((sc.pos - prev_pos) <= 5, sc.tri / prev_tri - 1, np.nan)
sc.loc[(sc.prccd <= 0) | (prev_tri <= 0), "ret"] = np.nan
sc["vta"] = sc.cshtrd * sc.ajexdi
sc["sha"] = sc.cshoc * sc.ajexdi
keep = sorted(set(sc.lid))
ca = ca[ca.lid.isin(keep)].sort_values("lid").reset_index(drop=True)
lid_index = pd.Index(ca.lid)
nd, nl = len(cal), len(ca)
print("lines", nl, "sessions", nd)

def wide(df, col):
    M = np.full((nd, nl), np.nan, dtype=np.float32)
    M[df.pos.values, lid_index.get_indexer(df.lid)] = df[col].values.astype(np.float32)
    return M

P = wide(sc, "prccd"); RET = wide(sc, "ret"); VT = wide(sc, "vta"); AJ = wide(sc, "ajexdi")
SH = pd.DataFrame(wide(sc, "sha")).ffill().values.astype(np.float32)

# US-exchange volume of the same gvkey (summed over US lines), split-adjusted with the US line's ajexdi
us = s[~s.iid.str.endswith("C") & s.exchg.isin([11, 12, 14]) & s.datadate.isin(cal)].copy()
us["v"] = us.cshtrd * us.ajexdi
usg = us.groupby(["gvkey", "datadate"]).v.sum().reset_index()
usg["pos"] = cal.searchsorted(usg.datadate)
VU = np.full((nd, nl), np.nan, dtype=np.float32)
g2cols = ca.groupby("gvkey").apply(lambda x: list(x.index), include_groups=False).to_dict()
for g, x in usg.groupby("gvkey"):
    for c in g2cols.get(g, []):
        VU[x.pos.values, c] = x.v.values
del us, usg

# Datastream link: CUSIP (from ISIN) or SEDOL, any region
dn = pd.read_parquet(DATA / "ds_names.parquet")
dn = dn[dn.typecode == "EQ"]
dn["pref"] = (dn.region == "CA") * 4 + (dn.isprimqt == 1) * 2 + (dn.primexchmnem == "TSE") * 1
c9 = ca.cusip.str[:9]
link = []
for i, (cus, sed) in enumerate(zip(c9, ca.sedol)):
    cand = dn[(dn.cusip9 == cus) | ((dn.primqtsedol == sed) & pd.notna(sed))]
    if len(cand):
        fl = cand.sort_values("pref", ascending=False).infocode.iloc[0]
        cac = sorted(set(cand.loc[cand.region == "CA", "infocode"]))
        link.append((i, int(fl), cac))
ca["ds_float_ic"] = np.nan; ca["ds_ca_ics"] = None
for i, fl, cac in link:
    ca.at[i, "ds_float_ic"] = fl; ca.at[i, "ds_ca_ics"] = ",".join(map(str, cac))
print("lines linked to Datastream:", ca.ds_float_ic.notna().mean().round(3))

# consolidated Canadian volume: if several CA infocodes, use the one with the most rows
cv = pd.read_parquet(DATA / "ds_consvol.parquet")
cv = cv[cv.marketdate.isin(cal) & cv.volume.notna()]
cnt = cv.groupby("infocode").size()
VC = np.full((nd, nl), np.nan, dtype=np.float32)
cvg = dict(tuple(cv.groupby("infocode")))
for i, ics in zip(ca.index, ca.ds_ca_ics):
    if not ics:
        continue
    ics = [int(x) for x in ics.split(",") if int(x) in cnt.index]
    if not ics:
        continue
    ic = max(ics, key=lambda x: cnt[x])
    x = cvg[ic]
    pos = cal.searchsorted(x.marketdate.values)
    aj = AJ[pos, i]
    VC[pos, i] = x.volume.values * np.where(np.isnan(aj), 1.0, aj)
print("lines with consolidated volume:", (~np.isnan(VC).all(0)).sum())

np.savez_compressed(DATA / "panel.npz", P=P, RET=RET, VT=VT, VC=VC, VU=VU, SH=SH, AJ=AJ,
                    dates=cal.values.astype("datetime64[D]"))
ca.to_parquet(DATA / "panel_lines.parquet", index=False)

# ---------- float estimates, monthly (first session of each month) ----------
ff = pd.read_parquet(DATA / "ds_float.parquet")
ff = ff[ff.stratholdpct.notna() & ff.freefloatpct.notna()]
ff["ff_ds"] = (ff.freefloatpct / 100).clip(0.01, 1.0)
lk = pd.read_parquet(DATA / "ds_ws_link.parquet").sort_values(["coderank", "inforank"]).drop_duplicates("infocode")
ws = pd.read_parquet(DATA / "ws_closely_held.parquet")
ws = ws[ws.item8021.notna() & ws.date_.notna()].copy()
ws["avail"] = ws.date_ + pd.Timedelta(days=120)          # assume item usable 4 months after its date
ws["ff_ws"] = (1 - ws.item8021 / 100).clip(0.01, 1.0)
months = pd.date_range("2004-01-01", "2026-07-01", freq="MS")
rows = []
ffg = dict(tuple(ff.groupby("infocode")))
wsg = dict(tuple(ws.groupby("code")))
ic2code = dict(zip(lk.infocode, lk.code))
for i, ic in zip(ca.index, ca.ds_float_ic):
    if np.isnan(ic):
        continue
    ic = int(ic)
    a = ffg.get(ic); b = wsg.get(ic2code.get(ic))
    if a is not None:
        a = a.sort_values("valdate")
        j = np.searchsorted(a.valdate.values, months.values, side="right") - 1
        ok = j >= 0
        v = np.where(ok, a.ff_ds.values[np.clip(j, 0, None)], np.nan)
        age = np.where(ok, (months.values - a.valdate.values[np.clip(j, 0, None)]).astype("timedelta64[D]").astype(int), 99999)
        v[age > 400] = np.nan
    else:
        v = np.full(len(months), np.nan)
    if b is not None:
        b = b.sort_values("avail")
        j = np.searchsorted(b.avail.values, months.values, side="right") - 1
        ok = j >= 0
        w = np.where(ok, b.ff_ws.values[np.clip(j, 0, None)], np.nan)
        age = np.where(ok, (months.values - b.avail.values[np.clip(j, 0, None)]).astype("timedelta64[D]").astype(int), 99999)
        w[age > 730] = np.nan
    else:
        w = np.full(len(months), np.nan)
    rows.append(pd.DataFrame({"line": i, "month": months, "ff_ds": v, "ff_ws": w}))
fm = pd.concat(rows, ignore_index=True)
fm.to_parquet(DATA / "float_monthly.parquet", index=False)
print("float rows", len(fm))

# ---------- index proxy: cap-weighted Composite members (actual membership), total return ----------
comp = comp.copy(); comp["fromdate"] = pd.to_datetime(comp.fromdate); comp["thru"] = pd.to_datetime(comp.thru)
MEM = np.zeros((nd, nl), dtype=bool)
for g, iid, f, t in zip(comp.gvkey, comp.iid, comp.fromdate, comp.thru):
    c = lid_index.get_indexer([g + "_" + iid])[0]
    if c < 0:
        continue
    a = cal.searchsorted(f); b = cal.searchsorted(t, side="right") if pd.notna(t) else nd
    MEM[a:b, c] = True
np.save(DATA / "member_matrix.npy", MEM)
cap = P * SH / pd.DataFrame(AJ).ffill().values     # SH is split-adjusted (x ajexdi); P is not
w = np.where(MEM, cap, 0.0)
w = np.nan_to_num(w)
wl = np.vstack([np.full((1, nl), 0.0), w[:-1]])          # weights at t-1
r = np.nan_to_num(RET)
mk = (wl * r).sum(1) / np.where(wl.sum(1) > 0, wl.sum(1), np.nan)
lev = pd.read_parquet(DATA / "index_levels.parquet")
lev = lev[lev.gvkeyx == "000010"].set_index("datadate").level.reindex(cal)
ip = pd.DataFrame({"date": cal, "proxy_tr": mk, "sptsx_price_ret": lev.pct_change(fill_method=None).values,
                   "n_members": MEM.sum(1)})
ip.to_parquet(DATA / "index_proxy.parquet", index=False)
ok = ip.dropna()
print("proxy vs S&P/TSX price index: corr", np.corrcoef(ok.proxy_tr, ok.sptsx_price_ret)[0, 1].round(4),
      "ann. mean diff (TR - price)", round((ok.proxy_tr - ok.sptsx_price_ret).mean() * 252, 4))
