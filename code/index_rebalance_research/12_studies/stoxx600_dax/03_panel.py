"""Step 3: daily wide panel (pan-European sessions x universe lines) + float estimates + index proxies.

Writes data/panel.npz (float32 matrices), data/panel_lines.parquet, data/float_monthly.parquet,
data/member_*.npy, data/index_proxy.parquet, data/eu_calendar.parquet, data/xetra_calendar.parquet.

Matrices (local currency unless stated):
  P    close (prccd), NaN when no row
  RET  daily total return from prccd*trfd/ajexdi (trfd forward-filled; gap > 5 sessions -> NaN)
  VP   primary-exchange volume (cshtrd, Compustat = primary venue only; r2_warehouse §5.2), split-adjusted (x ajexdi)
  SH   shares outstanding (cshoc x ajexdi), forward-filled
  AJ   ajexdi;  FX  EUR per unit of the line's currency (forward-filled)
  VAL  primary value traded in EUR = cshtrd * prccd * FX
Market cap (EUR) = P * SH / AJ * FX.
Line roles: 'ret line' of a company = its index-member line when that line is on a main home exchange, else
the primary ROW line (g_company.prirow).
"""
import numpy as np, pandas as pd
from common import DATA, MAIN_EXCH

L0 = pd.read_parquet(DATA / "lines.parquet")
U = pd.read_parquet(DATA / "universe_lines.parquet")
mem = pd.read_parquet(DATA / "members.parquet")
s = pd.read_parquet(DATA / "secd.parquet")
s["datadate"] = pd.to_datetime(s.datadate)
s["lid"] = s.gvkey + "_" + s.iid

# ---------------- calendars
n = s.groupby("datadate").size()
cal = pd.DatetimeIndex(sorted(n[n >= 300].index))
pd.DataFrame({"date": cal}).to_parquet(DATA / "eu_calendar.parquet", index=False)
de = s[s.exchg == 171].groupby("datadate").size()
xcal = pd.DatetimeIndex(sorted(de[de >= 80].index))
xcal = xcal[xcal.isin(cal)]
pd.DataFrame({"date": xcal}).to_parquet(DATA / "xetra_calendar.parquet", index=False)
print("EU sessions", len(cal), cal[0].date(), cal[-1].date(), "| Xetra sessions", len(xcal))

# ---------------- lines
lines = L0.merge(U, on=["gvkey", "iid"])
lines["lid"] = lines.gvkey + "_" + lines.iid
lines = lines[lines.lid.isin(set(s.lid))].sort_values("lid").reset_index(drop=True)
lines["main_exch"] = lines.exchg.isin(list(MAIN_EXCH))
lines["is_prirow"] = lines.iid == lines.prirow
prim_cty = lines[lines.is_prirow].set_index("gvkey").excntry
lines["home_cty"] = lines.gvkey.map(prim_cty).fillna(lines.excntry)
lines["home_line"] = lines.main_exch & (lines.excntry == lines.home_cty)
lid_ix = pd.Index(lines.lid)
nd, nl = len(cal), len(lines)
print("lines", nl)

s = s[s.lid.isin(set(lines.lid)) & s.datadate.isin(cal)].copy()
s = s.sort_values(["lid", "datadate"]).drop_duplicates(["lid", "datadate"])
fx = pd.read_parquet(DATA / "fx_daily.parquet")
fx["datadate"] = pd.to_datetime(fx.datadate)
s = s.merge(fx.rename(columns={"cur": "curcdd"}), on=["datadate", "curcdd"], how="left")
s["trfd"] = s.groupby("lid").trfd.ffill().fillna(1.0)
s["ajexdi"] = s.ajexdi.where(s.ajexdi > 0)
s["ajexdi"] = s.groupby("lid").ajexdi.ffill().fillna(1.0)
s["tri"] = s.prccd * s.trfd / s.ajexdi
s["pos"] = cal.searchsorted(s.datadate)
prev_tri = s.groupby("lid").tri.shift(1)
prev_pos = s.groupby("lid").pos.shift(1)
prev_cur = s.groupby("lid").curcdd.shift(1)
s["ret"] = np.where(((s.pos - prev_pos) <= 5) & (s.curcdd == prev_cur), s.tri / prev_tri - 1, np.nan)
s.loc[(s.prccd <= 0) | (prev_tri <= 0), "ret"] = np.nan
s["vpa"] = s.cshtrd * s.ajexdi
s["sha"] = s.cshoc * s.ajexdi
s["val"] = s.cshtrd * s.prccd * s.eur_per
col = lid_ix.get_indexer(s.lid)


def wide(c, ffill=False):
    M = np.full((nd, nl), np.nan, dtype=np.float32)
    M[s.pos.values, col] = s[c].values.astype(np.float32)
    if ffill:
        M = pd.DataFrame(M).ffill().values.astype(np.float32)
    return M


P = wide("prccd"); RET = wide("ret"); VP = wide("vpa"); VAL = wide("val")
SH = wide("sha", True); AJ = wide("ajexdi", True); FX = wide("eur_per", True)
np.savez_compressed(DATA / "panel.npz", P=P, RET=RET, VP=VP, VAL=VAL, SH=SH, AJ=AJ, FX=FX,
                    dates=cal.values.astype("datetime64[D]"))
has = ~np.isnan(P)
lines["first_px"] = cal[np.minimum(np.where(has.any(0), has.argmax(0), nd - 1), nd - 1)]
lines["n_px"] = has.sum(0)
lines["cshtrd_null_share"] = 1 - (~np.isnan(VP)).sum(0) / np.maximum(lines.n_px.values, 1)

# ---------------- membership matrices (by line as recorded, and mapped to the company's ret line)
mem["fromdate"] = pd.to_datetime(mem.fromdate); mem["thru"] = pd.to_datetime(mem.thru)
mem = mem[~((mem.gvkeyx == "150095") & (mem.thru.isna()) & (mem.fromdate < "1990-01-01"))]   # stale LINDE AG 1988 spell
mem["lid"] = mem.gvkey + "_" + mem.iid
ok_ret = set(lines.loc[lines.home_line & (lines.n_px >= 60), "lid"])
prirow_lid = lines[lines.is_prirow].set_index("gvkey").lid.to_dict()


def ret_line(lid, gv):
    if lid in ok_ret:
        return lid
    return prirow_lid.get(gv, lid)


mem["rlid"] = [ret_line(l, g) for l, g in zip(mem.lid, mem.gvkey)]
mem.to_parquet(DATA / "members_mapped.parquet", index=False)
for gx, nm in [("150376", "stoxx")]:      # DAX family membership comes from Datastream lists (04_events.py)
    M = np.zeros((nd, nl), dtype=bool)
    x = mem[mem.gvkeyx == gx]
    for c, f, t in zip(lid_ix.get_indexer(x.rlid), x.fromdate, x.thru):
        if c < 0:
            continue
        a = cal.searchsorted(f); b = cal.searchsorted(t, side="right") if pd.notna(t) else nd
        M[a:b, c] = True
    np.save(DATA / f"member_{nm}.npy", M)
    print(nm, "members per session (median, 2008+):", int(np.median(M[cal >= "2008-01-01"].sum(1))),
          "| unmapped spells:", int((lid_ix.get_indexer(x.rlid) < 0).sum()))

# ---------------- Datastream link (ISIN of the line, else primary-quote SEDOL)
dn = pd.read_parquet(DATA / "ds_names.parquet")
dn = dn[dn.typecode == "EQ"].copy()
by_isin = dn.groupby("isin"); by_sed = dn.groupby("primqtsedol")
lines["ds_ic"] = np.nan; lines["ds_pq"] = np.nan
for i, (isin, sed, cty) in enumerate(zip(lines["isin"], lines["sedol"], lines["excntry"])):
    cand = None
    if isinstance(isin, str) and isin in by_isin.groups:
        cand = by_isin.get_group(isin)
    elif isinstance(sed, str) and sed in by_sed.groups:
        cand = by_sed.get_group(sed)
    if cand is None:
        continue
    cand = cand.assign(pref=(cand.isprimqt == 1) * 2 + (cand.enddate >= pd.Timestamp("2079-01-01")) * 1)
    best = cand.sort_values("pref", ascending=False).iloc[0]
    lines.at[i, "ds_ic"] = best.infocode
    lines.at[i, "ds_pq"] = best.primqtinfocode if pd.notna(best.primqtinfocode) else best.infocode
print("lines linked to Datastream:", round(lines.ds_ic.notna().mean(), 3),
      "| home ret lines linked:", round(lines.loc[lines.home_line, "ds_ic"].notna().mean(), 3))
lines.to_parquet(DATA / "panel_lines.parquet", index=False)

# ---------------- float, monthly (first day of each month), as in the TSX study
ff = pd.read_parquet(DATA / "ds_float.parquet")
ff["valdate"] = pd.to_datetime(ff.valdate)
ff = ff[ff.stratholdpct.notna() & ff.freefloatpct.notna()]
ff["ff_ds"] = (ff.freefloatpct / 100).clip(0.01, 1.0)
lk = pd.read_parquet(DATA / "ds_ws_link.parquet").sort_values(["coderank", "inforank"]).drop_duplicates("infocode")
ws = pd.read_parquet(DATA / "ws_closely_held.parquet")
ws = ws[ws.item8021.notna() & ws.date_.notna()].copy()
ws["avail"] = pd.to_datetime(ws.date_) + pd.Timedelta(days=120)
ws["ff_ws"] = (1 - ws.item8021 / 100).clip(0.01, 1.0)
months = pd.date_range("2006-01-01", "2026-07-01", freq="MS")
ffg = dict(tuple(ff.groupby("infocode"))); wsg = dict(tuple(ws.groupby("code")))
ic2code = dict(zip(lk.infocode, lk.code))
rows = []
for i, ic in zip(lines.index, lines.ds_pq):
    if np.isnan(ic):
        continue
    ic = int(ic)
    v = np.full(len(months), np.nan); w = np.full(len(months), np.nan)
    a = ffg.get(ic)
    if a is not None:
        a = a.sort_values("valdate")
        j = np.searchsorted(a.valdate.values, months.values, side="right") - 1
        ok = j >= 0; jj = np.clip(j, 0, None)
        v = np.where(ok, a.ff_ds.values[jj], np.nan)
        age = (months.values - a.valdate.values[jj]).astype("timedelta64[D]").astype(int)
        v[(~ok) | (age > 400)] = np.nan
    b = wsg.get(ic2code.get(ic))
    if b is not None:
        b = b.sort_values("avail")
        j = np.searchsorted(b.avail.values, months.values, side="right") - 1
        ok = j >= 0; jj = np.clip(j, 0, None)
        w = np.where(ok, b.ff_ws.values[jj], np.nan)
        age = (months.values - b.avail.values[jj]).astype("timedelta64[D]").astype(int)
        w[(~ok) | (age > 730)] = np.nan
    rows.append(pd.DataFrame({"line": i, "month": months, "ff_ds": v, "ff_ws": w}))
fm = pd.concat(rows, ignore_index=True)
fm.to_parquet(DATA / "float_monthly.parquet", index=False)

# ---------------- STOXX 600 local-currency total-return proxy (EUR cap weights at t-1, actual members)
MS = np.load(DATA / "member_stoxx.npy")
capE = P * SH / AJ * FX
w = np.nan_to_num(np.where(MS, capE, 0.0))
wl = np.vstack([np.zeros((1, nl)), w[:-1]])
r = np.nan_to_num(np.clip(RET, -0.5, 1.0))
proxy = (wl * r).sum(1) / np.where(wl.sum(1) > 0, wl.sum(1), np.nan)
lev = pd.read_parquet(DATA / "index_levels.parquet")
lev["datadate"] = pd.to_datetime(lev.datadate)
lw = lev.pivot_table(index="datadate", columns="gvkeyx", values="level").reindex(cal).ffill()
ip = pd.DataFrame({"date": cal, "stoxx_proxy_tr_local": proxy,
                   "stoxx600_price_eur": lw["150376"].pct_change(fill_method=None).values,
                   "dax_tr": lw["150095"].pct_change(fill_method=None).values,
                   "mdax_tr": lw["153256"].pct_change(fill_method=None).values,
                   "tecdax_tr": lw["153432"].pct_change(fill_method=None).values,
                   "n_stoxx_members": MS.sum(1)})
ip.to_parquet(DATA / "index_proxy.parquet", index=False)
ok = ip[ip.date >= "2008-01-01"].dropna(subset=["stoxx_proxy_tr_local", "stoxx600_price_eur"])
print("STOXX proxy (local TR) vs STOXX 600 price (EUR): corr",
      round(np.corrcoef(ok.stoxx_proxy_tr_local, ok.stoxx600_price_eur)[0, 1], 4),
      "ann. mean diff", round((ok.stoxx_proxy_tr_local - ok.stoxx600_price_eur).mean() * 252, 4))
