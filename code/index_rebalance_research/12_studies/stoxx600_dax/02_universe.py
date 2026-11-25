"""Step 2: choose the daily-data universe from the month-end pass (data/secm.parquet).

Keeps, for 2007-01 .. 2026-06 month-ends:
  - companies whose primary ROW line (g_company.prirow) is listed in a European country and that rank in the
    top 1,500 of Europe by full EUR market cap at any month-end (STOXX 600 buffer band = FFMC ranks 550-750;
    1,500 covers low-float names and leaves room for matched controls);
  - German-listed companies ranked in the top 350 of Germany at any month-end (SDAX boundary = rank 160-180);
  - every line that was ever a STOXX 600 / DAX / MDAX / TecDAX member in Compustat.
For each kept company: its primary ROW line, all member lines, and its other common/preferred lines on a main
exchange of its home country.
Output: data/universe_lines.parquet, data/fx_daily.parquet (EUR per unit of local currency).
"""
import numpy as np, pandas as pd
from common import DATA

m = pd.read_parquet(DATA / "secm.parquet")
L = pd.read_parquet(DATA / "lines.parquet")
mem = pd.read_parquet(DATA / "members.parquet")
fx = pd.read_parquet(DATA / "fx.parquet")

# EUR per unit of currency X = exratd(EUR) / exratd(X)   (exratd = units of X per GBP)
w = fx.pivot_table(index="datadate", columns="tocurd", values="exratd")
eur = w.div(w["EUR"], axis=0).rdiv(1.0)          # = exratd(EUR)/exratd(X) ... computed below explicitly
eur = pd.DataFrame({c: w["EUR"] / w[c] for c in w.columns})
eur.index = pd.to_datetime(eur.index)
eur = eur.sort_index().ffill()
eur.reset_index().melt(id_vars="datadate", var_name="cur", value_name="eur_per").to_parquet(DATA / "fx_daily.parquet", index=False)

m["datadate"] = pd.to_datetime(m.datadate)
m["ym"] = m.datadate.dt.to_period("M")
lk = eur.stack().rename("eur_per").reset_index().rename(columns={"level_1": "cur", "tocurd": "cur"})
lk.columns = ["datadate", "cur", "eur_per"]
m = m.merge(lk, left_on=["datadate", "curcdd"], right_on=["datadate", "cur"], how="left")
m["mcap_eur"] = m.prccd * m.cshoc * m.eur_per
prim = L[L.iid == L.prirow][["gvkey", "iid", "excntry"]]
mp = m.merge(prim, on=["gvkey", "iid"])
mp = mp[(mp.ym >= pd.Period("2007-01", "M")) & mp.mcap_eur.notna() & (mp.mcap_eur > 0)]
mp["rk_eu"] = mp.groupby("ym").mcap_eur.rank(ascending=False)
de = mp[mp.excntry == "DEU"].copy()
de["rk_de"] = de.groupby("ym").mcap_eur.rank(ascending=False)
keep_g = set(mp.loc[mp.rk_eu <= 1500, "gvkey"]) | set(de.loc[de.rk_de <= 350, "gvkey"]) | set(mem.gvkey)
# other share classes / lines of the kept companies on a main exchange of the home country (common or preferred):
# needed for the "most liquid line" (STOXX) and "largest share class" (DAX) rules, e.g. preference shares
from common import MAIN_EXCH
pc = prim.set_index("gvkey").excntry
oth = L[L.gvkey.isin(keep_g) & L.exchg.isin(list(MAIN_EXCH)) & L.tpci.isin(["0", "1"]) & L.iid.str.endswith("W")]
oth = oth[oth.excntry == oth.gvkey.map(pc)]
u = pd.concat([prim[prim.gvkey.isin(keep_g)][["gvkey", "iid"]], mem[["gvkey", "iid"]], oth[["gvkey", "iid"]]]).drop_duplicates()
u.to_parquet(DATA / "universe_lines.parquet", index=False)
print("companies kept", len(keep_g), "lines", len(u))
print("by country (primary line):", prim[prim.gvkey.isin(keep_g)].excntry.value_counts().head(25).to_dict())
print("median EUR mcap of rank 1500 / 750 / 550 at month-ends:",
      mp[mp.rk_eu.round() == 1500].mcap_eur.median() / 1e9, mp[mp.rk_eu.round() == 750].mcap_eur.median() / 1e9,
      mp[mp.rk_eu.round() == 550].mcap_eur.median() / 1e9)
