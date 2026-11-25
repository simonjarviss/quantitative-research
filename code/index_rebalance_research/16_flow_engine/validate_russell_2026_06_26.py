"""Reduced validation on the Fri 26 Jun 2026 Russell reconstitution close (US; optional input path of the engine).

    python3 validate_russell_2026_06_26.py        (run from 16_flow_engine with PYTHONPATH=<repo>/code)

What is feasible with free data and R2 (and what is not):
  - Changes: FTSE Russell's final R3000 addition and deletion lists (13_shadow/russell_dec2026/raw/ftse_2026-06-26/,
    parsed with that folder's read-only parser). R1000<->R2000 migrations and Russell share/float updates are NOT
    available: iShares US no longer serves dated holdings (checked 2026-10-11) and R2's Russell weights are zero-filled.
  - Index shares F: Compustat shares outstanding (secd.cshoc, 25 Jun 2026) x CRSP float ratio (msf_v2
    mthfloatshrqty / shrout, last month <= 2025-12); 1.0 when CRSP has no float (IPOs; flagged) - not Russell's float.
  - All adds/deletes are given the R2000 pi (most R3000 changes are R2000 changes; a large-cap add is R1000, flagged).
  - Volume: Compustat secd consolidated US volume; ADV = median T-25..T-6; background lift from 600 random US
    common stocks not on either list.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import r2

import engine as E
import inputs as I
from common import R2DIR, SHADOW_RUSSELL, VALID
from marketdata import fx_r2

T = "2026-06-26"


def lists() -> pd.DataFrame:
    sys.dont_write_bytecode = True    # read-only import: leave no __pycache__ in the shadow folder
    sys.path.insert(0, str(SHADOW_RUSSELL))
    import score as S                    # read-only import of the Russell shadow's PDF parser
    d = SHADOW_RUSSELL / "raw" / "ftse_2026-06-26"
    a = S.read_list(d / "ru3000-additions-20260626.pdf").assign(side="add")
    x = S.read_list(d / "ru3000-deletions-20260626.pdf").assign(side="delete")
    return pd.concat([a, x], ignore_index=True)


def pull(tics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = R2DIR / "russell_jun2026.parquet"
    if p.exists():
        z = pd.read_parquet(p)
        return z[z.kind == "sec"], z[z.kind == "px"], z[z.kind == "flt"]
    con = r2.connect()
    con.execute("CREATE OR REPLACE TEMP TABLE t AS SELECT unnest(?)::VARCHAR AS tic", [tics])
    sec = r2.sql(con, """SELECT s.gvkey, s.iid, s.tic, s.isin, s.exchg::INT exchg, s.tpci, c.conm
                         FROM comp_na_daily_all.security s JOIN comp_na_daily_all.company c USING (gvkey)
                         WHERE s.exchg IN (11, 12, 14) AND s.tpci = '0' AND (s.tic IN (SELECT tic FROM t)
                               OR random() < 0.12)""")
    sec = sec.drop_duplicates(["gvkey", "iid"])
    con.execute("CREATE OR REPLACE TEMP TABLE k AS SELECT * FROM (SELECT unnest(?) gvkey, unnest(?) iid)",
                [sec.gvkey.tolist(), sec.iid.tolist()])
    px = r2.sql(con, """SELECT d.gvkey, d.iid, d.datadate, d.cshtrd::DOUBLE cshtrd, d.prccd::DOUBLE prccd,
                               d.cshoc::DOUBLE cshoc FROM comp_na_daily_all.secd d JOIN k USING (gvkey, iid)
                        WHERE d.datadate BETWEEN DATE '2026-05-01' AND DATE '2026-06-26'""")
    lk = r2.sql(con, """SELECT gvkey, liid AS iid, lpermno::BIGINT permno FROM crsp_a_ccm.ccmxpf_lnkhist
                        WHERE linktype IN ('LC','LU') AND linkprim IN ('P','C') AND (linkenddt IS NULL OR linkenddt >= DATE '2025-06-01')""")
    lk = lk.merge(sec[["gvkey", "iid"]], on=["gvkey", "iid"])
    con.execute("CREATE OR REPLACE TEMP TABLE pm AS SELECT unnest(?)::BIGINT permno", [lk.permno.dropna().astype(int).tolist()])
    fl = r2.sql(con, """SELECT m.permno::BIGINT permno, m.mthcaldt, m.mthfloatshrqty::DOUBLE floatsh, m.shrout::DOUBLE shrout
                        FROM crsp_a_stock.msf_v2 m JOIN pm USING (permno) WHERE m.mthcaldt >= DATE '2025-06-01'""")
    fl = fl.sort_values("mthcaldt").groupby("permno").tail(1).merge(lk, on="permno")
    fl["ffr"] = (fl.floatsh / fl.shrout).clip(0.01, 1.0)
    z = pd.concat([sec.assign(kind="sec"), px.assign(kind="px"), fl[["gvkey", "iid", "ffr"]].assign(kind="flt")],
                  ignore_index=True)
    for c in z.columns:
        if z[c].dtype == object:
            z[c] = z[c].astype("string")
    z.to_parquet(p, index=False)
    return sec, px, fl


def main():
    L = lists()
    tics = sorted(set(L.symbol.str.replace("/", ".", regex=False)))
    sec, px, fl = pull(tics)
    px = px.copy()
    px["datadate"] = pd.to_datetime(px.datadate)
    sec = sec.copy()
    on_list = sec[sec.tic.isin(tics)].sort_values("exchg").drop_duplicates("tic")
    ffr = dict(zip(zip(fl.gvkey, fl.iid), fl.ffr))
    last = px[px.datadate <= "2026-06-25"].sort_values("datadate").groupby(["gvkey", "iid"]).tail(1)
    sh = dict(zip(zip(last.gvkey, last.iid), last.cshoc))
    pr = dict(zip(zip(last.gvkey, last.iid), last.prccd))
    rows = []
    for r in L.itertuples():
        m = on_list[on_list.tic == r.symbol.replace("/", ".")]
        if m.empty:
            continue
        g, i = m.gvkey.iloc[0], m.iid.iloc[0]
        S = sh.get((g, i), np.nan)
        if not np.isfinite(S):
            continue
        f = ffr.get((g, i), np.nan)
        note = "" if np.isfinite(f) else "no CRSP float: F = shares outstanding"
        F = S * (f if np.isfinite(f) else 1.0)
        rows.append(dict(change_type=r.side, isin=m["isin"].iloc[0] if isinstance(m["isin"].iloc[0], str) else None,
                         ticker=r.symbol, country="US", name=r.company, currency="USD",
                         old_index_shares=0.0 if r.side == "add" else F, new_index_shares=F if r.side == "add" else 0.0,
                         price=pr.get((g, i), np.nan), prob=1.0, status="final",
                         source="FTSE Russell final R3000 lists 2026-06-26 (PDF); F = Compustat cshoc x CRSP float",
                         source_ts="2026-06-26", notes=note))
    inp = pd.DataFrame(rows)
    inp["isin"] = inp["isin"].where(inp["isin"].notna(), None)
    I.write(inp, T, "R2000")
    print(f"lists {len(L)} names -> {len(inp)} with Compustat shares ({(inp.notes != '').sum()} without CRSP float)")
    # market data: Compustat consolidated volume
    fx = fx_r2(T)
    key_of = {}
    md = []
    Tt = pd.Timestamp(T)
    for (g, i), x in px.groupby(["gvkey", "iid"]):
        x = x.sort_values("datadate")
        pre = x[x.datadate < Tt]
        win = pre.iloc[-25:-5]
        win = win[win.cshtrd > 0]
        vt = x.loc[x.datadate == Tt, "cshtrd"]
        md.append(dict(gvkey=g, iid=i, adv_sh=float(win.cshtrd.median()) if len(win) >= 8 else np.nan,
                       v_T=float(vt.iloc[0]) if len(vt) else np.nan, price=float(x.prccd.iloc[-1])))
    md = pd.DataFrame(md).merge(sec[["gvkey", "iid", "tic", "isin"]].drop_duplicates(["gvkey", "iid"]),
                                on=["gvkey", "iid"])
    md["key"] = np.where(md["isin"].notna() & (md["isin"].astype(str).str.len() >= 12), md["isin"], "US:" + md.tic)
    md = md.assign(country="US", ccy="USD", adv_basis="Compustat secd consolidated", close_base_sh=md.adv_sh * 0.09,
                   close_basis="ADV x 9% (US normal close share)", close_n_days=0,
                   eur_per_ccy=fx["eur"]["USD"], cad_per_ccy=fx["cad"]["USD"])
    r = E.run(T, "best", include_us=True, md=md, label="final",
              extra_notes=["Russell reduced validation: adds/deletes only, R2000 pi for all, F = shares x CRSP float."])
    s = r["sheet"]
    chg = set(s.key)
    ctrl = md[~md.key.isin(chg) & (md.adv_sh > 0) & md.v_T.notna() & (md.price >= 1)]
    lift = float((ctrl.v_T / ctrl.adv_sh).median())
    from scipy import stats
    s = s[np.isfinite(s.adv_sh) & np.isfinite(s.v_T) & (s.adv_sh > 0)].copy()
    s["abnormal"] = s.v_T - lift * s.adv_sh
    s["excess"] = s.v_T - s.adv_sh
    s["absD"] = s.net_sh.abs()
    out = []
    for lab, x in [("all", s), ("adds", s[s.side == "BUY"]), ("deletes", s[s.side == "SELL"])]:
        o = dict(sample=lab, n=len(x), sp_excess=stats.spearmanr(x.absD, x.excess).statistic,
                 sp_abnormal=stats.spearmanr(x.absD, x.abnormal).statistic,
                 sp_x_mT=stats.spearmanr(x.absD / x.adv_sh, x.v_T / x.adv_sh).statistic)
        pos = x[x.abnormal > 0]
        b = np.polyfit(np.log(pos.absD), np.log(pos.abnormal), 1)
        pred = np.polyval(b, np.log(pos.absD))
        o.update(n_pos=len(pos), slope=b[0], r2_loglog=1 - ((np.log(pos.abnormal) - pred) ** 2).sum()
                 / ((np.log(pos.abnormal) - np.log(pos.abnormal).mean()) ** 2).sum(),
                 median_abnormal_over_Dvis=float(np.median(pos.abnormal / pos.net_sh_visible.abs())))
        out.append(o)
    t = pd.DataFrame(out)
    t.to_csv(VALID / f"{T}_russell_fit.csv", index=False, float_format="%.4g")
    s.to_csv(VALID / f"{T}_russell_names.csv", index=False, float_format="%.6g")
    pos = s[s.abnormal > 0].copy()
    b = np.polyfit(np.log(pos.absD), np.log(pos.abnormal), 1)
    pos["log_resid"] = np.log(pos.abnormal) - np.polyval(b, np.log(pos.absD))
    miss = pd.concat([pos.nlargest(6, "log_resid"), pos.nsmallest(6, "log_resid")])
    md_txt = [f"# Reduced validation: Russell reconstitution close {T}", "",
              f"Background lift (median V_T/ADV of {len(ctrl)} random non-changed US stocks): {lift:.2f}x.", "",
              t.round(3).to_markdown(index=False), "",
              "Biggest misses (log residual of the log-log fit):", "",
              miss[["ticker", "name", "side", "net_sh", "adv_sh", "v_T", "abnormal", "log_resid"]]
              .to_markdown(index=False, floatfmt=",.2f"), "",
              "Limits: no migrations or share/float updates (no dated US holdings); F is not Russell's float; "
              "every change uses the R2000 pi; volume is Compustat consolidated."]
    (VALID / f"{T}_russell_validation.md").write_text("\n".join(md_txt) + "\n")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
