"""Step 8: checks. Writes table_checks.csv and table_known_events.csv.

1. Known MSCI changes appear in the holdings diffs: MSCI's public Standard lists for Nov 2025, Feb 2026 and Aug 2026
   and the Small Cap list for Aug 2026 (11_methodology/sources/msci/*.txt), Canada and the 15 DM-Europe countries,
   matched by name to s04 events at that review.
2. The two Standard rulers agree: CSCA/IMEA vs the MSCI World fund SWDA (adds/deletes found by both).
3. Rulers' ownership fraction k: s12 quarter-end (shrout2) vs the CH daily file with panel shares outstanding.
4. pi_visible in line with 04_models/flow_sizing.md §4.4: the Europe-only MSCI trackers vs the §4.4 proxy (0.32%,
   EUR 41.9bn / STOXX 600 cap); MSCI Canada vs the S&P/TSX Composite trackers (1.00%); implied index MVs vs Datastream
   S&P/TSX Composite and STOXX 600 MVs.
5. Headline R recomputed with independent code (explicit date dummies, own clustered SEs).
"""
import re

import numpy as np
import pandas as pd

import holdings_ch as H
from common import DATA, OUT, STUDIES, reviews

SRC = STUDIES.parent / "11_methodology" / "sources" / "msci"
CTRY_NAME = {"CANADA": "CA", "UNITED KINGDOM": "GB", "FRANCE": "FR", "GERMANY": "DE", "SWITZERLAND": "CH",
             "NETHERLANDS": "NL", "SWEDEN": "SE", "SPAIN": "ES", "ITALY": "IT", "DENMARK": "DK", "BELGIUM": "BE",
             "FINLAND": "FI", "NORWAY": "NO", "IRELAND": "IE", "AUSTRIA": "AT", "PORTUGAL": "PT"}
STOP = {"PLC", "AG", "SA", "SE", "NV", "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "GROUP", "HOLDING", "HOLDINGS",
        "CO", "THE", "AB", "ASA", "OYJ", "SPA", "AS", "A/S", "NEW", "CLASS", "CL", "B", "A", "GB", "CA", "DE", "FR",
        "UK", "SHS", "REG", "PREF", "VOTING", "SUBORDINATE", "SV", "SVS", "UNIT", "UNITS", "TRUST", "REIT", "&", "OF",
        "AND", "DE", "LA", "ORD", "N", "I", "SDB", "SER", "CDI", "INTL", "INTERNATIONAL", "COMPANY"}


def toks(s):
    s = re.sub(r"\(.*?\)", " ", str(s).upper())
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return [t for t in s.split() if t not in STOP and len(t) > 1]


def same(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return False
    return ta[0] == tb[0] or (ta[0][:5] == tb[0][:5] and len(ta[0]) >= 5) or len(set(ta) & set(tb)) >= 2


def parse_list(path):
    """MSCI public list: blocks 'MSCI <COUNTRY> INDEX' / 'Additions   Deletions' / two columns."""
    out, ctry, cut = [], None, None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*MSCI (.+?) (SMALL CAP )?INDEX\s*$", line)
        if m:
            ctry, cut = m.group(1).strip(), None
            continue
        if ctry and re.match(r"\s*Additions\s+Deletions", line):
            cut = line.index("Deletions")
            continue
        if ctry and cut is not None:
            if not line.strip():
                continue
            if line.strip().startswith(("MSCI", "Region", "Page")):
                ctry, cut = None, None
                continue
            left, right = line[:cut].strip(), line[cut:].strip()
            for side, nm in (("add", left), ("delete", right)):
                if nm and nm.lower() != "none":
                    out.append(dict(country=ctry, side=side, name=nm))
    return pd.DataFrame(out)


def known_events():
    E = pd.read_csv(OUT / "events.csv", parse_dates=["T"])
    lists = [("2025-11", "msci_nov25_standard_public_list.txt", "STD"), ("2026-02", "msci_feb26_standard_public_list.txt", "STD"),
             ("2026-08", "msci_aug26_standard_public_list.txt", "STD"), ("2026-08", "msci_aug26_smallcap_public_list.txt", "SC")]
    rows = []
    for rev, fn, seg in lists:
        L = parse_list(SRC / fn)
        L = L[L.country.isin(CTRY_NAME)]
        x = E[E.review == rev]
        for r in L.itertuples():
            c = CTRY_NAME[r.country]
            mkt = "CA" if c == "CA" else "EU"
            if seg == "STD":
                ev = ["ADD_STD", "MIG_UP"] if r.side == "add" else ["DEL_STD", "MIG_DOWN"]
            else:
                ev = ["ADD_SC", "MIG_DOWN"] if r.side == "add" else ["DEL_SC", "MIG_UP"]
            cand = x[(x.mkt == mkt) & x.event.isin(ev)]
            hit = cand[[same(r.name, n) for n in cand["name"]]]
            rows.append(dict(review=rev, segment=seg, country=c, side=r.side, msci_name=r.name, found=len(hit) > 0,
                             event=";".join(hit.event) if len(hit) else "", holdings_name=";".join(hit["name"]) if len(hit) else ""))
    K = pd.DataFrame(rows)
    K.to_csv(OUT / "table_known_events.csv", index=False)
    return K


def ruler_agreement():
    E = pd.read_csv(OUT / "events.csv", parse_dates=["T"])
    out = []
    for mkt in ("CA", "EU"):
        x = E[(E.mkt == mkt) & E.event.isin(["ADD_STD", "MIG_UP", "DEL_STD", "MIG_DOWN"])]
        add = x.event.isin(["ADD_STD", "MIG_UP"])
        w_add = (x.h0_wld.fillna(0) == 0) & (x.h1_wld.fillna(0) > 0)
        w_del = (x.h0_wld.fillna(0) > 0) & (x.h1_wld.fillna(0) == 0)
        out.append(dict(check=f"{mkt}: Standard entries in the ruler also entered SWDA (MSCI World)", value=float(w_add[add].mean()), n=int(add.sum())))
        out.append(dict(check=f"{mkt}: Standard exits in the ruler also left SWDA", value=float(w_del[~add].mean()), n=int((~add).sum())))
    return out


def k_check():
    """CSCA / IMEA ownership fraction: s12 quarter-end (shrout2) vs CH file + sister-panel shares outstanding."""
    import s04_events as S4
    R = pd.read_parquet(DATA / "ref_names.parquet")
    out = []
    for q in ["2019-12-31", "2022-12-30", "2025-12-31"]:
        for fund, mkt, ref in (("CSCA", "CA", "CSCA"), ("IMEA", "EU", "IMEA")):
            d = pd.Timestamp(q)
            while H.load(fund, d) is None and d > pd.Timestamp(q) - pd.Timedelta(days=5):
                d -= pd.Timedelta(days=1)
            x = S4.lines(fund, d, mkt)
            if x is None:
                continue
            so = S4.shares_out(mkt, list(x.index), d)
            own = pd.Series({s: x.loc[s, "shares"] / v for s, v in so.items()})
            k_ch = float(own[(own > 0) & (own < 1)].quantile(0.9))
            qq = pd.Timestamp(q) + pd.offsets.QuarterEnd(0)
            r = R[(R.q == qq) & (R.ref == ref)]
            k12 = float(r.own.quantile(0.9)) if len(r) else np.nan
            out.append(dict(check=f"k {fund} {q}: CH file + panel shares / s12 shrout2", value=k_ch / k12, n=len(own)))
    return out


def pi_checks():
    F = pd.read_parquet(DATA / "pi_funds.parquet")
    S = pd.read_parquet(DATA / "pi_series.parquet")
    out = []
    last = F.q.max()
    x = F[(F.q == last) & (F.seg == "STD") & (F.mkt == "EU")].copy()
    W = x.groupby("ctry").W.first()
    x["kw"] = x.k * x.ctry.map(W / W.sum())
    reg = x.fundname.str.upper().str.contains(r"EUROPE|EMU|EURO |EUROZONE|UNITED KINGDOM|\bUK\b|GERMANY|FRANCE|SWITZ|NETHERL|SWEDEN|SPAIN|ITALY|DENMARK|BELG|FINLAND|NORWAY|AUSTRIA|IRELAND|EURO IMI")
    glob = x.fundname.str.upper().str.contains(r"WORLD|ACWI|EAFE|ALL COUNTRY|INTL|INTERNATIONAL|TOTAL|DEVELOPED|KOKUSAI|GLOBAL")
    eu_only = float(x.kw[reg & ~glob].sum())
    pan = float(x.kw[x.fundname.str.upper().str.contains("EUROPE") & ~glob].sum())
    out.append(dict(check=f"MSCI Europe pi_visible {last.date()} (all visible MSCI trackers)", value=float(x.kw.sum()), n=int(x.fundno.nunique())))
    out.append(dict(check="  of which Europe-region and country MSCI funds", value=eu_only, n=int(x[reg & ~glob].fundno.nunique())))
    out.append(dict(check="  of which pan-European MSCI Europe funds only (flow_sizing §4.4 proxy 0.32% on the STOXX 600 cap)", value=pan,
                    n=int(x[x.fundname.str.upper().str.contains("EUROPE") & ~glob].fundno.nunique())))
    c = F[(F.q == last) & (F.seg == "STD") & (F.mkt == "CA")]
    ca_only = c.fundname.str.upper().str.contains("CANADA|CNDA")
    out.append(dict(check=f"MSCI Canada pi_visible {last.date()} (all)", value=float(c.k.sum()), n=int(c.fundno.nunique())))
    out.append(dict(check="  of which MSCI Canada country funds (S&P/TSX Composite trackers: 1.00%, flow_sizing §4.4)", value=float(c.k[ca_only].sum()), n=int(ca_only.sum())))
    out.append(dict(check="  of which IMI funds (do not trade Standard<->Small Cap migrations)", value=float(c.k[c.ftype == "IMI"].sum()), n=int((c.ftype == "IMI").sum())))
    # implied MSCI index MV from the ruler vs Datastream index MV (close_dislocation caches)
    mv = pd.read_parquet(STUDIES / "close_dislocation" / "data" / "index_mv.parquet")
    mv["valuedate"] = pd.to_datetime(mv.valuedate)
    for fund, mkt, fam, q in (("CSCA", "CA", "TSX", "2025-12-31"), ("IMEA", "EU", "STOXX", "2025-12-31")):
        d = pd.Timestamp(q)
        while H.load(fund, d) is None:
            d -= pd.Timedelta(days=1)
        import s04_events as S4
        x = S4.lines(fund, d, mkt)
        so = S4.shares_out(mkt, list(x.index), d)
        own = pd.Series({s: x.loc[s, "shares"] / v for s, v in so.items()})
        k = float(own[(own > 0) & (own < 1)].quantile(0.9))
        val = float(x.mv.sum())             # fund currency (CSCA USD, IMEA EUR)
        m = mv[(mv.family == fam) & (mv.valuedate <= d)].sort_values("valuedate").tail(1)
        if len(m):
            ref_mv = float(m.mv.iloc[0]) * 1e6
            if fam == "TSX":            # CSCA reports USD prices: USD per CAD = file price / panel CAD close
                Z, D, COL = S4.panel_for("CA")
                i = D.searchsorted(d, side="right") - 1
                r = [x.loc[s, "price"] / Z["P"][i, COL.get(s)] for s in x.index if COL.get(s) is not None]
                ref_mv *= float(np.nanmedian(r))
            out.append(dict(check=f"implied MSCI {'Canada' if mkt == 'CA' else 'Europe'} float MV {d.date()} / Datastream {'S&P/TSX Composite' if fam == 'TSX' else 'STOXX 600'} MV",
                            value=(val / k) / ref_mv, n=len(own)))
    return out


def r_recompute():
    """MAIN R with explicit event-date dummies and a hand-written CR1 cluster-robust SE."""
    import s06_calibrate as S6
    E = pd.read_csv(OUT / "demand.csv", parse_dates=["T"])
    out = []
    T = pd.read_csv(OUT / "table_R.csv")
    for mkt in ("CA", "EU"):
        Em = E[(E.mkt == mkt) & (E["T"] <= "2026-06-30")]
        m = S6.prep(Em[Em.event.isin(S6.STD_EVENTS + ["U_STD"])], p=0.002)
        D = pd.get_dummies(m.Tk).astype(float).values
        X = np.column_stack([m[["x"] + S6.CTRL].values, D])
        y = m.y.values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        XtXi = np.linalg.pinv(X.T @ X)
        meat = np.zeros((X.shape[1], X.shape[1]))
        for g in np.unique(m.Tk.values):
            i = m.Tk.values == g
            s = X[i].T @ e[i]
            meat += np.outer(s, s)
        G, n, k = len(np.unique(m.Tk.values)), len(y), X.shape[1]
        V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
        ref = T[T["sample"] == f"{mkt} MAIN: Standard adds/deletes/migrations vs unchanged members"].iloc[0]
        out.append(dict(check=f"{mkt} MAIN R recomputed (explicit dummies) vs table_R", value=float(b[0]), n=n,
                        ref=f"{ref.R:.3f} (se {ref.se:.3f}); recomputed se {np.sqrt(V[0, 0]):.3f}"))
    return out


def main():
    K = known_events()
    rows = []
    for (rev, seg, mk), g in K.assign(mk=np.where(K.country == "CA", "CA", "EU")).groupby(["review", "segment", "mk"]):
        rows.append(dict(check=f"known events {rev} {seg} {mk}: share of MSCI-listed changes found in the holdings diff",
                         value=float(g.found.mean()), n=len(g)))
    rows += ruler_agreement() + k_check() + pi_checks() + r_recompute()
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "table_checks.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(C.to_string())
    print(K[~K.found].to_string())


if __name__ == "__main__":
    main()
