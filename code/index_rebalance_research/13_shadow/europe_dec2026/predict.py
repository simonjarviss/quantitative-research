"""Live shadow predictions of the December 2026 STOXX Europe 600, DAX-family and FTSE 100/250 reviews (paper only).

Usage
  python3 predict.py                                    # fetch into raw/<today>/, predict on the latest close
  python3 predict.py --asof 2026-11-30 --tag cutoff     # cut-off run (after the close / after 17:30 CET)
  python3 predict.py --no-fetch --fetch-date 2026-10-12 --asof 2026-10-12 --tag initial   # from saved raw data
  python3 predict.py --family dax                       # one family (stoxx600 | dax | ftse_uk | all)
  python3 predict.py --ftse-indicative raw/<d>/ftse_indicative.csv   # condition FTSE on the 24 Nov indicative list

Each family writes predictions/<family>/{pred_<asof>_<tag>_all.csv, pred_..._changes.csv, watchlist_...,
paper_portfolio_..., summary_....json, engine_rows_....csv}; an existing file with the same name is moved to
predictions/<family>/superseded/ first (never overwritten). The engine inputs
16_flow_engine/inputs/2026-12-18/{stoxx600,dax_family,ftse_uk}.csv are rewritten with the latest run (status
'predicted'). A block with SHA-256 hashes of every output is appended to shadow_log.md.
Model docs: model_stoxx.py, model_dax.py, model_ftse.py; data: data.py, fetch.py, universe.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as Dm
import lib

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------ helpers
def engine_modules():
    sys.dont_write_bytecode = True
    if str(C.ENGINE) not in sys.path:
        sys.path.insert(0, str(C.ENGINE))
    import inputs as EI      # noqa: E402  (engine schema + validate(), read-only use)
    import pi as EP          # noqa: E402  (index_mv from R2 caches, read-only)
    return EI, EP


def index_caps(P: pd.DataFrame, t: pd.Timestamp) -> dict:
    """Index float cap at t = last R2 index MV (16_flow_engine pi.index_mv, to 2026-06-23) x Yahoo level ratio.
    Levels: ^STOXX (price), ^FTSE/^FTMC (price), ^GDAXI/^MDAXI/^TECDAX (total return: k carries ~1-2% dividend
    drift). Used only to turn holdings into index shares (F = H / k) for engine rows."""
    _, EP = engine_modules()
    W, _ = lib.price_panel(P)
    out = {}
    for idx, tick in C.LEVEL_TICKERS.items():
        mv, d = EP.index_mv(idx, pd.Timestamp("2026-06-23"))
        if not np.isfinite(mv) or tick not in W:
            continue
        s = W[tick].dropna()
        then = s[s.index <= d].iloc[-1]
        now = s[s.index <= t].iloc[-1]
        out[idx] = mv * now / then
    return out


def holdings_date(h: pd.DataFrame, W: pd.DataFrame, sym: pd.Series) -> pd.Timestamp | None:
    """Date whose Yahoo closes best match the holdings prices (median abs log difference)."""
    best, bd = 9.0, None
    for d in W.index[-8:]:
        x = W.loc[d, sym.values].values
        ok = np.isfinite(x) & (h.price.values > 0)
        if ok.sum() < 10:
            continue
        e = np.nanmedian(np.abs(np.log(x[ok] / h.price.values[ok])))
        if e < best:
            best, bd = e, d
    return bd


def horizon_label(asof: pd.Timestamp, cutoff: pd.Timestamp) -> str:
    if asof >= cutoff:
        return "C"
    if asof >= C.TRADE - pd.DateOffset(months=3):
        return "H3"
    return "H6"


def write_csv(df: pd.DataFrame, p: Path) -> Path:
    lib.archive_old(p)
    df.to_csv(p, index=False, float_format="%.6g")
    return p


# ------------------------------------------------------------------------------------------ STOXX
def do_stoxx(D, args, asof, caps, sym) -> dict:
    import model_stoxx as M
    r = M.run(D["stoxx_list"], D["H"]["EXSA"], D["prices"], sym, asof, C.CAL["stoxx600"]["cutoff"],
              index_cap_eur=caps.get("STOXX600"), n_mc=args.n_mc)
    E = r["E"]
    E["family"] = "stoxx600"
    E["category"] = np.where(E.member_now, "delete", "add")
    E["predicted"] = E.pred != "none"
    # headline probability: adds = margin to the fixed rank-550/551 line; deletes = joint Monte Carlo, because the
    # delete line moves with buffer crowding (June 2026 replay, 7-week run: the margin model put 0.1 of probability on
    # the 10 actual deletes, the Monte Carlo 1.7; Brier 0.0165 vs 0.0128; scores/validate_june2026.md)
    E["prob"] = np.where(E.member_now, E.p_mc_change, E.p_change)
    E["prob_basis"] = np.where(E.member_now, "monte carlo (buffer crowding)", "margin")
    ch = E[(E.prob >= C.P_ENGINE_MIN) | E.predicted].copy()
    ch = ch.sort_values(["category", "prob"], ascending=[True, False])
    out = pd.DataFrame(dict(
        family="stoxx600", index="STOXX600", category=ch.category, predicted=ch.predicted, key=ch.key,
        isin=ch.exsa_isin, name=ch.Instrument_Name, symbol=ch.symbol, country=ch.Country, currency=ch.ccy_px,
        prob=ch.prob, prob_basis=ch.prob_basis, p_margin=ch.p_change, p_mc=ch.p_mc_change, margin_pct=ch.margin_pct,
        line=ch.line_name,
        rank_list=ch["rank"], rank_t=ch.rank_t, ffmc_t_eur=ch.ffmc_t_eur, sigma_tot=ch.sigma_tot,
        F_old=np.where(ch.member_now, ch.F_index, 0.0), F_new=np.where(ch.member_now, 0.0, ch.F_implied),
        price_t=ch.px_t, flags=ch["flags"]))
    wl = E[(E.pred == "none") & (E.margin_log.abs() < C.WATCH_MARGIN) & (E.prob < C.P_ENGINE_MIN)]
    wl = wl.sort_values("prob", ascending=False)
    return dict(r=r, all=E, changes=out, watch=wl, meta=r["meta"])


# ------------------------------------------------------------------------------------------ DAX
DAX_INDEX = {"DAX": "DAX", "MDAX": "MDAX", "SDAX": "SDAX"}


def do_dax(D, args, asof, caps, sym) -> dict:
    import model_dax as M
    files = D["files"]
    main = Dm.F.read_list(Dm.latest_list(files, "mkdx", asof)[1])
    tec = Dm.F.read_list(Dm.latest_list(files, "tdxk", asof)[1])
    adj = pd.read_csv(C.HERE / "adjustments_dax.csv", dtype=str) if (C.HERE / "adjustments_dax.csv").exists() else None
    r = M.run(main, tec, D["H"], D["prices"], sym, D["info"], asof, C.CAL["dax"]["cutoff"], caps, n_mc=args.n_mc_dax,
              adjustments=adj)
    E = r["E"].set_index("ISIN")
    MV, MC = r["MV"].copy(), r["MC"].copy()
    if len(MV):
        MV["move_simple"] = MV.move.str.replace(r" \(.*\)$", "", regex=True)
    rows = {}
    for x in MV.itertuples() if len(MV) else []:
        rows[(x.ISIN, x.move_simple)] = dict(p_margin=x.p_margin, line=x.line, margin_log=x.margin_log)
    for x in MC.itertuples() if len(MC) else []:
        rows.setdefault((x.ISIN, x.move_simple), {})["p_mc"] = x.p_mc
    for g, m in r["det"]:
        rows.setdefault((g, m.replace(" (unranked)", "")), {})["predicted"] = True
    recs = []
    for (g, m), v in rows.items():
        pm, pc = v.get("p_margin", np.nan), v.get("p_mc", 0.0)
        primary = np.isfinite(pm)
        prob = pm if primary else pc
        if not (v.get("predicted") or prob >= 0.05 or pc >= 0.05):
            continue
        e = E.loc[g] if g in E.index else None
        recs.append(dict(family="dax", move=m, predicted=bool(v.get("predicted", False)), key=g, isin=g,
                         name=e.Instrument_Name if e is not None else g, symbol=e.symbol if e is not None else None,
                         country="DE", currency="EUR", prob=prob, p_margin=pm, p_mc=pc,
                         prob_basis="margin" if primary else "monte carlo (forced/cascade move)",
                         margin_pct=np.exp(abs(v.get("margin_log", np.nan))) - 1, line=v.get("line"),
                         rank_list=e["rank"] if e is not None else np.nan, rank_t=e.rank_t if e is not None else np.nan,
                         rank_tec_list=e.rank_tec if e is not None else np.nan,
                         rank_tec_t=e.rank_tec_t if e is not None else np.nan,
                         ffmc_t_eur=np.exp(e.logF_t) if e is not None else np.nan,
                         sigma_tot=e.sigma_tot if e is not None else np.nan,
                         F_est=e.F_est if e is not None else np.nan, price_t=e.close_t if e is not None else np.nan,
                         flags=(e["flags"] if e is not None else "") + ("" if primary else "forced;")))
    ch = pd.DataFrame(recs)
    if len(ch):
        ch = ch.sort_values(["predicted", "prob"], ascending=[False, False])
    # watchlist: names within 15% of a rule line, not already listed
    wl_rows = []
    for x in MV.itertuples() if len(MV) else []:
        if abs(x.margin_log) < C.WATCH_MARGIN and x.p_margin < C.P_ENGINE_MIN:
            e = E.loc[x.ISIN]
            wl_rows.append(dict(isin=x.ISIN, name=e.Instrument_Name, move=x.move, line=x.line,
                                margin_pct=np.exp(abs(x.margin_log)) - 1, p_margin=x.p_margin, rank_t=e.rank_t,
                                rank_tec_t=e.rank_tec_t))
    wl = pd.DataFrame(wl_rows).sort_values("p_margin", ascending=False) if wl_rows else pd.DataFrame()
    allE = r["E"].copy()
    allE["family"] = "dax"
    return dict(r=r, all=allE, changes=ch, watch=wl, meta=r["meta"])


# ------------------------------------------------------------------------------------------ FTSE
def do_ftse(D, args, asof, caps, sym) -> dict:
    import model_ftse as M
    notices = load_notices(D["out"])
    r = M.run(D["ftse_u"], D["prices"], sym, D["info"], notices, asof, C.CAL["ftse_uk"]["cutoff"], caps, D["H"],
              n_mc=args.n_mc_dax)
    E = r["E"].set_index("key")
    MV, MC = r["MV"].copy(), r["MC"].copy()
    rows = {}
    for x in MV.itertuples() if len(MV) else []:
        rows[(x.key, x.move)] = dict(p_margin=x.p_margin, line=x.line, margin_log=x.margin_log)
    for x in MC.itertuples() if len(MC) else []:
        rows.setdefault((x.key, x.move_simple), {})["p_mc"] = x.p_mc
    for g, m in r["det"]:
        rows.setdefault((g, m), {})["predicted"] = True
    ind = load_indicative(args.ftse_indicative)
    recs = []
    for (g, m), v in rows.items():
        pm, pc = v.get("p_margin", np.nan), v.get("p_mc", 0.0)
        primary = np.isfinite(pm)
        prob = pm if primary else pc
        e = E.loc[g]
        basis = "margin" if primary else "monte carlo (constant-count move)"
        if ind is not None:
            hit = any(lib_sim(e["name"], n) >= 0.8 or str(e.ticker).upper() == str(tk).upper()
                      for n, tk in zip(ind.name, ind.ticker))
            prob, basis = (0.97, "FTSE indicative list") if hit else (min(prob, 0.10), "not on FTSE indicative list")
        if not (v.get("predicted") or prob >= 0.05 or pc >= 0.05):
            continue
        recs.append(dict(family="ftse_uk", move=m, predicted=bool(v.get("predicted", False)), key=g, isin=e["isin"],
                         name=e["name"], symbol=e.symbol, country="GB", currency=e.ccy_px, prob=prob, p_margin=pm,
                         p_mc=pc, prob_basis=basis, margin_pct=np.exp(abs(v.get("margin_log", np.nan))) - 1,
                         line=v.get("line"), rank_t=e.rank_t, cap_gbp=e.cap_gbp, sigma_tot=e.sigma_tot,
                         F_index=e.F_index, F_est=e.F_est, price_t=e.px_t, shares=e.shares, shares_src=e.shares_src,
                         flags=e["flags"] + ("" if primary else "forced;")))
    ch = pd.DataFrame(recs)
    if len(ch):
        ch = ch.sort_values(["predicted", "prob"], ascending=[False, False])
    wl_rows = []
    for x in MV.itertuples() if len(MV) else []:
        if abs(x.margin_log) < C.WATCH_MARGIN and x.p_margin < C.P_ENGINE_MIN:
            e = E.loc[x.key]
            wl_rows.append(dict(key=x.key, name=e["name"], move=x.move, line=x.line,
                                margin_pct=np.exp(abs(x.margin_log)) - 1, p_margin=x.p_margin, rank_t=e.rank_t,
                                cap_gbp=e.cap_gbp))
    wl = pd.DataFrame(wl_rows).sort_values("p_margin", ascending=False) if wl_rows else pd.DataFrame()
    allE = r["E"].copy()
    allE["family"] = "ftse_uk"
    return dict(r=r, all=allE, changes=ch, watch=wl, meta=r["meta"])


def lib_sim(a, b):
    import universe as U
    return U.sim(a, b)


def load_notices(out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(out.glob("ftse_notices_UK*.json")):
        try:
            j = json.loads(p.read_text())
            rows += (j.get("data") or {}).get("NoticesList") or []
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(rows).drop_duplicates("Id") if rows else pd.DataFrame()


def load_indicative(path):
    """Optional FTSE indicative list (24 Nov): CSV with columns index (FTSE100/FTSE250), side (add/delete), name,
    ticker. Names on it get prob 0.97; predicted moves not on it are capped at 0.10."""
    if not path:
        return None
    x = pd.read_csv(path, dtype=str)
    x.columns = [c.strip().lower() for c in x.columns]
    for c in ("name", "ticker"):
        if c not in x:
            x[c] = ""
    return x


# ------------------------------------------------------------------------------------------ engine rows
def isin_map(D) -> dict:
    """Yahoo symbol -> ISIN from every source we hold (netting key across indices in the engine)."""
    import universe as U
    m = {}
    L = D["lines"]
    for s, i in zip(L.symbol, L["isin"]):
        if isinstance(s, str) and isinstance(i, str) and len(i) == 12:
            m.setdefault(s, i)
    H = D["H"]
    for r in H["EXSA"].itertuples():
        m[U.exsa_symbol(r.ticker, r.exchange)] = r.isin
    for etf in ("ISF", "MIDD"):
        for r in H[etf].itertuples():
            m[U.lse_symbol(r.ticker)] = r.isin
    for etf in ("EXS1", "EXS3", "EXS2"):
        for r in H[etf].itertuples():
            m[f"{r.ticker}.DE"] = r.isin
    return m


def engine_rows(fam: str, ch: pd.DataFrame, asof, tag, run_ts, imap: dict | None = None) -> pd.DataFrame:
    if ch is None or ch.empty:
        return pd.DataFrame()
    imap = imap or {}
    ch = ch.copy()
    ok = ch["isin"].astype(str).str.len().eq(12)
    ch.loc[~ok, "isin"] = ch.loc[~ok, "symbol"].map(imap)
    x = ch[ch.prob >= C.P_ENGINE_MIN].copy()
    rows = []
    src = f"13_shadow/europe_dec2026 {fam} {asof:%Y-%m-%d} {tag}"
    for r in x.itertuples():
        sfx = "." + str(r.symbol).rsplit(".", 1)[-1] if isinstance(r.symbol, str) and "." in str(r.symbol) else ""
        base = dict(isin=r.isin if isinstance(r.isin, str) and len(str(r.isin)) == 12 else np.nan,
                    ticker=str(r.symbol).rsplit(".", 1)[0] if isinstance(r.symbol, str) else None,
                    country=SUFFIX_CTRY.get(sfx, r.country),
                    name=r.name, currency=r.currency if isinstance(r.currency, str) else ("GBP" if fam == "ftse_uk" else "EUR"),
                    price=r.price_t, prob=round(float(r.prob), 3), status="predicted", source=src, source_ts=run_ts)
        note = f"p_margin {r.p_margin:.2f}, p_mc {r.p_mc:.2f}; margin {r.margin_pct:+.1%} to {r.line}" \
            if np.isfinite(r.p_margin) else f"p_mc {r.p_mc:.2f} (forced move)"
        if fam == "stoxx600":
            if r.category == "add":
                rows.append(base | dict(index="STOXX600", change_type="add", old_index_shares=0.0,
                                        new_index_shares=r.F_new, notes=note + "; F = rank-implied free-float shares"))
            else:
                rows.append(base | dict(index="STOXX600", change_type="delete", old_index_shares=r.F_old,
                                        new_index_shares=0.0, notes=note + "; F from EXSA holdings"))
        elif fam == "dax":
            F = r.F_est
            m = r.move
            if m.startswith("TecDAX"):
                rows.append(base | dict(index="TECDAX", change_type="add" if m.endswith("in") else "delete",
                                        old_index_shares=0.0 if m.endswith("in") else F,
                                        new_index_shares=F if m.endswith("in") else 0.0, notes=note))
                continue
            a, b = m.split("->")
            if a != "none" and b != "none":
                rows.append(base | dict(index=a, change_type="migration_out", old_index_shares=F, new_index_shares=0.0,
                                        notes=note + f"; {m}"))
                rows.append(base | dict(index=b, change_type="migration_in", old_index_shares=0.0, new_index_shares=F,
                                        notes=note + f"; {m}"))
            elif a == "none":
                rows.append(base | dict(index=b, change_type="add", old_index_shares=0.0, new_index_shares=F, notes=note))
            else:
                rows.append(base | dict(index=a, change_type="delete", old_index_shares=F, new_index_shares=0.0, notes=note))
        else:
            F = r.F_index if np.isfinite(r.F_index) else r.F_est
            a, b = r.move.split("->")
            if a in ("FTSE100", "FTSE250") and b in ("FTSE100", "FTSE250"):
                rows.append(base | dict(index=a, change_type="migration_out", old_index_shares=F, new_index_shares=0.0,
                                        notes=note + f"; {r.move}"))
                rows.append(base | dict(index=b, change_type="migration_in", old_index_shares=0.0, new_index_shares=F,
                                        notes=note + f"; {r.move}"))
            elif b in ("FTSE100", "FTSE250"):
                rows.append(base | dict(index=b, change_type="add", old_index_shares=0.0, new_index_shares=F,
                                        notes=note + f"; {r.move}; F = shares x Yahoo float ratio"))
            else:
                rows.append(base | dict(index=a, change_type="delete", old_index_shares=F, new_index_shares=0.0,
                                        notes=note + f"; {r.move}"))
    out = pd.DataFrame(rows)
    out.insert(0, "close_date", str(C.TRADE.date()))
    return out


SUFFIX_CTRY = {".L": "GB", ".DE": "DE", ".PA": "FR", ".AS": "NL", ".BR": "BE", ".LS": "PT", ".IR": "IE", ".MI": "IT",
               ".MC": "ES", ".SW": "CH", ".ST": "SE", ".CO": "DK", ".HE": "FI", ".OL": "NO", ".VI": "AT", ".WA": "PL",
               ".AT": "GR"}      # listing country of the traded line (engine netting / volume / close share)
ENGINE_FILE = {"stoxx600": "stoxx600.csv", "dax": "dax_family.csv", "ftse_uk": "ftse_uk.csv"}


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", help="close date (YYYY-MM-DD); default = latest complete close in the price data")
    ap.add_argument("--fetch-date", default=dt.date.today().isoformat())
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--family", default="all", choices=["all"] + C.FAMILIES)
    ap.add_argument("--n-mc", type=int, default=C.N_MC)
    ap.add_argument("--n-mc-dax", type=int, default=2000)
    ap.add_argument("--no-engine", action="store_true", help="do not rewrite the engine input files")
    ap.add_argument("--ftse-indicative", help="CSV of the FTSE indicative list (index, side, name, ticker)")
    args = ap.parse_args()
    run_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    asof_req = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(args.fetch_date)
    D = Dm.prepare(args.fetch_date, asof_req, offline=args.no_fetch)
    W, _ = lib.price_panel(D["prices"])
    sessions = W.index[W.notna().sum(axis=1) > 200]
    asof = sessions[sessions <= asof_req][-1]
    log(f"as-of close {asof.date()} (requested {asof_req.date()})")
    sym = dict(zip(D["lines"].key, D["lines"].symbol))
    caps = index_caps(D["prices"], asof)
    log("index caps (bn):", {k: round(v / 1e9, 1) for k, v in caps.items()})
    fams = C.FAMILIES if args.family == "all" else [args.family]
    stamp = f"{asof:%Y-%m-%d}" + (f"_{args.tag}" if args.tag else "")
    EI, _ = engine_modules()
    imap = isin_map(D)
    retry = sorted((D["out"] / "yahoo").glob("prices_retry_*.jsonl"))
    import re
    _d8 = lambda q: (lambda m: f"{m[:4]}-{m[4:6]}-{m[6:]}")(re.search(r"(\d{8})", q.stem).group(1))
    retry_txt = "".join(f"; plus a targeted Yahoo price retry `{q.relative_to(C.HERE)}` fetched {_d8(q)} "
                        f"with bars ending {args.fetch_date} ({sum(1 for _ in q.open())} symbols)" for q in retry)
    timing = (f"- Timing: this run was made {run_utc:%Y-%m-%d %H:%M} UTC with `--no-fetch` (nothing downloaded by the run). "
              f"Inputs are the downloads saved in `raw/{args.fetch_date}/`, fetched {args.fetch_date}{retry_txt}. "
              f"Latest price bar used: {asof:%Y-%m-%d}." if args.no_fetch else
              f"- Timing: inputs fetched by this run into `raw/{args.fetch_date}/` (run {run_utc:%Y-%m-%d %H:%M} UTC){retry_txt}. "
              f"Latest price bar used: {asof:%Y-%m-%d}.")
    blocks = [f"\n## Run {run_utc:%Y-%m-%d %H:%M:%S} UTC — as-of close {asof:%Y-%m-%d}" + (f" ({args.tag})" if args.tag else ""),
              "", f"- Command: `python3 predict.py {' '.join(sys.argv[1:])}`. Raw inputs: `raw/{args.fetch_date}/`.", timing,
              f"- Index caps used for holdings -> index shares (R2 MV 2026-06-23 x Yahoo level): "
              + ", ".join(f"{k} {v / 1e9:,.0f}bn" for k, v in caps.items()) + "."]
    summaries = {}
    for fam in fams:
        t1 = time.time()
        res = {"stoxx600": do_stoxx, "dax": do_dax, "ftse_uk": do_ftse}[fam](D, args, asof, caps, sym)
        pdir = C.PRED / fam
        pdir.mkdir(parents=True, exist_ok=True)
        hz = horizon_label(asof, C.CAL[fam]["cutoff"])
        if len(res["changes"]):
            res["changes"]["horizon"] = hz
            res["changes"]["backtest"] = [bt_text(fam, r, hz) for r in res["changes"].itertuples()]
        f_all = write_csv(res["all"], pdir / f"pred_{stamp}_all.csv")
        f_ch = write_csv(res["changes"], pdir / f"pred_{stamp}_changes.csv")
        f_wl = write_csv(res["watch"], pdir / f"watchlist_{stamp}.csv")
        ch = res["changes"]
        pp = ch[ch.predicted].copy() if len(ch) else ch
        if len(pp):
            pp = pp[["family", "key", "name", "symbol", "prob", "price_t"] + (["category"] if "category" in pp else ["move"])]
            catc = "category" if "category" in pp else "move"
            pp["side"] = np.where(pp[catc].astype(str).str.contains(r"add|->DAX|->MDAX \(|none->SDAX|TecDAX in|->FTSE100|SmallCap->FTSE250"), 1, -1)
            if fam == "dax":
                up = pp.move.str.contains("TecDAX in") | pp.move.map(lambda m: _dax_up(m))
                pp["side"] = np.where(up, 1, -1)
            if fam == "ftse_uk":
                pp["side"] = np.where(pp.move.str.endswith("FTSE100") | pp.move.str.startswith("SmallCap"), 1, -1)
            pp["asof"] = str(asof.date())
            pp["weight_in_side"] = 1 / pp.groupby("side").key.transform("count")
        f_pp = write_csv(pp, pdir / f"paper_portfolio_{stamp}.csv")
        er = engine_rows(fam, ch, asof, args.tag, run_utc.isoformat(), imap)
        if len(er):
            er = EI.validate(er, fam)
        f_er = write_csv(er, pdir / f"engine_rows_{stamp}.csv")
        meta = res["meta"]
        summ = dict(family=fam, asof=str(asof.date()), tag=args.tag, run_utc=run_utc.isoformat(), horizon=hz,
                    meta=meta, n_changes=int(len(ch)), n_predicted=int(ch.predicted.sum()) if len(ch) else 0,
                    expected_changes=float(ch.prob.sum()) if len(ch) else 0.0, seconds=round(time.time() - t1, 1))
        f_sum = pdir / f"summary_{stamp}.json"
        lib.archive_old(f_sum)
        f_sum.write_text(json.dumps(summ, indent=1, default=str))
        summaries[fam] = summ
        if not args.no_engine:
            C.ENGINE_INPUTS.mkdir(parents=True, exist_ok=True)
            ep = C.ENGINE_INPUTS / ENGINE_FILE[fam]
            (er if len(er) else pd.DataFrame(columns=EI.COLUMNS)).to_csv(ep, index=False)
        files = [f_all, f_ch, f_wl, f_pp, f_er, f_sum]
        blocks += family_block(fam, res, hz, files, ep if not args.no_engine else None)
        log(f"{fam}: {len(ch)} change rows, {summ['n_predicted']} predicted ({summ['seconds']}s)")
    blocks += raw_input_block(D["out"])
    lib.append_log(blocks)
    log("logged to shadow_log.md")


def raw_input_block(out: Path) -> list[str]:
    """SHA-256 of every raw input the run read: everything under raw/<fetch date>/, the cumulative symbol map and the
    hand-entered DAX adjustments (index MVs come read-only from 16_flow_engine pi.index_mv, R2 caches)."""
    files = sorted(q for q in out.rglob("*") if q.is_file() and "superseded" not in q.parts)
    files += [C.RAW / "symbol_map.csv", C.HERE / "adjustments_dax.csv"]
    L = ["", f"### Raw inputs used ({len(files)} files)", "", "| File | Bytes | SHA-256 |", "|---|---|---|"]
    L += [f"| `{q.relative_to(C.HERE)}` | {q.stat().st_size:,} | `{lib.sha(q)}` |" for q in files if q.exists()]
    return L


def _dax_up(m: str) -> bool:
    if m.startswith("TecDAX"):
        return m.endswith("in")
    a, b = m.split("->")
    order = {"none": 0, "SDAX": 1, "MDAX": 2, "DAX": 3}
    return order.get(b, 0) > order.get(a, 0)


def bt_side(fam: str, r) -> tuple[str, str] | None:
    """(index, add|delete) of a call, for the backtest precision lookup."""
    if fam == "stoxx600":
        return "STOXX600", r.category
    if fam == "dax":
        m = r.move
        if m.startswith("TecDAX"):
            return "TECDAX", "add" if m.endswith("in") else "delete"
        a, b = m.split("->")
        return (b, "add") if _dax_up(m) else (a, "delete")
    return None


def bt_text(fam: str, r, hz: str) -> str:
    s = bt_side(fam, r)
    if fam == "stoxx600":
        p, rc = C.BACKTEST[("stoxx600", s[1])][hz]
        return f"backtest precision {p:.0%} ({hz}, STOXX 600 {s[1]}s)"
    if fam == "dax":
        p, rc = C.BACKTEST_DAX_INDEX[s][hz]
        return f"backtest precision {p:.0%} ({hz}, {s[0]} {s[1]}s)"
    return f"backtest precision n/a ({C.BACKTEST_FTSE_NOTE})"


def fmt_row(r, fam, hz=None):
    mv = r.category if fam == "stoxx600" else r.move
    rk = f"rank {r.rank_list:.0f}→{r.rank_t:.0f}" if fam != "ftse_uk" and np.isfinite(getattr(r, "rank_list", np.nan)) \
        else f"rank {r.rank_t:.0f}"
    mg = f"margin {r.margin_pct:+.1%}" if np.isfinite(r.margin_pct) else "forced"
    bt = f"; {bt_text(fam, r, hz)}" if hz else ""
    return f"{r.name} ({mv}; p {r.prob:.2f}, p_mc {r.p_mc:.2f}; {rk}; {mg}{bt})"


def family_block(fam, res, hz, files, engine_path) -> list[str]:
    ch, meta = res["changes"], res["meta"]
    lab = {"stoxx600": "STOXX Europe 600", "dax": "DAX family", "ftse_uk": "FTSE 100/250"}[fam]
    L = ["", f"### {lab}", ""]
    if fam == "stoxx600":
        L.append(f"- Selection list {meta['list_date']} (STOXX Rank FINAL at C0 = {meta['C0']}); as-of {meta['t']}; "
                 f"{meta['h_days']} trading days to the 30 Nov cut-off. Anchors {meta['anchors']} members "
                 f"(Spearman of anchor FFMC vs official rank {meta['anchor_rank_spearman']:.4f}). Add line €{meta['line_add_eur'] / 1e9:.2f}bn, "
                 f"delete line €{meta['line_del_eur'] / 1e9:.2f}bn; members in the 551-750 buffer {meta['buffer_members']} for {meta['buffer_slots']} slots.")
        L.append(f"- Checks: members now {meta['members_now']} (EXSA lines 600); deleted since the list: "
                 f"{', '.join(meta['deleted_since']) or 'none'}; added since: {', '.join(meta['added_since']) or 'none'}; "
                 f"lines without a price {meta['no_price']}. Capping: 20% component cap not binding (EXSA max weight {meta['exsa_max_weight']:.2%}).")
        bt = C.BACKTEST
        L.append(f"- Backtest accuracy at this horizon ({hz}; 12_studies/stoxx600_dax RESULTS §8): adds precision/recall "
                 f"{bt[('stoxx600', 'add')][hz][0]:.0%}/{bt[('stoxx600', 'add')][hz][1]:.0%}, deletes "
                 f"{bt[('stoxx600', 'delete')][hz][0]:.0%}/{bt[('stoxx600', 'delete')][hz][1]:.0%} (without the public list).")
    elif fam == "dax":
        L.append(f"- Selection lists {meta['list_date']} (C0 = {meta['C0']}); as-of {meta['t']}; {meta['h_days']} Xetra days to "
                 f"the 30 Nov cut-off (VWAP-averaged effective horizon {meta['h_eff']:.1f} days). December = Fast Exit/Entry only. "
                 f"Anchors {meta['anchors']} (Spearman vs official rank {meta['anchor_rank_spearman']:.4f}).")
        L.append(f"- Between-review changes applied after the list (adjustments_dax.csv, from stoxx.com/feed/): "
                 f"{'; '.join(meta['adjustments_applied']) or 'none'}.")
        L.append(f"- Checks: list counts {meta['counts_list']}; fund counts {meta['counts_funds']}; unranked members "
                 f"{meta['unranked_members'] or 'none'}. Capping 15%: max weights "
                 + ", ".join(f"{k} {v:.1%}" for k, v in meta["max_weight"].items())
                 + " (TecDAX capping rows already in 16_flow_engine/inputs/2026-12-18/TECDAX.csv; not duplicated). EBITDA assumed positive for DAX candidates.")
    else:
        L.append(f"- Full-cap ranking at {meta['t']}; {meta['h_days']} LSE days to the Tue 1 Dec cut-off. Lines: rank 90 "
                 f"£{meta['lines_gbp']['in100'] / 1e9:.2f}bn, 110/111 £{meta['lines_gbp']['out100'] / 1e9:.2f}bn, 325 "
                 f"£{meta['lines_gbp']['in250'] / 1e9:.2f}bn, 375/376 £{meta['lines_gbp']['out250'] / 1e9:.2f}bn.")
        L.append(f"- Checks: counts {meta['counts']}; deleted by notice (SmallCap) {len(meta['deleted_by_notice'])}; "
                 f"no cap {len(meta['no_cap'])}; shares flagged vs fund {meta['shares_flagged']}. No capping in the headline FTSE 100/250.")
    if len(ch):
        pr = ch[ch.predicted]
        L.append(f"- Predicted changes (deterministic at the as-of close): {len(pr)}; rows with P >= {C.P_ENGINE_MIN}: "
                 f"{int((ch.prob >= C.P_ENGINE_MIN).sum())}; expected count (sum of prob over listed rows) {ch.prob.sum():.1f}.")
        for r in ch.sort_values("prob", ascending=False).head(12).itertuples():
            L.append(f"  - {fmt_row(r, fam, hz)}{' [predicted]' if r.predicted else ''}")
    else:
        L.append("- No change rows.")
    L += ["", "| File | SHA-256 |", "|---|---|"]
    L += [f"| `{p.relative_to(C.HERE) if p.is_relative_to(C.HERE) else p}` | `{lib.sha(p)}` |" for p in files]
    if engine_path is not None:
        L.append(f"| `{engine_path.relative_to(C.PACK)}` (engine input, rewritten each run) | `{lib.sha(engine_path)}` |")
    return L


if __name__ == "__main__":
    main()
