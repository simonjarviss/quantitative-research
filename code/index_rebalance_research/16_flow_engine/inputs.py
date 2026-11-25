"""Input schema (one CSV per index per close: inputs/<close_date>/<INDEX>.csv) and the loaders that write it.

SCHEMA (see README.md "Input schema" for the full text):
  close_date        YYYY-MM-DD, the close at which index funds trade (T)
  index             key of common.INDEX (SPTSX_COMP, SPTSX_60, STOXX600, FTSE100, FTSE250, DAX, MDAX, TECDAX, SDAX,
                    MSCI_EUROPE, MSCI_CANADA, FTSE_AW, NDX, SP500, R2000, R1000)
  change_type       add | delete | share_update | iwf_update | capping | migration_in | migration_out
  isin              ISIN of the traded line (the netting key); blank allowed only with ticker+country
  ticker            local ticker on the listing exchange (TSX 'TECK.B', LSE 'AZN', Xetra 'SIE')
  country           ISO2 listing country of the line traded at the close (CA, GB, DE, FR, ...; US for US lines)
  name, currency    issuer name; trading currency of the line (GBP for LSE lines, not GBp)
  old_index_shares, new_index_shares
                    index (float-adjusted, capped) shares before/after: F_old, F_new. An add has old = 0, a delete
                    new = 0. Preferred form.
  old_weight, new_weight, index_mcap
                    alternative form (fractions; index_mcap in index currency, units). Used when only weights are
                    known (capping, provider weight files): dF = (new_weight - old_weight) * index_mcap / price.
  price             reference price in `currency` (needed for the weight form and for notional)
  prob              probability the change happens (1 for announced/final; model probability for predicted)
  status            predicted | announced | final
  source            where the row came from (file, URL, model run); source_ts = ISO timestamp of the source
  notes             free text
Status precedence in the engine (status='best'): per index and per group (membership = add/delete/migration;
weights = share_update/iwf_update/capping) the highest status present wins: final > announced > predicted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import INDEX, INPUTS, SHADOW_TSX

COLUMNS = ["close_date", "index", "change_type", "isin", "ticker", "country", "name", "currency",
           "old_index_shares", "new_index_shares", "old_weight", "new_weight", "index_mcap", "price", "prob",
           "status", "source", "source_ts", "notes"]
CHANGE_TYPES = {"add", "delete", "share_update", "iwf_update", "capping", "migration_in", "migration_out"}
STATUS_RANK = {"predicted": 1, "announced": 2, "final": 3}
MEMBERSHIP = {"add", "delete", "migration_in", "migration_out"}


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def validate(df: pd.DataFrame, where: str = "") -> pd.DataFrame:
    """Coerce types and check every row; raises ValueError listing the problems."""
    df = df.copy()
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df = df[COLUMNS]
    for c in ["old_index_shares", "new_index_shares", "old_weight", "new_weight", "index_mcap", "price", "prob"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["prob"] = df.prob.fillna(1.0)
    for c in ["index", "change_type", "status", "isin", "ticker", "country", "currency"]:
        df[c] = df[c].astype("string").str.strip()
    errs = []
    bad = ~df["index"].isin(list(INDEX))
    if bad.any():
        errs.append(f"unknown index {sorted(df.loc[bad, 'index'].unique())}")
    bad = ~df.change_type.isin(CHANGE_TYPES)
    if bad.any():
        errs.append(f"unknown change_type {sorted(df.loc[bad, 'change_type'].astype(str).unique())}")
    bad = ~df.status.isin(list(STATUS_RANK))
    if bad.any():
        errs.append(f"unknown status {sorted(df.loc[bad, 'status'].astype(str).unique())}")
    has_sh = df.old_index_shares.notna() & df.new_index_shares.notna()
    has_w = df.old_weight.notna() & df.new_weight.notna() & df.index_mcap.notna() & df.price.notna()
    bad = ~(has_sh | has_w)
    if bad.any():
        errs.append(f"{int(bad.sum())} rows have neither old/new_index_shares nor old/new_weight+index_mcap+price "
                    f"(e.g. {df.loc[bad, 'ticker'].head(3).tolist()})")
    nokey = df["isin"].isna() & (df.ticker.isna() | df.country.isna())
    if nokey.any():
        errs.append(f"{int(nokey.sum())} rows without ISIN or ticker+country")
    if ((df.prob < 0) | (df.prob > 1)).any():
        errs.append("prob outside [0, 1]")
    if errs:
        raise ValueError(f"{where}: " + "; ".join(errs))
    return df


def write(df: pd.DataFrame, close_date: str, index: str, mode: str = "replace") -> Path:
    """Write rows for one index; mode='replace' overwrites the file, 'append' adds rows (dropping exact repeats)."""
    d = INPUTS / str(pd.Timestamp(close_date).date())
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{index}.csv"
    df = validate(df.assign(close_date=str(pd.Timestamp(close_date).date()), index=index), str(p))
    if mode == "append" and p.exists():
        df = pd.concat([pd.read_csv(p, dtype=str), df.astype(str)], ignore_index=True).drop_duplicates()
        df = validate(df, str(p))
    df.to_csv(p, index=False)
    return p


def load(close_date: str) -> pd.DataFrame:
    d = INPUTS / str(pd.Timestamp(close_date).date())
    frames = []
    for p in sorted(d.glob("*.csv")):
        x = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
        if x.empty:
            continue
        frames.append(validate(x, str(p)).assign(_file=p.name))
    return pd.concat(frames, ignore_index=True) if frames else empty().assign(_file=[])


# ------------------------------------------------------------------------------------------------ loaders
def from_holdings_diff(etf: str, d0: str, d1: str, index: str, close_date: str, index_mv: float,
                       status: str = "final", min_rel: float = 0.0025, add_delete_only: bool = False) -> pd.DataFrame:
    """Rows from an iShares holdings diff across the trade close (d0 = T-1, d1 = T; holdings dated T already show
    the rebalance, checked on 19 Jun 2026 for all eight funds). Share/float/capping changes below min_rel are noise."""
    import holdings as H
    x = H.index_share_changes(etf, d0, d1, index_mv=index_mv)
    keep = x.change_type.isin(["add", "delete"]) | (~add_delete_only & (x.change_type == "share_update")
                                                   & (x.rel.abs() >= min_rel))
    x = x[keep].copy()
    ts = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    return pd.DataFrame(dict(
        change_type=x.change_type, isin=x["isin"], ticker=x.ticker, country=x.country, name=x.name,
        currency=x.mkt_ccy.replace({"GBX": "GBP", "GBp": "GBP"}), old_index_shares=x.F0, new_index_shares=x.F1,
        price=x.price1.fillna(x.price0), prob=1.0, status=status,
        source=f"iShares {etf} holdings diff {d0} -> {d1} (k0={x.k0.iloc[0]:.6g}, fund flow {x.flow_ratio.iloc[0]:.4f})"
        if len(x) else "", source_ts=ts, notes=np.where(x.change_type == "share_update",
                                                        [f"dF/F {r:+.2%}" for r in x.rel], "")))


def from_tsx_shadow(run_dir: Path, close_date: str, rules_only_prob: float = 0.10) -> pd.DataFrame:
    """S&P/TSX Composite predicted adds/deletes from a logged shadow run (13_shadow/tsx_dec2026/runs/...).

    F for an add = S&P float shares estimate (float_sh = O/S x IWF proxy); for a delete = XIC-implied float.
    prob = the run's random-walk P(size test passes) (p_size_pass_R) for main-model picks. Rules-only picks the main
    model rejects (committee memory: S&P has passed on BEPC/BIPC at every review since 2025) get rules_only_prob
    (a judgment, stated in notes)."""
    p = pd.read_csv(Path(run_dir) / "predictions.csv")
    p = p[p.status.isin(["pred_add", "pred_delete"])].copy()
    import json
    meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
    rows = []
    for r in p.itertuples():
        is_add = r.status == "pred_add"
        main = bool(r.pred_add_main) if is_add else bool(r.pred_del_main)
        prob = float(r.p_size_pass_R) if is_add else 1.0 - float(r.p_size_pass_R)
        if not np.isfinite(prob):
            prob = 0.5
        note = f"rules margin {r.margin:+.1%}; main model {'yes' if main else 'NO'}"
        if not main:
            prob = rules_only_prob
            note += f"; committee memory rejects -> prob set to {rules_only_prob} (judgment)"
        F = float(r.float_sh)
        rows.append(dict(change_type="add" if is_add else "delete", isin=np.nan, ticker=r.symbol, country="CA",
                         name=r.issuer, currency="CAD", old_index_shares=0.0 if is_add else F,
                         new_index_shares=F if is_add else 0.0, price=float(r.close_asof), prob=round(prob, 3),
                         status="predicted", source=f"13_shadow/tsx_dec2026/{Path(run_dir).relative_to(SHADOW_TSX)}",
                         source_ts=meta.get("run_finished_utc") or meta.get("run_started_utc") or "", notes=note))
    out = pd.DataFrame(rows)
    # ISINs from the run's R2 link table (shadow cache) where available
    try:
        links = pd.read_parquet(SHADOW_TSX / "cache" / "r2_links.parquet")
        links = links.sort_values("last_px").drop_duplicates("symbol", keep="last")
        out["isin"] = out.ticker.map(dict(zip(links.symbol, links["isin"])))
    except Exception:  # noqa: BLE001
        pass
    return out


def capping_rows(h: pd.DataFrame, index: str, cap: float, index_mcap: float, status: str = "predicted",
                 source: str = "") -> pd.DataFrame:
    """Predicted capping flows at the next quarterly review from current holdings weights h (holdings.parse frame).

    Approximation: today's weights are treated as uncapped except names now above the cap, which are cut to the cap
    with the excess redistributed pro rata (iterated). Weight form, so the funding buys in every other member are
    included: demand = pi * index_mcap * (w_new - w_old) / price. Unknown: current cap factors of names capped at the
    last review that have since fallen below the cap (their factor would be reset up)."""
    w = h.set_index("isin").weight.astype(float)
    w = w / w.sum()
    new = w.copy()
    for _ in range(50):
        over = new > cap + 1e-12
        if not over.any():
            break
        excess = (new[over] - cap).sum()
        new[over] = cap
        free = ~over & (new < cap)
        new[free] += excess * new[free] / new[free].sum()
    if (w > cap + 1e-9).sum() == 0:
        return pd.DataFrame(columns=COLUMNS)
    hh = h.set_index("isin")
    ts = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    out = pd.DataFrame(dict(change_type="capping", isin=w.index, ticker=hh.ticker, country=hh.country, name=hh.name,
                            currency=hh.mkt_ccy, old_weight=w.values, new_weight=new.reindex(w.index).values,
                            index_mcap=index_mcap, price=hh.price, prob=1.0, status=status, source=source,
                            source_ts=ts, notes=[f"weight {a:.2%} -> {b:.2%} (cap {cap:.0%})"
                                                 for a, b in zip(w.values, new.reindex(w.index).values)]))
    return out.reset_index(drop=True)
