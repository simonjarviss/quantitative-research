"""C4 without its market tilt: beta-neutral, family-neutral and SPY-hedged variants of the C4 book.

RESEARCH ONLY. NOTHING HERE TOUCHES THE LIVE BOOK. It reads the C4 spec, never writes it; it
imports no order code; it writes only under desk/results/cef/c4_beta_neutral_2026-11-24/ (aggregates) and that
folder's gitignored _cache_* subfolders (daily series, which derive from licensed CRSP returns).

    python3 -B scripts/research/c4_beta_neutral/c4_beta_neutral.py fetch --asof YYYY-MM-DD
        keyless yfinance (the exposure readout's own fetcher): SPY IEF HYG PFF and ^VIX
    python3 -B scripts/research/c4_beta_neutral/c4_beta_neutral.py run \
        --families-dir ~/quantt_data/cef_families --core-dir ~/quantt_data/cef \
        --c4-cache <a trial worktree>/results/cef/c4/_cache_c4
        the backtest, 2005 -> 2026-09-30: tables, splits, regimes, exposures, capacity, chart
    python3 -B scripts/research/c4_beta_neutral/c4_beta_neutral.py live --vm-panel DIR --hist DIR
        the live window (C4 armed 2026-11-03): each variant rebuilt on a copy of the VM's panel
    python3 -B scripts/research/c4_beta_neutral/c4_beta_neutral.py trunc --families-dir ... --c4-cache ...
        harness.truncation_check on the new target builders

WHY (2026-11-24). Over C4's 15 armed sessions
(2026-11-03..11-23) its daily official return had a beta of -0.60 to the equal-weight return of its
own 74 funds, and its nightly exposure readout put EQUITY at -0.44 of equity on 14 of 14 dates
(the caller's measurement from the VM's records; re-measured here, `live`). The book is
dollar-neutral, but its shorts are high-beta equity funds and its longs lower-beta ones, so it is
short the market. The question: does a variant without that tilt
keep C4's gross signal, and what does it do to vol and drawdown?

THE BASE IS THE BOOK PAPER TRADES, NOT A LOOKALIKE. C4's target is CEF trial 54's blend, built by
`results/cef/prereg_c4_level/trial54.py::level_targets` (the trial's line-for-line construction
that DG-L4 matched to the live sleeve on 907 of 907 fit sessions), UNMASKED as the sleeve is, then
m x `band_frontier.band_gross_capped(T, band, cap / m)` with every parameter read from
ops/specs/c4_families.frozen.json. `run` asserts that this base reproduces the H11 spec-book series
(the reference paper is scored against) to 1e-12 where that file is given, and the trial-54 holdout
P&L by construction (same function, same returns). Scored by `harness.evaluate`: shift(2), decide t,
MOC fill t+1, earn t+2. Returns are the trial-54 holdout's: CRSP `ret` to 2024-12-31, then the
yfinance formula on the ruled actions with C4's bar and split rulings and IGR's ruled fill.

THE VARIANTS. Each starts from the base's unit target on every trading row (TB / sum|TB|), which is
the blend before its vol scalar, and changes ONE thing: it is projected onto the null space of a set
of exposures (`neutralise_row` (this file)), renormalised to unit gross and re-scaled by the builder's own
vol scalar (clip(vol_target / rv, 0.2, 2.5), rv on the last 63 price returns through t). So every
variant carries the same ex-ante vol target as the base, and then goes through the same band, cap and
multiplier. A row the base holds (fewer than min_names eligible, or a level book with no gross) is
held by every variant too.
  base           C4 as it trades
  beta_spy       dollar- and SPY-beta-neutral: exposures [1, beta_SPY]
  beta_cef       dollar- and CEF-market-beta-neutral: [1, beta to the equal-weight return of the 74]
  family         dollar-neutral within each of C4's five families (the spec's family_of)
  family_beta    within each family AND SPY-beta-neutral: [family dummies, beta_SPY]
  placebo_s<k>   CONTROL, PERSISTENT (quant-reviewer, 2026-11-24): [1, b], where b gives each day's
                 SORTED eligible betas to the funds in a FIXED random order (seed k, k = 1..8). It has
                 beta's cross-section and beta's persistence, and nothing else of beta. The first
                 version re-shuffled betas across funds every day, which is the wrong control because
                 beta is persistent: its daily-changing direction jittered the target and the band.
                 Whatever beta_spy shares with the placebo seeds is the cost of removing a persistent
                 direction; what it does not share is specific to beta.
  spy_hedge      (c) the base book unchanged, plus SPY at -(the decided book's ex-ante SPY beta).
                 A different instrument, so it is reported on its own line.

BETAS (H7). beta_i(t) = the summed Dimson slopes, lags 0..L, of fund i's daily total return on the
factor's daily total return, OLS with a constant, on the `window` sessions STRICTLY BEFORE t (rows
t-window .. t-1; asserted on every date). L = the exposure readout's C4_DIMSON_LAGS (read at run
time, 2 as of 2026-11-09), the window = C4's z_window (read from the spec, 252), as the readout uses.
SPY is the readout's EQUITY factor: in its Gram-Schmidt order EQUITY is raw SPY and TERM, CREDIT and
PREFERRED are orthogonal to SPY on the window, so a one-factor SPY fit is the readout's EQUITY
loading up to the cross-lag terms. That is checked, not assumed (`check_against_readout`: exact on
the one-factor set, and the 4-factor EQUITY loading reported beside it). The one-factor fit is used
because HYG and PFF start in 2007 and C4 starts in 2005. A name needs BETA_MIN_OBS returns in its
window; otherwise its beta is unmeasured, and a beta variant does not hold it that day (counted
and reported: the share of the base's unit gross on such names). Fund return history for the betas
starts BETA_START (CRSP `ret`, the same source), so the first 2005 decisions have a full window.

REGIMES, NAMED BEFORE ANY RESULT WAS READ, each label built from data through the decision date t
and applied to the P&L that t's decision earns (session t+2):
  VIX        ^VIX close at t above its trailing VIX_MEDIAN_WINDOW-session median through t: stressed
  TREND      SPY's trailing TREND_WINDOW-session total return through t below zero: down market
and, as a diagnostic of the tilt itself (contemporaneous, not a tradeable regime): SPY down days
against up days.

NO COSTS (desk research convention). Gross only. `harness.report`'s cost columns are never read here.
Capacity is the position and the trade against ADV at $1M, $5M and $10M: feasibility, not a cost.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

def _credittrading_root() -> Path:
    """The CreditTrading checkout this study reads (the harness, C4's spec, trial 54's builder).

    WHY: the study moved to this separate research repository on 2026-11-24, so `parents[3]` no longer finds that checkout;
    `desk/README.md` says to name it on PYTHONPATH. No default: a guessed checkout could supply a
    different spec or builder without saying so.
    """
    for entry in sys.path:
        p = Path(entry) if entry else None
        if p and (p / "scripts/research/harness.py").is_file() and (p / "quantt").is_dir():
            return p.resolve()
    raise SystemExit("set PYTHONPATH to a CreditTrading checkout (one holding "
                     "scripts/research/harness.py and quantt/): see desk/README.md")


REPO = _credittrading_root()
pd.set_option("compute.use_numexpr", False)      # the desk's rules, landmine 11

from scripts.research import harness as hz       # noqa: E402


def neutralise_row(w: np.ndarray, X: np.ndarray, on: np.ndarray, *, rank_tol: float = 1e-10) -> np.ndarray:
    """The weights closest to `w` (least squares) that carry zero exposure to every column of `X`,
    held only on the names `on` allows. One decision row; the caller loops over dates.

    LIVES IN THIS STUDY, NOT IN THE SHARED HARNESS (2026-11-24), so CreditTrading's harness is untouched. C4's paper book was measured short the equity market
    (EQUITY -0.44 of equity in its exposure readout) while dollar-neutral, because its shorts
    were the high-beta equity funds. A dollar-neutral builder (`cross_sectional_weights`,
    `band_frontier.build_targets`) cannot remove that: it demeans against a column of ones only.
    This is the general form of that demeaning, so any family can neutralise against any set of
    exposures without a second copy of the arithmetic:

        w' = w_on - X_on (X_on' X_on)^+ X_on' w_on     on the names `on` allows; 0 elsewhere

    which is the orthogonal projection of w onto the null space of X_on'. With X = a column of
    ones it is the cross-sectional demeaning the CEF sleeve already does; with X = family
    dummies it is the demeaning within each family (a family with one name gets zero, as the
    level leg's rule does); with X = [ones, beta] it is dollar- AND beta-neutral.

    THE CALLER OWNS SCALE. The result is not renormalised: projection removes gross, and how
    much gross the book then carries (unit gross, a vol scalar) is the target builder's rule,
    not this function's.

    NO FALLBACKS. A NaN or infinite exposure on a name `on` allows raises: the caller decides,
    and states, which names have no measured exposure (drop them from `on`, or stop), rather
    than this function treating the gap as a zero. A NaN weight raises for the same reason.
    Rank deficiency (two identical exposure columns, a family dummy that equals the ones column)
    is handled by the pseudo-inverse with `rank_tol`, which is exact for a projection: the
    null space of a rank-deficient X' is the null space of its independent columns.

    w   1-D, n names.  X   2-D, n names x k exposures.  on   1-D bool, n names.
    """
    w = np.asarray(w, dtype=float)
    X = np.asarray(X, dtype=float)
    on = np.asarray(on, dtype=bool)
    if w.ndim != 1 or on.shape != w.shape:
        raise ValueError("w and on must be 1-D and the same length")
    if X.ndim != 2 or X.shape[0] != w.shape[0]:
        raise ValueError(f"X must be names x exposures with {w.shape[0]} rows, got {X.shape}")
    if not np.isfinite(w).all():
        raise ValueError("w holds a NaN or infinite weight; every weight must be decided")
    Xo = X[on]
    if not np.isfinite(Xo).all():
        i, j = np.argwhere(~np.isfinite(Xo))[0]
        raise ValueError(f"exposure column {j} is not finite on {int((~np.isfinite(Xo)).any(axis=1).sum())} "
                         f"allowed name(s), first allowed name #{int(np.flatnonzero(on)[i])}; take them out "
                         "of `on` and say so, or stop")
    out = np.zeros_like(w)
    if not on.any():
        return out
    wo = w[on]
    coef = np.linalg.pinv(Xo, rcond=rank_tol) @ wo
    out[on] = wo - Xo @ coef
    return out

# Outputs stay in the private repository, beside the note: desk/results/cef/<study>/.
OUT = Path(__file__).resolve().parents[3] / "results" / "cef" / "c4_beta_neutral_2026-11-24"
FACTOR_CACHE = OUT / "_cache_factors"             # gitignored: results/**/_cache_*
SERIES_CACHE = OUT / "_cache_series"              # gitignored: daily series from CRSP returns
C4_SPEC = REPO / "ops" / "specs" / "c4_families.frozen.json"
TRIAL54 = REPO / "results" / "cef" / "prereg_c4_level" / "trial54.py"

VIX = "^VIX"
FETCH_START = dt.date(2001, 1, 2)
BETA_START = pd.Timestamp("2002-01-02")    # three years before C4's first 2005 decision: room for the 504-session neighbour
BETA_MIN_OBS = 126                         # half a window: fewer returns and the beta is unmeasured
NEIGHBOUR_WINDOWS = (63, 126, 504)         # the beta window's neighbours (the chosen one is the spec's z_window)
HEDGE_CAPS = (0.1, 0.2, 0.3)               # the SPY overlay's cap neighbours: hedge only the ex-ante beta beyond +-c
TILT = 0.2                                 # the base's ex-ante beta state rows: <= -TILT, between, >= +TILT (chosen after
#                                            reading the base's beta distribution, 5th-95th pct -0.31..+0.37, and before any row in these states was computed)
VIX_MEDIAN_WINDOW = 252
TREND_WINDOW = 126
CAPITALS = (1e6, 5e6, 1e7)
PARTICIPATION = (0.05, 0.10)
PLACEBO_SEED = 20261124     # seeds the Sharpe-difference bootstrap and the short-only timing placebo
PLACEBO_SEEDS = tuple(range(1, 9))   # the persistent placebo's eight fixed fund orderings
PLACEBO_KEY_BASE = 1000              # ordering k is default_rng(PLACEBO_KEY_BASE + k).permutation(n), the reviewer's
SHORT_ONLY_SHIFTS = 500              # circular shifts in the short-only overlay's timing placebo
SHARPE_DIFF_BLOCK = 42      # trial 54 P3's calibrate_block pick for C4 books (p3_2026-11-03.json: coverage
#                             0.944-0.950 for blocks 1-42, flat); a K>=1,000 calibration here takes hours
SHARPE_DIFF_M = 1999
MIN_EW_NAMES = 5            # the CEF-market factor needs at least this many fund returns on a session

PROJECTED = ("beta_spy", "beta_cef", "family", "family_beta")
PLACEBOS = tuple(f"placebo_s{k}" for k in PLACEBO_SEEDS)


class InputError(RuntimeError):
    """A required input is missing, stale or not what this script was written for."""


LookaheadError = hz.LookaheadError     # one class for "a row read the future", the harness's


# --------------------------------------------------------------------------- inputs

def _t54():
    """trial54.py, imported from its committed path (it is a results-folder script, not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("trial54_c4bn", TRIAL54)
    if spec is None or spec.loader is None:
        raise InputError(f"cannot import {TRIAL54}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@contextlib.contextmanager
def c4_inputs(families_dir: Path, c4_cache: Path | None):
    """Point C4's trial machinery at a families panel and a cache, read-only, and put it back.

    `c4_families` reads `REPO/data/cef_families` and `REPO/results/cef/c4/_cache_c4`. A worktree has
    neither (data/ is a gitignored symlink in the main checkout; the cache is licensed and
    gitignored). Nothing is copied: the module's paths are pointed at the files where they are, and
    the CRSP extract's sha256 is checked against the one every C4 trial read."""
    from scripts.cef import c4_families as c4
    from scripts.cef.c4_paper_benchmark import CRSP_SHA256
    names = ("FAM_DIR", "FAM_SRC", "CACHE", "CRSP_RET", "ALP_POOL", "ALP_POOL_META")
    saved = {k: getattr(c4, k) for k in names}
    families_dir = Path(families_dir).expanduser().resolve()
    for f in ("cef_prices.parquet", "cef_nav.parquet"):
        if not (families_dir / f).is_file():
            raise InputError(f"{families_dir / f} is missing")
    c4.FAM_DIR = families_dir
    c4.FAM_SRC = (families_dir / "cef_prices.parquet", families_dir / "cef_nav.parquet")
    if c4_cache is not None:
        c4_cache = Path(c4_cache).expanduser().resolve()
        c4.CACHE = c4_cache
        c4.CRSP_RET = c4_cache / saved["CRSP_RET"].name
        c4.ALP_POOL = c4_cache / saved["ALP_POOL"].name
        c4.ALP_POOL_META = c4_cache / saved["ALP_POOL_META"].name
        if not c4.CRSP_RET.is_file():
            raise InputError(f"{c4.CRSP_RET} is missing")
        got = _sha(c4.CRSP_RET)
        if got != CRSP_SHA256:
            raise InputError(f"{c4.CRSP_RET.name}: sha256 {got[:16]}... is not the C4 trials' extract {CRSP_SHA256[:16]}...")
    c4._PAIRED.clear()
    try:
        yield c4
    finally:
        for k, v in saved.items():
            setattr(c4, k, v)
        c4._PAIRED.clear()


def c4_params(t54) -> dict:
    """C4's live parameters, read from its frozen spec at run time (desk rule: the spec is the only
    authority). `level_targets` takes z_window, the ADV floor and min_names from the CORE spec module
    (build_targets' globals) and the risk share from the trial's constant, so each is asserted equal
    to C4's spec value, as trial54.c4_spec does; a mismatch raises."""
    from scripts.cef import spec as core
    from src.deploy import registry
    doc = json.loads(C4_SPEC.read_text())
    f = doc["frozen"]
    lv = registry.cef_level_signal(f)
    if lv is None:
        raise InputError(f"{doc['spec_id']} carries no level_signal; this script was written for the blended C4 book")
    if float(lv["risk_share"]) != float(t54.RISK_SHARE_Z):
        raise InputError(f"spec risk_share {lv['risk_share']} != trial54.RISK_SHARE_Z {t54.RISK_SHARE_Z}")
    pairs = {"z_window": core.Z_WINDOW, "vol_target_annual": core.VOL_TARGET, "min_adv_usd": core.MIN_ADV_USD,
             "min_names": core.MIN_NAMES}
    bad = {k: (f[k], v) for k, v in pairs.items() if float(f[k]) != float(v)}
    if bad:
        raise InputError(f"build_targets reads these from the core spec and they differ from C4's: {bad}")
    if registry.cef_below_min_names(f) != "hold":
        raise InputError("C4's spec no longer holds on thin sessions; level_targets does, so the base would not be C4")
    if f.get("sizing_multiplier") is None or f.get("max_gross_stress") is None:
        raise InputError("C4's spec has no sizing_multiplier or max_gross_stress; the base book needs both")
    return dict(spec_id=doc["spec_id"], spec_sha256=_sha(C4_SPEC), m=float(f["sizing_multiplier"]),
                cap=float(f["max_gross_stress"]), band=float(f["band_width"]), vol=float(f["vol_target_annual"]),
                min_names=int(f["min_names"]), min_adv=float(f["min_adv_usd"]), z_window=int(f["z_window"]),
                universe=list(f["universe"]), family_of=dict(lv["family_of"]))


def build_base(c4, t54, S: dict, end: pd.Timestamp) -> dict:
    """The base's unit target (trial 54's blend, unmasked as the live sleeve is) and everything the
    variants need from the same panel: eligibility, the price returns the vol scalar reads, ADV."""
    from scripts.cef import band_frontier as bf
    U, pool, UF, px, nav, crsp, qr, fam = t54.inputs()
    if sorted(UF) != sorted(S["universe"]):
        raise InputError("the frozen universe CSV differs from the spec's universe")
    fam = fam.reindex(UF)
    if fam.to_dict() != {t: S["family_of"][t] for t in UF}:
        raise InputError("the frozen family CSV differs from the spec's level_signal.family_of")
    # The funds with a paired price and NAV row in this panel. On the full panel that is all 74 (the
    # run asserts nothing changes there: the H11 check). On a panel CUT for the truncation check, funds
    # that list later have no row, so the trial machinery runs on the funds it has and the others
    # enter as zero columns -- what the full build gives them before they list (c4_paper_benchmark's
    # trial_targets does the same). A zero column is a fund with no z, never a filled value.
    have = set(bf.build_panel(universe=UF, source=c4.FAM_SRC)["px"].columns)
    UFd = [t for t in UF if t in have]
    B = t54.level_targets(UFd, fam.reindex(UFd), S, level_mask=None, end=end)
    TB = B["TB"].reindex(columns=UF, fill_value=0.0)
    p = bf.build_panel(universe=UFd, source=c4.FAM_SRC, mask=c4.declared_mask(UFd, "M1M2"))
    idx = TB.index
    z, adv = p["z"].reindex(index=idx, columns=UF), p["adv"].reindex(index=idx, columns=UF)
    # level_targets' own eligibility rule: a z on t and ADV through t-1 at or above the floor
    ok = (z.notna() & (adv.fillna(-1) >= S["min_adv"])).to_numpy()
    retF = p["ret"].reindex(columns=UF)               # a fund with no row has no return (NaN), and no weight
    posF = retF.index.get_indexer(idx)
    if (posF < 0).any():
        raise AssertionError("the target index is not a subset of the panel's own index")
    return dict(U=U, pool=pool, UF=list(UF), px=px, nav=nav, crsp=crsp, qr=qr, fam=fam, TB=TB, ok=ok,
                RF=retF.to_numpy(), posF=posF, adv=adv, panel_px_last=str(p["px"].index.max().date()))


def primary_returns(c4, D: dict, scratch: Path, *, identity: bool = True) -> pd.DataFrame:
    """C4's total returns on the target index, exactly as the trial-54 holdout built them: CRSP `ret`
    to 2024-12-31 (identity-checked), then the yfinance formula on amendment 4's ruled actions with
    C4's bar and split rulings, and IGR's ruled Alpaca fill.

    `identity=False` is for `trunc` only. The identity gate is a share of agreeing days over the
    TRIAL'S windows (2005..2022 and 2023..2024), and on a shorter cut window it reads a different
    share (one fund is 98.76% on 2005..2009-06, under the 99% line, and passes on the full window). It
    feeds no return into the book, so the truncation check, which asks only whether a row moves when
    later data is removed, skips it; `run` always applies it on the trial's own windows."""
    sess = D["TB"].index
    RT = pd.read_csv(c4.RESOLUTION_TABLE, parse_dates=["date"])
    if (RT["resolved"] == "STOP").any():
        raise InputError("amendment 4 carries an unresolved STOP; C4's returns cannot be built")
    excluded = json.loads(c4.EXCLUSIONS_A4.read_text())["excluded"]
    if excluded:
        raise InputError(f"C4's holdout excludes {excluded}; this script was written for an empty list")
    UF, px, crsp, qr = D["UF"], D["px"], D["crsp"], D["qr"]
    if identity:
        c4.crsp_identity(crsp, px, UF, c4.SAMPLE_START, c4.FIT_END)
        c4.crsp_identity(crsp, px, UF, c4.HOLDOUT_START, c4.CRSP_END)
    cols = sorted(set(UF) | set(D["U"]["snap68"]))
    yfa = pd.read_parquet(c4.FAM_DIR / "cef_actions.parquet")
    yfa["date"] = pd.to_datetime(yfa["date"])
    scratch.mkdir(parents=True, exist_ok=True)
    act = scratch / "holdout_actions_ruled.parquet"
    c4.ruled_actions(yfa, RT, excluded).to_parquet(act, index=False)
    cr = c4.crsp_pivot(crsp, cols, sess, qr, "ret", px=px, check_identity=identity)
    if sess[-1] < c4.YF_START:
        # a panel cut before 2025 (`trunc`) has no session on the yfinance leg; the CRSP leg is all of it
        R = cr.loc[:c4.CRSP_END]
        if not R.index.equals(sess):
            raise InputError("the CRSP leg does not cover the session index")
        return R[UF]
    fill = c4.igr_fill(c4.alpaca_pool(), px, sess)[0] if sess[-1] >= max(c4.IGR_FILL) else {}
    yf = c4.yf_total_returns(px, cols, sess, actions=act, rulings=c4.BAR_RULINGS, split_rulings=c4.SPLIT_RULINGS)
    for d, v in fill.items():
        if pd.notna(yf.loc[d, "IGR"]):
            raise InputError(f"IGR {d.date()} already has a yfinance return; the ruled fill does not apply")
        yf.loc[d, "IGR"] = v
    R = pd.concat([cr.loc[:c4.CRSP_END], yf.loc[c4.YF_START:]])
    if not R.index.equals(sess):
        raise InputError("the CRSP and yfinance legs do not tile the session index at the seam")
    return R[UF]


def beta_history(c4, D: dict, R: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fund returns for the beta windows: CRSP `ret` from BETA_START to the day before C4's first
    session, then R itself. The pre-2005 CRSP leg gets the same identity check as the trial's windows;
    a fund that fails it there has its pre-2005 returns removed (named in the output), never booked."""
    UF, px, crsp, qr = D["UF"], D["px"], D["crsp"], D["qr"]
    close = px.pivot_table(index="date", columns="ticker", values="close").sort_index()
    pre = close.index[(close.index >= BETA_START) & (close.index < R.index[0])]
    if not len(pre):
        raise InputError("the families panel has no sessions before C4's first; the beta windows cannot be filled")
    tab = c4.crsp_identity(crsp, px, UF, pre[0], pre[-1], raise_on_fail=False)
    failed = sorted(tab.index[tab["identity"] == "FAIL"])
    cr = c4.crsp_pivot(crsp, UF, pre, qr, "ret", px=px, check_identity=False)
    for t in failed:
        cr[t] = np.nan
    RB = pd.concat([cr, R[UF]])
    if not (RB.index.is_monotonic_increasing and RB.index.is_unique):
        raise AssertionError("the beta history's index is not sorted and unique")
    return RB, dict(pre_first=str(pre[0].date()), pre_sessions=int(len(pre)), identity_failed_pre2005=failed)


def load_factors(sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, dict]:
    """The C4 exposure readout's factor ETFs (SPY IEF HYG PFF) as total returns on `sessions`, by the
    readout's own formula (`exposure_readout.returns_panel`), and ^VIX's close. Each ETF's own session
    rows must be exactly `sessions` over the span it covers: a missing or extra session would put a
    two-day return on one day, so it raises."""
    from scripts.cef import exposure_readout as er
    fs = er.factor_set("c4")
    p = FACTOR_CACHE / "etf_px.parquet"
    v = FACTOR_CACHE / "vix_px.parquet"
    m = FACTOR_CACHE / "meta.json"
    for f in (p, v, m):
        if not f.is_file():
            raise InputError(f"{f} is missing: run `c4_beta_neutral.py fetch` first")
    px = pd.read_parquet(p)
    px["date"] = pd.to_datetime(px["date"])
    meta = json.loads(m.read_text())
    Rf, C, _ = er.returns_panel(px)
    lo, hi = sessions[0], sessions[-1]
    for tk in fs.etfs:
        if tk not in C.columns:
            raise InputError(f"factor ETF {tk} is not in {p}")
        d = C[tk].dropna().index
        a = max(lo, d[0])
        if d[-1] < hi:
            raise InputError(f"{tk}'s cache ends {d[-1].date()}, before the last session needed {hi.date()}")
        want = sessions[(sessions >= a) & (sessions <= hi)]
        got = d[(d >= a) & (d <= hi)]
        if not want.equals(got):
            raise InputError(f"{tk}: cache sessions differ from the C4 sessions on {a.date()}..{hi.date()}: "
                             f"missing {[str(x.date()) for x in want.difference(got)[:5]]}, "
                             f"extra {[str(x.date()) for x in got.difference(want)[:5]]}")
    vx = pd.read_parquet(v)
    vx["date"] = pd.to_datetime(vx["date"])
    vix = vx[vx["ticker"] == VIX].set_index("date")["close"].sort_index()
    if not np.isfinite(vix.to_numpy()).all() or (vix <= 0).any():
        raise InputError("^VIX has a non-finite or non-positive close")
    return Rf.reindex(sessions)[list(fs.etfs)], vix, meta


# --------------------------------------------------------------------------- betas (H7)

def dimson_betas(Y: pd.DataFrame, f: pd.Series, decide: pd.DatetimeIndex, *, window: int, lags: int,
                 min_obs: int) -> pd.DataFrame:
    """beta_i(t) for every t in `decide`: the summed slopes on f_s, f_{s-1}, .., f_{s-lags} of an OLS of
    Y_i,s with a constant, over the `window` rows of Y's index STRICTLY BEFORE t (t-window .. t-1).

    A name with every return in the window is fitted in one solve; a name with at least `min_obs`
    returns is fitted on its own rows; fewer, and its beta is NaN (unmeasured), which the caller must
    handle by name. A date with fewer than `window` earlier rows is unmeasured for every name. A
    missing FACTOR return inside a window raises: the factor is the yardstick, and a gap in it is not
    the name's problem to absorb."""
    from scripts.cef.exposure_readout import assert_strictly_before
    if not f.index.equals(Y.index):
        raise ValueError("the factor and the returns must share one index")
    idx = Y.index
    F = np.column_stack([np.ones(len(idx))] + [f.shift(l).to_numpy(float) for l in range(lags + 1)])
    Yv = Y.to_numpy(float)
    pos = idx.get_indexer(decide)
    if (pos < 0).any():
        raise ValueError("a decision date is not a row of the returns' index")
    fin = np.isfinite(F).all(axis=1)
    if not fin.any():
        raise InputError("the factor has no finite return at all")
    first = int(np.argmax(fin))          # the factor's history starts here (its first return and lags exist)
    out = np.full((len(decide), Y.shape[1]), np.nan)
    for k, i in enumerate(pos):
        lo = i - window
        if lo < first:
            continue                     # not enough factor history yet: unmeasured, not a gap
        rows = idx[lo:i]
        assert_strictly_before(rows, idx[i], "beta window")          # H7, every date
        # The DATA is taken through the positions of the labels just asserted, never through a second
        # slice expression, so the rows that reach the fit are the rows that passed the assert
        # (quant-reviewer, 2026-11-24: assert on the data, not only on the date labels). The factor
        # rows' own lags reach lo-lags .. i-1-lags, all before t as well.
        rp = idx.get_indexer(rows)
        if len(rp) != window or (rp < 0).any() or rp.max() >= i:
            raise LookaheadError(f"the beta window for {idx[i].date()} reaches row {rp.max()} of {i}: it "
                                 "must end strictly before the decision row (H7)")
        X = F[rp]
        if not np.isfinite(X).all():
            raise InputError(f"the factor return is missing inside the beta window for {idx[i].date()}")
        y = Yv[rp]
        okc = np.isfinite(y)
        full = okc.all(axis=0)
        if full.any():
            B = np.linalg.lstsq(X, y[:, full], rcond=None)[0]
            out[k, full] = B[1:].sum(axis=0)
        for j in np.flatnonzero(~full & (okc.sum(axis=0) >= min_obs)):
            mm = okc[:, j]
            b = np.linalg.lstsq(X[mm], y[mm, j], rcond=None)[0]
            out[k, j] = b[1:].sum()
    return pd.DataFrame(out, index=decide, columns=Y.columns)


def check_against_readout(RB: pd.DataFrame, Rf: pd.DataFrame, betas: pd.DataFrame, probes, *, window: int,
                          lags: int) -> dict:
    """Second method for the betas: the exposure readout's own `fit_window`, on (a) a one-factor set
    (EQUITY = SPY), which must agree to 1e-10 on names with a complete window, and (b) the readout's
    full C4 set (SPY IEF HYG PFF, Gram-Schmidt), whose EQUITY loading is reported beside it."""
    from scripts.cef import exposure_readout as er
    fs4 = er.factor_set("c4")
    fs1 = er.FactorSet("c4_spy_only", (("EQUITY", "SPY"),), "EQUITY", fs4.spec_id, fs4.vol_target, window,
                       fs4.universe, lags, "family", fs4.sizing_multiplier)
    W = pd.concat([RB, Rf], axis=1)
    out = []
    for t in probes:
        t = pd.Timestamp(t)
        i = W.index.get_loc(t)
        win = W.iloc[i - window:i]
        names = [n for n in RB.columns if win[n].notna().all()]
        f1 = er.fit_window(W, W.index, t, names, window=window, lags=lags, fs=fs1)
        d1 = (f1.beta_sum["EQUITY"] - betas.loc[t, names]).abs().max()
        if not d1 < 1e-10:
            raise AssertionError(f"dimson_betas differs from exposure_readout.fit_window (SPY only) on {t.date()} by {d1}")
        row = dict(date=str(t.date()), names=len(names), max_abs_diff_one_factor=float(d1))
        if win[list(fs4.etfs)].notna().all().all() and W.iloc[i - window - lags:i][list(fs4.etfs)].notna().all().all():
            f4 = er.fit_window(W, W.index, t, names, window=window, lags=lags, fs=fs4)
            e4 = f4.beta_sum["EQUITY"]
            b1 = betas.loc[t, names]
            row.update(four_factor_equity_minus_spy_only_mean=float((e4 - b1).mean()),
                       four_factor_equity_minus_spy_only_mad=float((e4 - b1).abs().median()),
                       corr=float(np.corrcoef(e4, b1)[0, 1]))
        out.append(row)
    return dict(probes=out)


# --------------------------------------------------------------------------- the variants

def hold_rows(TB: pd.DataFrame, ok: np.ndarray, min_names: int) -> np.ndarray:
    """The rows the base HOLDS (copies the previous target): fewer than min_names eligible, or a level
    book with no gross or a non-positive leg vol (level_targets' holds). A trading row never equals
    the previous one exactly (the vol scalar moves daily), so exact equality identifies the latter."""
    A = TB.to_numpy()
    same = np.zeros(len(A), bool)
    same[1:] = (A[1:] == A[:-1]).all(axis=1)
    thin = ok.sum(axis=1) < min_names
    if (thin[1:] & ~same[1:]).any():
        raise AssertionError("a thin row of the base is not a copy of the previous row; level_targets changed")
    hold = thin | same
    hold[0] = bool(thin[0])
    return hold


def variant_targets(D: dict, S: dict, hold: np.ndarray, exposures_of, label: str) -> tuple[pd.DataFrame, dict]:
    """One variant's unit target. On each trading row: the base's unit-gross blend, projected by
    `neutralise_row` (this file) onto zero exposure to `exposures_of(i)` on the names it allows,
    renormalised to unit gross, times the builder's vol scalar on its own trailing 63 price returns.
    A held row holds the variant's previous row. A row whose projection leaves no gross holds too,
    and is counted (`no_gross_rows`)."""
    from scripts.cef import band_frontier as bf
    t54 = D["t54"]
    TB, ok, RF, posF = D["TB"], D["ok"], D["RF"], D["posF"]
    A = TB.to_numpy()
    TV = np.zeros_like(A)
    dropped, no_gross, active = [], 0, 0
    for i in range(len(A)):
        if hold[i]:
            TV[i] = TV[i - 1] if i else 0.0
            continue
        g = np.abs(A[i]).sum()
        if not g > 0:
            raise AssertionError(f"{label}: trading row {TB.index[i].date()} has no gross in the base")
        w = A[i] / g
        if (w[~ok[i]] != 0).any():
            raise AssertionError(f"{label}: the base holds an ineligible name on {TB.index[i].date()}")
        X, on = exposures_of(i)
        dropped.append(float(np.abs(w[ok[i] & ~on]).sum()))
        w2 = neutralise_row(w, X, on)
        g2 = np.abs(w2).sum()
        if not g2 > 1e-12:
            TV[i] = TV[i - 1] if i else 0.0
            no_gross += 1
            continue
        w2 = w2 / g2
        rv = t54._rv(None, RF, posF[i], w2, bf.VOL_LOOKBACK)
        sc = float(np.clip(S["vol"] / rv, 0.2, 2.5)) if rv and rv > 0 else 1.0     # level_targets' `sc`, as written
        TV[i] = w2 * sc
        active += 1
    d = np.asarray(dropped)
    return (pd.DataFrame(TV, index=TB.index, columns=TB.columns),
            dict(trading_rows=active, no_gross_rows=no_gross,
                 unit_gross_dropped_for_no_beta_mean=float(d.mean()) if len(d) else 0.0,
                 unit_gross_dropped_for_no_beta_max=float(d.max()) if len(d) else 0.0))


def exposure_makers(D: dict, betas: dict, families: list):
    """The exposure set of each projected variant, as a function of the row: (X, on)."""
    ok = D["ok"]
    n = len(D["UF"])
    ones = np.ones(n)
    famv = D["fam"].to_numpy()
    dummies = np.column_stack([(famv == g).astype(float) for g in families])
    Bs, Bc = betas["spy"].to_numpy(), betas["cef"].to_numpy()

    def beta_spy(i):
        on = ok[i] & np.isfinite(Bs[i])
        return np.column_stack([ones, Bs[i]]), on

    def beta_cef(i):
        on = ok[i] & np.isfinite(Bc[i])
        return np.column_stack([ones, Bc[i]]), on

    def family(i):
        return dummies, ok[i].copy()

    def family_beta(i):
        on = ok[i] & np.isfinite(Bs[i])
        return np.column_stack([dummies, Bs[i]]), on

    def persistent_placebo(seed):
        """A fixed random ordering of the 74 funds receives each day's sorted eligible betas: the
        lowest-ranked eligible fund gets the day's lowest beta, and so on. Persistent like beta, with
        beta's cross-section, and unrelated to which fund has which beta (quant-reviewer, 2026-11-24;
        the reviewer's construction, `scratchpad/rev/placebo2.py`)."""
        key = np.random.default_rng(PLACEBO_KEY_BASE + seed).permutation(n)

        def placebo(i):
            on = ok[i] & np.isfinite(Bs[i])
            k = np.flatnonzero(on)
            b = np.zeros(n)                       # off `on` neutralise_row never reads it
            b[k[np.argsort(key[k])]] = np.sort(Bs[i][k])
            return np.column_stack([ones, b]), on
        return placebo

    out = dict(beta_spy=beta_spy, beta_cef=beta_cef, family=family, family_beta=family_beta)
    out.update({f"placebo_s{k}": persistent_placebo(k) for k in PLACEBO_SEEDS})
    return out


def held_book(T: pd.DataFrame, S: dict) -> pd.DataFrame:
    """What C4's account holds: m x band_gross_capped(unit target, band, cap / m) (trial54.book)."""
    from scripts.cef import band_frontier as bf
    return S["m"] * bf.band_gross_capped(T, S["band"], S["cap"] / S["m"])


def book_beta(H: pd.DataFrame, B: pd.DataFrame) -> tuple[pd.Series, int]:
    """Ex-ante beta of the decided book: sum_i H[t, i] beta_i(t), and the count of held name-days with
    no measured beta (they are left out of the sum, and the count is reported; the hedge refuses them)."""
    Hv = H.to_numpy()
    Bv = B.reindex(index=H.index, columns=H.columns).to_numpy()
    held = Hv != 0
    miss = held & ~np.isfinite(Bv)
    return pd.Series(np.where(held & np.isfinite(Bv), Hv * Bv, 0.0).sum(axis=1), index=H.index), int(miss.sum())


def hedge_weight(beta: pd.Series, rule) -> pd.Series:
    """The SPY weight for a book whose ex-ante SPY beta is `beta`, under one hedging rule:
      "full"         -beta: the book's ex-ante beta to zero (the pre-specified overlay, (c));
      ("cap", c)     -(beta - clip(beta, -c, c)): only the part beyond +-c, so the book's ex-ante
                     |beta| is at most c (a NEIGHBOUR, added after the first read showed the base's
                     beta swinging between about -0.3 and +0.4);
      "short_only"   -min(beta, 0): only a SHORT tilt is hedged; a long tilt keeps the equity premium
                     (NOT pre-specified: added after the first read showed the hedge's cost is the
                     premium the base's average long tilt earned)."""
    if rule == "full":
        return -beta
    if rule == "short_only":
        return -beta.clip(upper=0.0)
    if isinstance(rule, tuple) and rule[0] == "cap" and rule[1] > 0:
        return -(beta - beta.clip(-rule[1], rule[1]))
    raise ValueError(f"unknown hedge rule {rule!r}")


def spy_overlay(H: pd.DataFrame, B: pd.DataFrame, *, rule="full", neighbour: bool = False) -> pd.DataFrame:
    """(c) the base book plus SPY at `hedge_weight(its ex-ante SPY beta, rule)`, decided at t like the
    book (so it is held over t+2 by the same harness lag). A held name with no measured beta raises: a
    hedge sized on a partial book would be a guess about the rest.

    `neighbour=True` is for the beta-window NEIGHBOUR rows only (never the headline): a 504-session
    window cannot be filled for funds listed in 2004 when the book first holds them in 2005-06, so
    there the hedge covers the measured names only, and the count of held name-days left out is
    stored on the frame (`attrs["held_name_days_no_beta"]`) and printed beside the row."""
    beta, miss = book_beta(H, B)
    if miss and not neighbour:
        Hv, Bv = H.to_numpy(), B.reindex(index=H.index, columns=H.columns).to_numpy()
        i, j = np.argwhere((Hv != 0) & ~np.isfinite(Bv))[0]
        raise InputError(f"{miss} held name-day(s) have no measured SPY beta, first {H.index[i].date()} {H.columns[j]}")
    out = H.copy()
    out["SPY"] = hedge_weight(beta, rule)
    out.attrs["held_name_days_no_beta"] = miss
    return out


# --------------------------------------------------------------------------- statistics

def _dimson_ols(y: pd.Series, f: pd.Series, lags: int) -> dict:
    """Realised beta: summed Dimson slopes of a daily P&L on a factor, OLS with HAC(5) errors."""
    import statsmodels.api as sm
    X = pd.concat({l: f.shift(l) for l in range(lags + 1)}, axis=1)
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(d) < 30:
        return dict(beta=np.nan, t=np.nan, n=len(d))
    r = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    c = np.ones(lags + 1)
    V = r.cov_params().iloc[1:, 1:].to_numpy()
    b = float(r.params.iloc[1:].sum())
    return dict(beta=b, t=float(b / np.sqrt(c @ V @ c)), n=int(len(d)))


def stats_row(P: pd.Series, TU: pd.Series, G: pd.Series, f_spy: pd.Series, f_ew: pd.Series, lags: int,
              exante: pd.Series | None = None) -> dict:
    """The row every table shows: gross only."""
    from scripts.research.inference import newey_west_t
    if P.isna().any():
        raise ValueError("a P&L slice holds NaN")
    ann, vol = P.mean() * 252, P.std() * np.sqrt(252)
    eq = (1 + P).cumprod()
    act = P[P != 0]
    out = dict(start=str(P.index[0].date()), end=str(P.index[-1].date()), days=int(len(P)),
               ann_pct=float(ann * 100), vol_pct=float(vol * 100), sr=float(ann / vol) if vol > 0 else np.nan,
               nw_t=float(newey_west_t(P.to_numpy())), max_dd_pct=float((eq / eq.cummax() - 1).min() * 100),
               hit=float((act > 0).mean()) if len(act) else np.nan, turn=float(TU.mean() * 252),
               gross=float(G.mean()))
    bs = _dimson_ols(P, f_spy.reindex(P.index), lags)
    be = _dimson_ols(P, f_ew.reindex(P.index), lags)
    out.update(beta_spy=bs["beta"], beta_spy_t=bs["t"], beta_cef=be["beta"], beta_cef_t=be["t"])
    if exante is not None:
        x = exante.reindex(P.index).dropna()
        out.update(exante_beta_mean=float(x.mean()), exante_beta_abs_mean=float(x.abs().mean()))
    return out


def factor_exposures(P: pd.Series, Rf: pd.DataFrame, lags: int) -> dict:
    """Equity, rates, credit and preferred betas of a daily P&L: the C4 readout's factor set
    (exposure_readout.factor_set('c4'): EQUITY=SPY, TERM=IEF, CREDIT=HYG, PREFERRED=PFF, Gram-Schmidt in
    that order, orthogonalised once on this sample), summed Dimson lags 0..lags, OLS with HAC(5)."""
    import statsmodels.api as sm
    from scripts.cef import exposure_readout as er
    fs = er.factor_set("c4")
    E = Rf[list(fs.etfs)].reindex(P.index)
    first = E.dropna().index[0]
    E = E.loc[first:]
    M = er.orthogonaliser(E.iloc[lags:], fs)
    F = E @ M
    F.columns = list(fs.names)
    cols = {(k, l): F[k].shift(l) for k in fs.names for l in range(lags + 1)}
    X = pd.concat(cols, axis=1)
    d = pd.concat([P.loc[first:].rename("y"), X], axis=1).dropna()
    r = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    out = dict(start=str(d.index[0].date()), end=str(d.index[-1].date()), n=int(len(d)),
               alpha_ann_pct=float(r.params["const"] * 252 * 100), alpha_t=float(r.tvalues["const"]),
               r2=float(r.rsquared))
    V = r.cov_params()
    for k in fs.names:
        keys = [c for c in r.params.index if isinstance(c, tuple) and c[0] == k]
        b = float(r.params[keys].sum())
        v = float(V.loc[keys, keys].to_numpy().sum())
        out[k] = dict(beta=b, t=b / np.sqrt(v))
    return out


def regime_labels(sessions: pd.DatetimeIndex, Rspy: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Labels on DECISION dates (data through t), then moved forward EXEC_LAG sessions so each sits on
    the P&L row its decision earns. Built only from the past: rolling windows end at t."""
    v = vix.reindex(sessions)
    if v.isna().any():
        missing = v.index[v.isna()]
        raise InputError(f"^VIX has no close on {len(missing)} session(s), first {missing[0].date()}")
    vix_hi = v > v.rolling(VIX_MEDIAN_WINDOW, min_periods=VIX_MEDIAN_WINDOW).median()
    vix_ok = v.rolling(VIX_MEDIAN_WINDOW, min_periods=VIX_MEDIAN_WINDOW).median().notna()
    lr = np.log1p(Rspy.reindex(sessions))
    trail = lr.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).sum()
    down = trail < 0
    lab = pd.DataFrame({"vix_stressed": vix_hi.where(vix_ok), "trend_down": down.where(trail.notna())})
    return lab.shift(hz.EXEC_LAG)


def capacity(H: pd.DataFrame, adv: pd.DataFrame, a, b) -> dict:
    """Position and trade size against ADV (USD, mean close x volume over 63 sessions through t-1, the
    sleeve's own ADV) at each capital. H is in fractions of capital, as the account holds it."""
    Hw = H.loc[a:b]
    A = adv.reindex(index=Hw.index, columns=Hw.columns)
    held = Hw.ne(0.0)
    unmeasured = int((held & ~(A > 0)).sum().sum())
    T = Hw.diff().abs()
    traded = T.gt(0)
    out = dict(window=[str(Hw.index[0].date()), str(Hw.index[-1].date())], held_name_days=int(held.sum().sum()),
               held_name_days_adv_unmeasured=unmeasured, names_held_mean=float(held.sum(axis=1).mean()),
               largest_weight=float(Hw.abs().max().max()))
    okA = A.where(A > 0)
    for C in CAPITALS:
        pos = (Hw.abs() * C / okA)[held].stack().dropna()
        trd = (T * C / okA)[traded].stack().dropna()
        out[f"${C / 1e6:.0f}M"] = dict(
            pos_max_pct_adv=float(pos.max() * 100), pos_p99_pct_adv=float(pos.quantile(0.99) * 100),
            **{f"pos_share_above_{int(p * 100)}pct": float((pos > p).mean()) for p in PARTICIPATION},
            trade_max_pct_adv=float(trd.max() * 100) if len(trd) else 0.0,
            trade_p99_pct_adv=float(trd.quantile(0.99) * 100) if len(trd) else 0.0)
    return out


def sharpe_vs_base(P: pd.Series, Pb: pd.Series) -> dict:
    from scripts.research.resampling import sharpe_diff
    out = {}
    for blk in (SHARPE_DIFF_BLOCK, 21):
        r = sharpe_diff(P.to_numpy(), Pb.to_numpy(), block=blk, M=SHARPE_DIFF_M, seed=PLACEBO_SEED)
        out[f"block_{blk}"] = dict(delta_annual=float(r["delta_annual"]), p_value=float(r["p_value"]),
                                   ci_annual=[float(x) for x in r["ci_annual"]])
    return out


def placebo_summary(table: dict) -> dict:
    """The persistent placebo across its seeds: the range and mean of the rows the note reads."""
    rows = {k: table[k] for k in PLACEBOS}
    out = {"seeds": list(PLACEBO_SEEDS), "key_base": PLACEBO_KEY_BASE}
    for lab, get in (("full_sr", lambda r: r["full"]["sr"]), ("full_max_dd_pct", lambda r: r["full"]["max_dd_pct"]),
                     ("late_sr", lambda r: r["late"]["sr"]), ("early_sr", lambda r: r["early"]["sr"]),
                     ("full_beta_spy", lambda r: r["full"]["beta_spy"]), ("turn", lambda r: r["full"]["turn"]),
                     ("tilt_short_sr", lambda r: r["base_tilt_short"]["sr"]),
                     ("tilt_long_sr", lambda r: r["base_tilt_long"]["sr"])):
        v = np.array([get(r) for r in rows.values()])
        out[lab] = dict(min=float(v.min()), mean=float(v.mean()), max=float(v.max()),
                        by_seed={k: float(x) for k, x in zip(rows, v)})
    return out


def short_only_controls(res: dict, exante: dict, spy: pd.Series, p0) -> dict:
    """Is the short-only overlay's edge over the base its TIMING, or just a long SPY position on average?
    (quant-reviewer, 2026-11-24; `scratchpad/rev/short_only_ctrl.py`.) Its SPY weight is
    -min(base ex-ante beta, 0): never short SPY, so on average long. Two controls on the same P&L rows:
      constant   the base plus SPY held at the short-only overlay's MEAN held weight every day;
      timing     the base plus the short-only weight series circularly shifted by a random offset of at
                 least a year (it keeps the series' persistence and level, and loses its timing)."""
    base = res["base"]["pnl"].loc[p0:]
    so = res["spy_short_only"]["pnl"].loc[p0:]
    beta = exante["base"]["series"]
    w = hedge_weight(beta, "short_only")                     # decided at t
    wh = hz.held(w.to_frame("SPY"))["SPY"].reindex(base.index)   # held over t+2, as the harness books it
    r = spy.reindex(base.index)
    if (r.isna() & wh.ne(0)).any():
        raise InputError("SPY has no return on a session the short-only weight is held")
    r = r.fillna(0.0)
    d = float((base + wh * r - so).abs().max())
    if not d < 1e-12:
        raise AssertionError(f"the short-only overlay is not base + held weight x SPY (max |diff| {d})")
    sr = lambda x: float(x.mean() / x.std() * np.sqrt(252))
    c = float(wh.mean())
    rng = np.random.default_rng(PLACEBO_SEED)
    whv, rv, bv = wh.to_numpy(), r.to_numpy(), base.to_numpy()
    shifted = np.array([sr(pd.Series(bv + np.roll(whv, int(rng.integers(252, len(whv) - 252))) * rv))
                        for _ in range(SHORT_ONLY_SHIFTS)])
    return dict(definition="SPY bought at minus the base book's ex-ante SPY beta on days that beta is negative; "
                           "no SPY on days it is zero or positive",
                mean_held_spy_weight=c, share_of_days_hedged=float((wh > 0).mean()),
                sr_base=sr(base), sr_short_only=sr(so), sr_base_plus_constant_spy=sr(base + c * r),
                timing_placebo=dict(shifts=SHORT_ONLY_SHIFTS, seed=PLACEBO_SEED, mean_sr=float(shifted.mean()),
                                    p95_sr=float(np.quantile(shifted, 0.95)),
                                    share_at_least_short_only=float((shifted >= sr(so)).mean())))


def tilt_uncertainty(P: pd.Series, tilt: pd.Series) -> dict:
    """How sure is "C4 earns little when it is tilted short"? The short-state days come in a few long
    episodes, so they are not independent: the mean difference is tested with SEs clustered by episode
    (`inference.clustered_mean_diff`; an episode is a run of consecutive sessions in one state), and
    again with 21-session calendar blocks as clusters; the state's Sharpe gets an episode bootstrap."""
    from scripts.research.inference import clustered_mean_diff
    m = tilt.notna()
    y, st = P[m], (tilt[m] <= -TILT)
    ep = (st != st.shift()).cumsum().to_numpy()
    out = {}
    for lab, cl in (("by_episode", ep), ("by_21_session_block", np.arange(len(y)) // 21)):
        r = clustered_mean_diff(y.to_numpy(), st.to_numpy(), cl)
        out[lab] = dict(short_ann_pct=r["mean_a"] * 252 * 100, short_se_ann_pct=r["se_a"] * 252 * 100,
                        rest_ann_pct=r["mean_b"] * 252 * 100, diff_ann_pct=r["diff"] * 252 * 100,
                        diff_se_ann_pct=r["se_diff"] * 252 * 100, t=r["diff"] / r["se_diff"],
                        short_clusters=r["clusters_a"], rest_clusters=r["clusters_b"], n_short=r["n_a"])
    eps = [y.to_numpy()[(ep == e) & st.to_numpy()] for e in np.unique(ep[st.to_numpy()])]
    rng = np.random.default_rng(PLACEBO_SEED)
    srs = []
    for _ in range(2000):
        x = np.concatenate([eps[j] for j in rng.integers(0, len(eps), len(eps))])
        srs.append(x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan)
    srs = np.array(srs)
    x = np.concatenate(eps)
    out["short_state_sr"] = dict(sr=float(x.mean() / x.std() * np.sqrt(252)), episodes=len(eps),
                                 episodes_10_or_more_sessions=int(sum(len(e) >= 10 for e in eps)),
                                 bootstrap_by_episode_p05=float(np.nanquantile(srs, 0.05)),
                                 bootstrap_by_episode_p95=float(np.nanquantile(srs, 0.95)))
    return out


# --------------------------------------------------------------------------- the core book (correlation)

@contextlib.contextmanager
def core_inputs(core_dir: Path):
    from scripts.cef import band_frontier as bf
    core_dir = Path(core_dir).expanduser().resolve()
    for f in ("cef_prices.parquet", "cef_nav.parquet", "cef_distributions.parquet"):
        if not (core_dir / f).is_file():
            raise InputError(f"{core_dir / f} is missing")
    saved = bf.DATA_DIR
    bf.DATA_DIR = core_dir
    try:
        yield bf
    finally:
        bf.DATA_DIR = saved


def core_book(core_dir: Path) -> tuple[pd.Series, dict]:
    """The core credit book as it trades (flatten on thin sessions; m x the unit book at cap / m), scored
    on total returns cut at the distribution panel's last ex-date, as the thin-days report did."""
    from scripts.cef.spec import BAND_WIDTH, MAX_GROSS_STRESS, SIZING_MULTIPLIER, SPEC_ID, UNIVERSE
    with core_inputs(core_dir) as bf:
        T, R = bf.build_targets()
        thin = bf.thin_sessions(T.index)
        m = 1.0 if SIZING_MULTIPLIER is None else float(SIZING_MULTIPLIER)
        H = m * bf.band_gross_capped(T, BAND_WIDTH, MAX_GROSS_STRESS / m, below_min_names="flatten", thin=thin)
        dist = pd.read_parquet(bf.DATA_DIR / "cef_distributions.parquet")
        dist["ex_date"] = pd.to_datetime(dist["ex_date"])
        dist_end = dist.loc[dist["ticker"].isin(UNIVERSE), "ex_date"].max()
        keep = T.index <= dist_end
        Rt = bf.total_returns(R.loc[keep])
        res = hz.evaluate(H.loc[keep], Rt.reindex(columns=H.columns))
    return res["pnl"], dict(spec_id=SPEC_ID, m=m, band=BAND_WIDTH, cap=MAX_GROSS_STRESS, dist_end=str(dist_end.date()))


# --------------------------------------------------------------------------- run

def _families(S: dict) -> list:
    return sorted(set(S["family_of"].values()))


def build_all(c4, t54, S: dict, end: pd.Timestamp, scratch: Path, *, with_neighbours: bool,
              identity: bool = True) -> dict:
    """Everything up to the held books: the base, returns, betas, every variant's unit target and book."""
    from scripts.cef import exposure_readout as er
    lags = er.C4_DIMSON_LAGS
    if lags is None:
        raise InputError("exposure_readout.C4_DIMSON_LAGS is None: C4's lag rule has not run")
    t0 = time.time()
    D = build_base(c4, t54, S, end)
    D["t54"] = t54
    R = primary_returns(c4, D, scratch, identity=identity)
    RB, bh = beta_history(c4, D, R)
    Rf, vix, fmeta = load_factors(RB.index)
    ew_n = RB.notna().sum(axis=1)
    f_ew = RB.mean(axis=1).where(ew_n >= MIN_EW_NAMES)
    decide = D["TB"].index
    win = S["z_window"]
    betas = {"spy": dimson_betas(RB, Rf["SPY"], decide, window=win, lags=lags, min_obs=BETA_MIN_OBS),
             "cef": dimson_betas(RB, f_ew, decide, window=win, lags=lags, min_obs=BETA_MIN_OBS)}
    hold = hold_rows(D["TB"], D["ok"], S["min_names"])
    mk = exposure_makers(D, betas, _families(S))
    T, diag = {"base": D["TB"]}, {}
    for k in PROJECTED + PLACEBOS:
        T[k], diag[k] = variant_targets(D, S, hold, mk[k], k)
    nb = {}
    if with_neighbours:
        for w in NEIGHBOUR_WINDOWS:
            nb[f"spy_w{w}"] = dimson_betas(RB, Rf["SPY"], decide, window=w, lags=lags, min_obs=w // 2)
        nb["spy_l0"] = dimson_betas(RB, Rf["SPY"], decide, window=win, lags=0, min_obs=BETA_MIN_OBS)
        for k, B in nb.items():
            m2 = exposure_makers(D, {"spy": B, "cef": betas["cef"]}, _families(S))
            T[f"beta_spy[{k}]"], diag[f"beta_spy[{k}]"] = variant_targets(D, S, hold, m2["beta_spy"], k)
    H = {k: held_book(v, S) for k, v in T.items()}
    H["spy_hedge"] = spy_overlay(H["base"], betas["spy"])
    if with_neighbours:
        for c in HEDGE_CAPS:
            H[f"spy_cap[{c}]"] = spy_overlay(H["base"], betas["spy"], rule=("cap", c))
        H["spy_short_only"] = spy_overlay(H["base"], betas["spy"], rule="short_only")
    for k, B in nb.items():
        H[f"spy_hedge[{k}]"] = spy_overlay(H["base"], B, neighbour=True)
        diag[f"spy_hedge[{k}]"] = dict(held_name_days_no_beta=H[f"spy_hedge[{k}]"].attrs["held_name_days_no_beta"])
    return dict(D=D, R=R, RB=RB, Rf=Rf, vix=vix, fmeta=fmeta, f_ew=f_ew, betas=betas, nb=nb, hold=hold,
                T=T, H=H, diag=diag, lags=lags, beta_history=bh, seconds=time.time() - t0)


def score_all(c4, X: dict) -> dict:
    """Each book's daily gross P&L through `harness.evaluate` (shift(2)); the SPY leg's return is
    SPY's total return. `check_held` (H15) refuses a held name-day with no return."""
    R, Rf = X["R"], X["Rf"]
    out = {}
    for k, H in X["H"].items():
        if "SPY" in H.columns:
            base_cols = [c for c in H.columns if c != "SPY"]
            R0 = c4.check_held(H[base_cols], R.reindex(index=H.index, columns=base_cols), k)
            spy = Rf["SPY"].reindex(H.index)
            if (spy.isna() & hz.held(H)["SPY"].ne(0)).any():
                raise InputError(f"{k}: SPY has no return on a session the hedge is held")
            R0 = R0.assign(SPY=spy.fillna(0.0))[H.columns]
        else:
            R0 = c4.check_held(H, R.reindex(index=H.index, columns=H.columns), k)
        out[k] = hz.evaluate(H, R0)
    return out


def run(args) -> int:
    t54 = _t54()
    S = c4_params(t54)
    end = None
    with c4_inputs(Path(args.families_dir), Path(args.c4_cache)) as c4:
        end = c4.HOLDOUT_END
        X = build_all(c4, t54, S, end, SERIES_CACHE, with_neighbours=True)
        res = score_all(c4, X)
    D, R = X["D"], X["R"]
    lags = X["lags"]
    # ---- the base is the spec book: H11's series where given; and the base P&L through band_frontier
    from scripts.cef import band_frontier as bf
    from scripts.cef import c4_families as c4m
    Pb_bf = bf.evaluate(X["H"]["base"], c4m.check_held(X["H"]["base"], R.reindex_like(X["H"]["base"]), "base"))["pnl"]
    if not np.allclose(Pb_bf.to_numpy(), res["base"]["pnl"].to_numpy(), atol=1e-15, rtol=0):
        raise AssertionError("the base P&L differs between harness.evaluate and band_frontier.evaluate")
    checks = {}
    # THE ONE INDEPENDENT REPRODUCTION (quant-reviewer, 2026-11-24): the H11 spec-book series was built
    # by c4_paper_benchmark through spec_targets, a second implementation of the blend, on the fit
    # (2005-2022) only. Without it the base is checked against nothing independent, so a missing file
    # RAISES. The trial-54 holdout row below is the same function on the same returns: it confirms the
    # plumbing, not the construction, and 2023-2026 has no independent check.
    h11 = Path(args.c4_cache) / f"h11_spec_book_{S['spec_id']}.parquet"
    if not h11.is_file():
        raise InputError(f"{h11} is missing: it is the only independent check that the base is the spec's book "
                         "(c4_paper_benchmark.py run writes it into the C4 cache)")
    s = pd.read_parquet(h11)["tr"]
    d = float((res["base"]["pnl"].reindex(s.index) - s).abs().max())
    if not d < 1e-12:
        raise AssertionError(f"the base does not reproduce the H11 spec-book series ({h11.name}): max |diff| {d}")
    checks["h11_spec_book"] = dict(file=h11.name, sessions=int(len(s)), first=str(s.index[0].date()),
                                   last=str(s.index[-1].date()), max_abs_diff=d, independent=True)
    # the trial-54 holdout's committed blend row (the SAME construction, opened once on 2026-11-03): not
    # independent of this script's base, which calls trial54.level_targets itself
    hj = REPO / "results" / "cef" / "prereg_c4_level" / "holdout_2026-11-03.json"
    hb = json.loads(hj.read_text())["books"]["blend"]
    Ph = res["base"]["pnl"].loc[t54.HOLDOUT_START:t54.HOLDOUT_END]
    got = (float(Ph.mean() * 252 * 100), float(Ph.mean() / Ph.std() * np.sqrt(252)))
    if abs(got[0] - hb["ann_pct"]) > 1e-9 or abs(got[1] - hb["sr"]) > 1e-9:
        raise AssertionError(f"the base does not reproduce trial 54's holdout blend row: {got} vs "
                             f"({hb['ann_pct']}, {hb['sr']}) in {hj.name}")
    checks["trial54_holdout_blend"] = dict(file=hj.name, ann_pct=got[0], sr=got[1], independent=False,
                                           window=[str(Ph.index[0].date()), str(Ph.index[-1].date())])
    probes = [d for d in pd.to_datetime(["2006-06-30", "2012-06-29", "2020-03-31", "2024-06-28", "2026-09-29"])
              if d in X["RB"].index]
    checks["betas_vs_readout"] = check_against_readout(X["RB"], X["Rf"], X["betas"]["spy"], probes,
                                                       window=S["z_window"], lags=lags)
    # ---- every row on the same dates: from the base's first held session
    held0 = hz.held(X["H"]["base"]).abs().sum(axis=1)
    p0 = held0.index[held0 > 0][0]
    f_spy, f_ew = X["Rf"]["SPY"], X["f_ew"]
    lab = regime_labels(R.index, f_spy, X["vix"])
    table, eras, exposures, splits, caps, corrs, sdiff = {}, {}, {}, {}, {}, {}, {}
    core_P, core_meta = core_book(Path(args.core_dir))
    exante = {}
    for k, r in res.items():
        H = X["H"][k]
        Hb = H.drop(columns="SPY") if "SPY" in H.columns else H
        eb, miss = book_beta(Hb, X["betas"]["spy"])
        if "SPY" in H.columns:
            eb = eb + H["SPY"]
        exante[k] = dict(series=eb, held_name_days_no_beta=miss)
    # the base's own tilt, as a state: its ex-ante SPY beta decided at t, on the P&L row t+2
    tilt = exante["base"]["series"].shift(hz.EXEC_LAG)
    for k, r in res.items():
        P, TU = r["pnl"].loc[p0:], r["turn_series"].loc[p0:]
        G = hz.held(X["H"][k]).abs().sum(axis=1).loc[p0:]
        eb = exante[k]["series"].shift(hz.EXEC_LAG)       # the beta decided at t sits on the P&L row t+2
        row = lambda m: stats_row(P[m], TU[m], G[m], f_spy, f_ew, lags, eb)
        allm = np.ones(len(P), bool)
        early = P.index < hz.HOLDOUT_START
        L = lab.reindex(P.index)
        tb = tilt.reindex(P.index)
        table[k] = dict(full=row(allm), early=row(early), late=row(~early),
                        vix_calm=row((L["vix_stressed"] == False).to_numpy()),      # noqa: E712 (NaN is neither)
                        vix_stressed=row((L["vix_stressed"] == True).to_numpy()),   # noqa: E712
                        trend_up=row((L["trend_down"] == False).to_numpy()),        # noqa: E712
                        trend_down=row((L["trend_down"] == True).to_numpy()),       # noqa: E712
                        base_tilt_short=row((tb <= -TILT).to_numpy()),
                        base_tilt_mid=row(((tb > -TILT) & (tb < TILT)).to_numpy()),
                        base_tilt_long=row((tb >= TILT).to_numpy()),
                        spy_up_day=row((f_spy.reindex(P.index) >= 0).to_numpy()),
                        spy_down_day=row((f_spy.reindex(P.index) < 0).to_numpy()))
        if k in PLACEBOS:
            continue                 # a placebo seed is summarised across seeds (placebo_summary), not row by row
        eras[k] = {lab_: stats_row(P.loc[a:b], TU.loc[a:b], G.loc[a:b], f_spy, f_ew, lags, eb)
                   for lab_, a, b in hz.ERAS if len(P.loc[a:b]) > 20}
        exposures[k] = dict(full=factor_exposures(P, X["Rf"], lags),
                            early=factor_exposures(P[early], X["Rf"], lags),
                            late=factor_exposures(P[~early], X["Rf"], lags))
        j = pd.concat([P.rename("v"), core_P.rename("core"), res["base"]["pnl"].loc[p0:].rename("base")], axis=1).dropna()
        jl = j[j.index >= hz.HOLDOUT_START]
        corrs[k] = dict(core_full=float(j["v"].corr(j["core"])), core_late=float(jl["v"].corr(jl["core"])),
                        c4_base_full=float(j["v"].corr(j["base"])), c4_base_late=float(jl["v"].corr(jl["base"])),
                        overlap=[str(j.index[0].date()), str(j.index[-1].date())])
        Hc = X["H"][k].drop(columns="SPY") if "SPY" in X["H"][k].columns else X["H"][k]
        caps[k] = dict(late=capacity(Hc, D["adv"], hz.HOLDOUT_START, P.index[-1]),
                       full=capacity(Hc, D["adv"], p0, P.index[-1]))
        if "SPY" in X["H"][k].columns:
            caps[k]["spy_leg"] = dict(max_abs_weight=float(X["H"][k]["SPY"].abs().max()),
                                      note="SPY's ADV is not in the factor cache (no volume column): UNMEASURED here")
        if k != "base":
            sdiff[k] = sharpe_vs_base(P, res["base"]["pnl"].loc[p0:])
    placebo = placebo_summary(table)
    short_only = short_only_controls(res, exante, X["Rf"]["SPY"], p0)
    Pb = res["base"]["pnl"].loc[p0:]
    tilt_test = tilt_uncertainty(Pb, tilt.reindex(Pb.index))
    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    SERIES_CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: r["pnl"] for k, r in res.items()}).to_parquet(SERIES_CACHE / "pnl.parquet")
    pd.DataFrame({k: v["series"] for k, v in exante.items()}).to_parquet(SERIES_CACHE / "exante_beta.parquet")
    for k in ("base", "beta_spy", "family", "family_beta", "spy_hedge"):
        X["H"][k].to_parquet(SERIES_CACHE / f"H_{k}.parquet")
    X["betas"]["spy"].to_parquet(SERIES_CACHE / "beta_spy.parquet")
    doc = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        spec=dict((k, S[k]) for k in ("spec_id", "spec_sha256", "m", "cap", "band", "vol", "min_names", "min_adv", "z_window")),
        panels=dict(families_px_last=D["panel_px_last"], returns_last=str(R.index[-1].date()),
                    factor_cache=X["fmeta"], core=core_meta, beta_history=X["beta_history"]),
        conventions=dict(exec_lag=hz.EXEC_LAG, dimson_lags=lags, beta_window=S["z_window"], beta_min_obs=BETA_MIN_OBS,
                         first_held=str(p0.date()), split=str(hz.HOLDOUT_START.date()),
                         vix_median_window=VIX_MEDIAN_WINDOW, trend_window=TREND_WINDOW),
        checks=checks, diag=X["diag"], held_name_days_no_beta={k: v["held_name_days_no_beta"] for k, v in exante.items()},
        table=table, eras=eras, exposures=exposures, correlations=corrs, capacity=caps, sharpe_vs_base=sdiff,
        placebo_persistent=placebo, short_only_controls=short_only, tilt_short_state_test=tilt_test,
        seconds=X["seconds"])
    (OUT / "results.json").write_text(json.dumps(doc, indent=1, default=float))
    flat = []
    for k, rows in table.items():
        for r_, v in rows.items():
            flat.append(dict(variant=k, row=r_, **{kk: vv for kk, vv in v.items()}))
    pd.DataFrame(flat).to_csv(OUT / "table.csv", index=False, float_format="%.6g")
    chart(res, exante, p0, OUT / "equity.png", S)
    print_summary(doc)
    return 0


def chart(res: dict, exante: dict, p0, path: Path, S: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ("base", "beta_spy", "beta_cef", "family", "family_beta", "spy_hedge")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    for j, k in enumerate(PLACEBOS):
        P = res[k]["pnl"].loc[p0:]
        ((1 + P).cumprod() - 1).mul(100).plot(ax=a1, lw=0.7, color="0.6", ls="--",
                                             label="placebo, persistent (8 seeds)" if j == 0 else "_nolegend_")
    for k in show:
        P = res[k]["pnl"].loc[p0:]
        ((1 + P).cumprod() - 1).mul(100).plot(ax=a1, lw=1.6 if k == "base" else 1.0, label=k)
    a1.set_ylabel("cumulative gross return, % (compounded)")
    a1.set_title(f"C4 and its market-neutral variants, gross, shift(2), {S['spec_id']}", fontsize=10)
    a1.legend(fontsize=8, ncol=4)
    for k in ("base", "beta_spy", "family", "spy_hedge"):
        exante[k]["series"].loc[p0:].rolling(63).mean().plot(ax=a2, lw=1.0, label=k)
    a2.axhline(0, color="k", lw=0.5)
    a2.set_ylabel("ex-ante SPY beta\n(63-session mean)")
    a2.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_summary(doc: dict) -> None:
    print(f"\nC4 variants, gross, {doc['conventions']['first_held']} .. {doc['panels']['returns_last']}")
    print(f"{'variant':<22}{'row':<14}{'ann%':>7}{'vol%':>7}{'SR':>7}{'NWt':>7}{'maxDD%':>8}{'hit':>6}"
          f"{'turn':>7}{'gross':>7}{'bSPY':>7}{'bCEF':>7}{'exante':>8}")
    for k, rows in doc["table"].items():
        for r_ in ("full", "early", "late", "vix_calm", "vix_stressed", "trend_up", "trend_down", "base_tilt_short",
                   "base_tilt_mid", "base_tilt_long", "spy_up_day", "spy_down_day"):
            v = rows[r_]
            print(f"{k:<22}{r_:<14}{v['ann_pct']:7.2f}{v['vol_pct']:7.2f}{v['sr']:7.2f}{v['nw_t']:7.2f}{v['max_dd_pct']:8.2f}"
                  f"{v['hit']:6.2f}{v['turn']:7.1f}{v['gross']:7.2f}{v['beta_spy']:7.3f}{v['beta_cef']:7.3f}"
                  f"{v.get('exante_beta_mean', np.nan):8.3f}")
    print("\nchecks:", json.dumps(doc["checks"], indent=1, default=float)[:3000])


# --------------------------------------------------------------------------- fetch

def fetch(args) -> int:
    """Keyless yfinance through the exposure readout's own `fetch_yf`: the C4 factor ETFs and ^VIX.
    The readout's NYSE-calendar validation is applied from 2010 (its own FETCH_START); before that
    its calendar counts two closures (2004-06-11, 2007-01-02) as sessions, so the session check is
    made against the C4 panel's own sessions in `load_factors` instead."""
    from scripts.cef import exposure_readout as er
    asof = dt.date.fromisoformat(args.asof)
    fs = er.factor_set("c4")
    px, cg = er.fetch_yf(list(fs.etfs), FETCH_START, asof)
    er.validate_prices(px[px["date"] >= pd.Timestamp(er.FETCH_START)], asof)
    vx, _ = er.fetch_yf([VIX], FETCH_START, asof)
    FACTOR_CACHE.mkdir(parents=True, exist_ok=True)
    px.to_parquet(FACTOR_CACHE / "etf_px.parquet", index=False)
    vx.to_parquet(FACTOR_CACHE / "vix_px.parquet", index=False)
    import yfinance
    meta = dict(asof=asof.isoformat(), start=FETCH_START.isoformat(), tickers=list(fs.etfs) + [VIX],
                source="yfinance Ticker.history(auto_adjust=False, actions=True), keyless, via exposure_readout.fetch_yf",
                yfinance_version=yfinance.__version__, capital_gains_column=cg,
                fetched_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                etf_px_sha256=_sha(FACTOR_CACHE / "etf_px.parquet"))
    (FACTOR_CACHE / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


# --------------------------------------------------------------------------- truncation check

def trunc(args) -> int:
    """`harness.truncation_check` on the new target builders: everything is rebuilt from the families
    panel and the returns cut at each probe date BEFORE anything is computed (the parquet files are
    rewritten cut into a temporary folder; the CRSP rows are filtered), and each probe's row of the
    variants' unit targets, held books and the SPY hedge must equal the full build's exactly."""
    t54 = _t54()
    S = c4_params(t54)
    probes = [pd.Timestamp(x) for x in args.probes]
    fam_src = Path(args.families_dir).expanduser().resolve()
    tmp = Path(tempfile.mkdtemp(prefix="c4bn_trunc_"))

    def build(end):
        cut = tmp / ("full" if end is None else end.date().isoformat())
        cut.mkdir(parents=True, exist_ok=True)
        hi = pd.Timestamp(args.end) if end is None else end
        for f in ("cef_prices", "cef_nav", "cef_actions"):
            d = pd.read_parquet(fam_src / f"{f}.parquet", filters=[("date", "<=", hi)])
            d.to_parquet(cut / f"{f}.parquet")
        with c4_inputs(cut, Path(args.c4_cache)) as c4:
            X = build_all(c4, t54, S, hi, cut / "_scratch", with_neighbours=False, identity=False)
        frames = {f"T:{k}": v for k, v in X["T"].items()}
        frames.update({f"H:{k}": v for k, v in X["H"].items()})
        return pd.concat(frames, axis=1)

    try:
        r = hz.truncation_check(build, probes, atol=args.atol)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(r, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truncation_check.json").write_text(json.dumps(dict(
        r, atol=args.atol, compared=["T:" + k for k in ("base",) + PROJECTED + PLACEBOS]
        + ["H:" + k for k in ("base",) + PROJECTED + PLACEBOS + ("spy_hedge",)],
        atol_reason=("a cut panel lacks the funds that list later, so the vol scalar's row sums run over fewer "
                     "zero terms in another order; at atol 0 the first difference was 2.2e-17 (beta_spy, BGR, "
                     "2009-06-30). A leak moves a weight by orders of magnitude more."),
        run_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), indent=1))
    return 0


# --------------------------------------------------------------------------- the live window

def live(args) -> int:
    """C4's armed sessions, re-scored. Each variant is rebuilt on a READ-ONLY COPY of the VM's C4 panel
    (prices to the last session, NAV a session behind), banded from FLAT at the first armed decision
    (the account was flat until then), and scored on price returns (Alpaca paper pays no dividends).
    Beside it: the official P&L of record (Alpaca's end-of-day equity) and the live plans' own books,
    so the reader can see whether the simulated base tracks what the account did before reading the
    variants. Fund betas here are on VM price returns (no distribution file on the VM panel)."""
    from scripts.cef import band_frontier as bf
    from scripts.cef import exposure_readout as er
    t54 = _t54()
    S = c4_params(t54)
    vm = Path(args.vm_panel).expanduser().resolve()
    hist = Path(args.hist).expanduser().resolve()
    for f in (vm / "cef_prices.parquet", vm / "cef_nav.parquet", hist / "eod_equity.csv"):
        if not f.is_file():
            raise InputError(f"{f} is missing")
    plans = {}
    for d in sorted(p for p in hist.iterdir() if p.is_dir()):
        pj = d / "plan.json"
        if pj.is_file():
            j = json.loads(pj.read_text())
            if j.get("book") != "c4" or j.get("spec_id") != S["spec_id"]:
                raise InputError(f"{pj}: book {j.get('book')} spec {j.get('spec_id')}, not C4's {S['spec_id']}")
            plans[pd.Timestamp(j["asof"])] = j
    if not plans:
        raise InputError(f"no plan.json under {hist}")
    first_decision = min(plans)
    lags = er.C4_DIMSON_LAGS
    with c4_inputs(vm, None) as c4:
        U = c4.universes()
        UF = U["frozen"]
        fam = pd.read_csv(c4.FROZEN).set_index("ticker")["family"].reindex(UF)
        px = pd.read_parquet(c4.FAM_SRC[0])
        px["date"] = pd.to_datetime(px["date"])
        close = px[px.ticker.isin(UF)].pivot_table(index="date", columns="ticker", values="close").sort_index()
        last = close.index.max()
        B = t54.level_targets(UF, fam, S, level_mask=None, end=last)
        TB = B["TB"]
        p = bf.build_panel(universe=UF, source=c4.FAM_SRC, mask=c4.declared_mask(UF, "M1M2"))
        idx = TB.index
        z, adv = p["z"].reindex(idx)[UF], p["adv"].reindex(idx)[UF]
        ok = (z.notna() & (adv.fillna(-1) >= S["min_adv"])).to_numpy()
        retF = p["ret"][UF]
        D = dict(UF=UF, fam=fam, TB=TB, ok=ok, RF=retF.to_numpy(), posF=retF.index.get_indexer(idx), t54=t54)
    # the paired panel (prices AND NAV) ends a session before the prices; nothing after its last
    # decision date is needed (the official P&L of record ends there too)
    sess = close.index[(close.index >= BETA_START) & (close.index <= idx[-1])]
    Rpx = close.reindex(sess).pct_change(fill_method=None)[UF]
    Rf, vix, fmeta = load_factors(sess)
    f_ew = Rpx.mean(axis=1).where(Rpx.notna().sum(axis=1) >= MIN_EW_NAMES)
    dec = idx[idx >= first_decision - pd.Timedelta(days=400)]
    betas = {"spy": dimson_betas(Rpx, Rf["SPY"], dec, window=S["z_window"], lags=lags, min_obs=BETA_MIN_OBS),
             "cef": dimson_betas(Rpx, f_ew, dec, window=S["z_window"], lags=lags, min_obs=BETA_MIN_OBS)}
    betas = {k: v.reindex(idx) for k, v in betas.items()}
    hold = hold_rows(TB, ok, S["min_names"])
    mk = exposure_makers(D, betas, _families(S))
    T = {"base": TB}
    for k in PROJECTED:
        T[k], _ = variant_targets(D, S, hold, mk[k], k)
    # rows before `dec` have no betas, so the beta variants carry nothing there; they are never held:
    # every book below is banded from FLAT at the first armed decision
    L = {k: held_book(v.loc[first_decision:], S) for k, v in T.items()}
    Bl = betas["spy"].loc[first_decision:]
    L["spy_hedge"] = spy_overlay(L["base"], Bl)
    for c in HEDGE_CAPS:
        L[f"spy_cap[{c}]"] = spy_overlay(L["base"], Bl, rule=("cap", c))
    L["spy_short_only"] = spy_overlay(L["base"], Bl, rule="short_only")
    b63 = dimson_betas(Rpx, Rf["SPY"], dec, window=63, lags=lags, min_obs=31).reindex(idx).loc[first_decision:]
    L["spy_hedge[spy_w63]"] = spy_overlay(L["base"], b63)
    # official P&L of record
    eod = pd.read_csv(hist / "eod_equity.csv", parse_dates=["date"]).set_index("date")["equity"]
    off = eod.pct_change().loc[first_decision + pd.Timedelta(days=1):]
    out = dict(first_decision=str(first_decision.date()), vm_px_last=str(last.date()),
               vm_nav_last=str(pd.read_parquet(vm / "cef_nav.parquet")["date"].max()),
               official_last=str(eod.index.max().date()), factor_cache=fmeta)
    R = Rpx.reindex(L["base"].index)
    pnl = {}
    for k, H in L.items():
        cols = [c for c in H.columns if c != "SPY"]
        R0 = R[cols].copy()
        heldm = hz.held(H[cols]).ne(0)
        if (R0.isna() & heldm).any().any():
            raise InputError(f"live {k}: a held name has no VM price return")
        R0 = R0.fillna(0.0)
        if "SPY" in H.columns:
            R0["SPY"] = Rf["SPY"].reindex(H.index).fillna(0.0)
        pnl[k] = hz.evaluate(H, R0[H.columns])["pnl"]
    P = pd.DataFrame(pnl)
    win = P.index[(P.index > first_decision + pd.Timedelta(days=1)) & P.index.isin(off.index)]
    P, offw = P.loc[win], off.reindex(win)
    fspy = Rf["SPY"].reindex(win)
    few = f_ew.reindex(win)

    def ols_beta(y, x):
        x = x - x.mean()
        return float((x * (y - y.mean())).sum() / (x * x).sum())

    rows = {}
    for k in P.columns:
        y = P[k]
        rows[k] = dict(sum_pct=float(y.sum() * 100), vol_ann_pct=float(y.std() * np.sqrt(252) * 100),
                       beta_spy_same_day=ols_beta(y, fspy), beta_cef_same_day=ols_beta(y, few),
                       corr_official=float(y.corr(offw)), worst_day_pct=float(y.min() * 100),
                       best_day_pct=float(y.max() * 100))
    rows["official"] = dict(sum_pct=float(offw.sum() * 100), vol_ann_pct=float(offw.std() * np.sqrt(252) * 100),
                            beta_spy_same_day=ols_beta(offw, fspy), beta_cef_same_day=ols_beta(offw, few),
                            worst_day_pct=float(offw.min() * 100), best_day_pct=float(offw.max() * 100))
    # the live plans' own books: ex-ante SPY beta of what the account held after each session's fills
    eb_live = {}
    for asof, j in sorted(plans.items()):
        tg = {s: v["target"] for s, v in j["targets"].items() if v.get("target")}
        cl = j["closes"]
        val = pd.Series({s: q * cl[s] for s, q in tg.items()})
        if asof not in betas["spy"].index:
            continue
        b = betas["spy"].loc[asof].reindex(val.index)
        if b.isna().any():
            raise InputError(f"plan {asof.date()}: no SPY beta for {list(b.index[b.isna()])}")
        eq = float(j["equity"])
        if sorted(j["targets"]) != sorted(UF):
            raise InputError(f"plan {asof.date()} does not carry a target for every one of the 74")
        wl = pd.Series({s: (j["targets"][s]["target"] * cl[s] / eq) if j["targets"][s]["target"] else 0.0
                        for s in UF})                    # target 0 shares is flat, by the plan's own record
        ws = L["base"].loc[asof] if asof in L["base"].index else None
        # second method: the readout's own 4-factor fit (EQUITY loading of SPY IEF HYG PFF, Gram-Schmidt,
        # Dimson 0..L) on the same window, on VM price returns; a name without a complete window is
        # named, never filled
        W = pd.concat([Rpx, Rf], axis=1)
        try:
            f4 = er.fit_window(W, W.index, asof, list(val.index), window=S["z_window"], lags=lags,
                               fs=er.factor_set("c4"))
            eq4 = float((val * f4.beta_sum["EQUITY"]).sum() / eq)
        except er.ReadoutError as e:
            eq4 = f"UNMEASURED: {e}"
        eb_live[str(asof.date())] = dict(beta_spy=float((val * b).sum() / eq), equity_4factor=eq4,
                                         beta_cef=float((val * betas["cef"].loc[asof].reindex(val.index)).sum() / eq),
                                         gross=float(val.abs().sum() / eq), net=float(val.sum() / eq),
                                         sim_base_gross=None if ws is None else float(ws.abs().sum()),
                                         sim_base_net=None if ws is None else float(ws.sum()),
                                         corr_sim_base_weights=None if ws is None else float(np.corrcoef(wl, ws[UF])[0, 1]),
                                         abs_diff_sim_base_weights=None if ws is None else float((wl - ws[UF]).abs().sum()))
    eb_sim = {}
    for k, H in L.items():
        Hb = H.drop(columns="SPY") if "SPY" in H.columns else H
        e, miss = book_beta(Hb, betas["spy"].reindex(Hb.index))
        if "SPY" in H.columns:
            e = e + H["SPY"]
        eb_sim[k] = dict(mean=float(e.mean()), last=float(e.iloc[-1]), held_name_days_no_beta=miss)
    # THE HEADLINE OF THE LIVE WINDOW (quant-reviewer, 2026-11-24; `scratchpad/rev/live_off.py`): the
    # overlay applied to the OFFICIAL return, not to a simulated book. Plan t's own book (its target
    # shares at its closes, over its equity) gives the ex-ante SPY beta; the SPY weight decided at t is
    # held over session t+2 on the exchange calendar, as the harness books every decision. The simulated
    # base bands from flat and runs at a lower gross than the account, so it is the secondary view.
    cal = close.index
    ofh = {}
    for rule in ("full", "short_only"):
        h = {}
        for t, v in eb_live.items():
            t = pd.Timestamp(t)
            i = cal.get_loc(t)
            if i + hz.EXEC_LAG < len(cal):
                h[cal[i + hz.EXEC_LAG]] = float(hedge_weight(pd.Series([v["beta_spy"]]), rule).iloc[0])
        h = pd.Series(h).reindex(win)
        if h.isna().any():
            raise InputError(f"no plan sizes the hedge for {[str(d.date()) for d in h.index[h.isna()]]}")
        ofh[rule] = h

    def ols_se(y, x):
        b = ols_beta(y, x)
        xc, yc = x - x.mean(), y - y.mean()
        e = yc - b * xc
        return float(np.sqrt((e ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))

    big = fspy.abs().sort_values().index[-3:]           # the three largest SPY moves in the window
    vol = lambda x: float(x.std() * np.sqrt(252) * 100)

    def orow(y):
        return dict(sum_pct=float(y.sum() * 100), vol_ann_pct=vol(y), beta_spy_same_day=ols_beta(y, fspy),
                    beta_spy_se=ols_se(y, fspy), worst_day_pct=float(y.min() * 100), best_day_pct=float(y.max() * 100),
                    vol_without_3_largest_spy_days_pct=vol(y.drop(big)))
    official_overlay = dict(
        hedge_rule="SPY weight decided at plan t = minus the plan book's ex-ante SPY beta (one-factor, Dimson 0-2, "
                   "252 sessions before t, VM price returns); held over t+2",
        official=orow(offw), official_plus_full_hedge=orow(offw + ofh["full"] * fspy),
        official_plus_short_only_hedge=orow(offw + ofh["short_only"] * fspy),
        hedge_leg_sum_pct=float((ofh["full"] * fspy).sum() * 100),
        hedge_weight_mean=float(ofh["full"].mean()), hedge_weight_range=[float(ofh["full"].min()), float(ofh["full"].max())],
        spy_vol_ann_pct=vol(fspy), spy_sum_pct=float(fspy.sum() * 100), cef_ew_vol_ann_pct=vol(few),
        three_largest_spy_days=[str(d.date()) for d in sorted(big)],
        note="same-day OLS betas on 13 points (s.e. given) are not the Dimson 0-2 ex-ante betas the hedge is sized on")
    gap = {d: (v["gross"], v["sim_base_gross"]) for d, v in eb_live.items() if v["sim_base_gross"] is not None}
    out.update(window=[str(win[0].date()), str(win[-1].date())], sessions=int(len(win)), rows=rows,
               official_overlay=official_overlay,
               gross_live_vs_simulated=dict(first=gap[min(gap)], ratio_min=float(min(s / g for g, s in gap.values())),
                                            ratio_max=float(max(s / g for g, s in gap.values()))),
               exante_live_plans=eb_live, exante_simulated=eb_sim,
               daily=dict((str(d.date()), dict(official=float(offw.loc[d]), spy=float(fspy.loc[d]), cef_ew=float(few.loc[d]),
                                                **{k: float(P.loc[d, k]) for k in P.columns})) for d in win))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live_window.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "daily"}, indent=1, default=float))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--asof", required=True)
    r = sub.add_parser("run")
    r.add_argument("--families-dir", required=True)
    r.add_argument("--core-dir", required=True)
    r.add_argument("--c4-cache", required=True)
    t = sub.add_parser("trunc")
    t.add_argument("--families-dir", required=True)
    t.add_argument("--c4-cache", required=True)
    t.add_argument("--end", default="2026-09-30")
    t.add_argument("--atol", type=float, default=0.0,
                   help="0 (exact) unless a stated reason: see the note on column order in build_base")
    t.add_argument("--probes", nargs="+", default=["2009-06-30", "2016-03-31", "2020-03-31", "2024-06-28", "2026-06-30"])
    lv = sub.add_parser("live")
    lv.add_argument("--vm-panel", required=True)
    lv.add_argument("--hist", required=True)
    a = ap.parse_args(argv)
    return dict(fetch=fetch, run=run, trunc=trunc, live=live)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
