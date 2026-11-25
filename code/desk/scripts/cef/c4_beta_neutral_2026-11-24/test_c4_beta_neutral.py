"""The C4 beta-neutral research script's own pieces, on synthetic frames (a check on the code, not a
claim about the market). Nothing here reads a panel or opens a socket: the module imports only
numpy, pandas and the harness at module level, and every heavy import is inside a function.

Each test was checked to FAIL against the defect it names before it was kept:
- the lookahead tests, against a window that includes row t (with the runtime assert removed too),
  and against data taken one row later than the asserted labels (`F[rp + 1]`);
- the variant_targets test, against no renormalisation, a vol scalar taken on the base row, a held
  row set flat, and an emptied row left uncounted;
- the factor-gap test, against a body that skips a window with a factor NaN instead of raising;
- the regime test, against labels left on the decision row (no EXEC_LAG shift);
- the hedge-rule tests, against `cap` hedging the whole beta;
- the overlay test, against a book beta that treats a missing beta as zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # the module under test; CreditTrading comes from PYTHONPATH (desk/README.md)

import c4_beta_neutral as bn  # noqa: E402


def _frames(n=400, k=3, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    f = pd.Series(rng.normal(0, 0.01, n), index=idx)
    true = np.array([0.5, 1.0, 1.5])[:k]
    Y = pd.DataFrame(np.outer(f.to_numpy(), true) + rng.normal(0, 0.004, (n, k)), index=idx,
                     columns=[f"N{i}" for i in range(k)])
    return Y, f


def _direct(Y, f, t, window, lags):
    """Second method: one OLS per name on the rows strictly before t, written plainly."""
    i = Y.index.get_loc(t)
    rows = range(i - window, i)
    X = np.array([[1.0] + [f.iloc[s - l] for l in range(lags + 1)] for s in rows])
    out = {}
    for c in Y.columns:
        y = Y[c].iloc[i - window:i].to_numpy()
        m = np.isfinite(y)
        out[c] = np.linalg.lstsq(X[m], y[m], rcond=None)[0][1:].sum()
    return pd.Series(out)


def test_dimson_betas_match_a_direct_ols_on_the_window_before_t():
    Y, f = _frames()
    Y.iloc[150:160, 1] = np.nan                      # a partial window for one name
    dec = Y.index[[300, 350, 399]]
    B = bn.dimson_betas(Y, f, dec, window=252, lags=2, min_obs=126)
    for t in dec:
        assert np.allclose(B.loc[t].to_numpy(), _direct(Y, f, t, 252, 2).to_numpy(), atol=1e-12)


def test_dimson_betas_do_not_read_row_t_or_later():
    Y, f = _frames()
    t = Y.index[300]
    B0 = bn.dimson_betas(Y, f, pd.DatetimeIndex([t]), window=252, lags=2, min_obs=126)
    Y2, f2 = Y.copy(), f.copy()
    Y2.iloc[300:] = 99.0                             # change the decision row and everything after it
    f2.iloc[300:] = -5.0
    B1 = bn.dimson_betas(Y2, f2, pd.DatetimeIndex([t]), window=252, lags=2, min_obs=126)
    assert np.array_equal(B0.to_numpy(), B1.to_numpy())


def test_dimson_betas_leave_a_short_history_unmeasured_and_refuse_a_factor_gap():
    Y, f = _frames()
    Y.iloc[:200, 0] = np.nan                         # name 0 has only 100 rows in the window at row 300
    B = bn.dimson_betas(Y, f, Y.index[[300]], window=252, lags=2, min_obs=126)
    assert np.isnan(B.iloc[0, 0]) and np.isfinite(B.iloc[0, 1:]).all()
    early = bn.dimson_betas(Y, f, Y.index[[100]], window=252, lags=2, min_obs=126)
    assert early.isna().all().all()                  # not enough history: unmeasured, not an error
    f.iloc[200] = np.nan                             # a gap INSIDE the factor's history
    with pytest.raises(bn.InputError, match="factor return is missing"):
        bn.dimson_betas(Y, f, Y.index[[300]], window=252, lags=2, min_obs=126)


def test_hedge_rules():
    b = pd.Series([-0.5, -0.15, 0.0, 0.25, 0.6])
    assert np.allclose(bn.hedge_weight(b, "full"), -b)
    assert np.allclose(bn.hedge_weight(b, ("cap", 0.2)), [0.3, 0.0, 0.0, -0.05, -0.4])
    assert np.allclose(bn.hedge_weight(b, "short_only"), [0.5, 0.15, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        bn.hedge_weight(b, ("cap", 0.0))


def test_spy_overlay_refuses_a_held_name_without_a_beta_unless_a_neighbour_says_so():
    idx = pd.bdate_range("2020-01-01", periods=3)
    H = pd.DataFrame({"A": [0.2, 0.2, 0.0], "B": [-0.2, -0.1, 0.0]}, index=idx)
    B = pd.DataFrame({"A": [1.2, 1.1, np.nan], "B": [0.8, np.nan, 0.9]}, index=idx)
    with pytest.raises(bn.InputError, match="no measured SPY beta"):
        bn.spy_overlay(H, B)
    out = bn.spy_overlay(H, B, neighbour=True)
    assert out.attrs["held_name_days_no_beta"] == 1
    B2 = B.fillna(1.0)
    ok = bn.spy_overlay(H, B2)
    assert np.allclose(ok["SPY"], -(H["A"] * B2["A"] + H["B"] * B2["B"]))
    assert ok.loc[idx[2], "SPY"] == 0.0              # a flat book is not hedged


def test_regime_labels_sit_on_the_pnl_row_the_decision_earns():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(2)
    spy = pd.Series(rng.normal(0, 0.01, 400), index=idx)
    vix = pd.Series(15 + np.cumsum(rng.normal(0, 0.5, 400)).clip(-10, 30), index=idx)
    lab = bn.regime_labels(idx, spy, vix)
    med = vix.rolling(bn.VIX_MEDIAN_WINDOW, min_periods=bn.VIX_MEDIAN_WINDOW).median()
    for s in (300, 350, 399):
        assert lab["vix_stressed"].iloc[s] == (vix.iloc[s - 2] > med.iloc[s - 2])
    assert lab.iloc[:bn.VIX_MEDIAN_WINDOW + 1]["vix_stressed"].isna().all()


def test_variant_targets_neutralise_renormalise_rescale_and_hold():
    """variant_targets on a synthetic base (added on the reviewer's request, 2026-11-24): a trading row
    is the projection of the base's unit-gross row, renormalised to unit gross, times the builder's vol
    scalar computed on its OWN weights with trial54's estimator; a held row copies the previous row; a
    row the projection empties holds and is counted; a fund off `on` carries nothing."""
    from scripts.cef import band_frontier as bf
    t54 = bn._t54()
    rng = np.random.default_rng(5)
    names = ["A", "B", "C", "D"]
    RF = rng.normal(0, 0.01, (100, 4))
    idx = pd.bdate_range("2021-01-04", periods=5)
    base_row = np.array([0.4, -0.3, 0.5, -0.6])
    TB = pd.DataFrame([base_row * 1.1, base_row * 1.1, base_row * 0.9, base_row, base_row * 1.2],
                      index=idx, columns=names)
    ok = np.ones((5, 4), bool)
    D = dict(TB=TB, ok=ok, RF=RF, posF=np.arange(70, 75), t54=t54, UF=names)
    S = dict(vol=0.06)
    beta = np.array([0.5, 1.0, 1.5, 0.8])
    hold = np.array([False, True, False, False, False])

    def exposures(i):
        on = np.ones(4, bool)
        if i == 3:
            on[:] = [True, False, False, False]          # one allowed fund and a ones column: nothing survives
        if i == 4:
            on[3] = False                                # fund D unmeasured on the last row
        return np.column_stack([np.ones(4), beta]), on

    T, diag = bn.variant_targets(D, S, hold, exposures, "test")
    for i in (0, 2, 4):
        w = TB.iloc[i].to_numpy() / np.abs(TB.iloc[i].to_numpy()).sum()
        X, on = exposures(i)
        w2 = bn.neutralise_row(w, X, on)
        w2 = w2 / np.abs(w2).sum()
        rv = t54._rv(None, RF, 70 + i, w2, bf.VOL_LOOKBACK)
        want = w2 * float(np.clip(S["vol"] / rv, 0.2, 2.5))
        assert np.allclose(T.iloc[i].to_numpy(), want, atol=1e-15)
        got = T.iloc[i].to_numpy()
        assert abs(got.sum()) < 1e-12 and abs(got[on] @ beta[on]) < 1e-12
    assert T.iloc[1].equals(T.iloc[0])                   # held row
    assert T.iloc[3].equals(T.iloc[2])                   # emptied row holds ...
    assert diag["no_gross_rows"] == 1 and diag["trading_rows"] == 3   # ... and is counted
    assert T.iloc[4]["D"] == 0.0                         # off `on`
    # unit gross on eligible funds left out of `on`: rows 0 and 2 none, row 3 B+C+D, row 4 D
    u = np.abs(base_row) / np.abs(base_row).sum()
    assert diag["unit_gross_dropped_for_no_beta_max"] == pytest.approx(u[1:].sum(), abs=1e-15)
    assert diag["unit_gross_dropped_for_no_beta_mean"] == pytest.approx((u[1:].sum() + u[3]) / 4, abs=1e-15)
    # Renormalisation is only visible where the clip binds (above it the vol scalar undoes any scale):
    # on near-flat returns the scalar sits at its 2.5 ceiling, so a trading row's gross must be 2.5
    D2 = dict(D, RF=RF * 1e-4)
    T2, _ = bn.variant_targets(D2, S, hold, exposures, "test-clip")
    for i in (0, 2, 4):
        assert np.abs(T2.iloc[i].to_numpy()).sum() == pytest.approx(2.5, abs=1e-12)


def test_hold_rows_follow_the_base():
    idx = pd.bdate_range("2020-01-01", periods=5)
    TB = pd.DataFrame([[0, 0], [0.1, -0.1], [0.1, -0.1], [0.2, -0.2], [0.2, -0.2]], index=idx,
                      columns=["A", "B"], dtype=float)
    ok = np.array([[False, False], [True, True], [False, True], [True, True], [True, True]])
    h = bn.hold_rows(TB, ok, min_names=2)
    assert h.tolist() == [True, False, True, False, True]
    TB.iloc[2] = [0.3, -0.3]                         # a thin row that is not a copy: level_targets changed
    with pytest.raises(AssertionError):
        bn.hold_rows(TB, ok, min_names=2)


# --- neutralise_row (the study's own helper; moved from the harness draft on 2026-11-25) -------------------------
# Checked to FAIL against a body that returns `w` unchanged (every exposure test), against one
# that skips the `on` mask (the off-mask and family tests), and against one that fills a NaN
# exposure with zero (the refusal test).

def test_neutralise_row_removes_every_exposure_on_the_allowed_names():
    rng = np.random.default_rng(3)
    n = 12
    w = rng.normal(0, 0.1, n)
    beta = rng.normal(1.0, 0.3, n)
    on = np.ones(n, bool)
    on[[2, 7]] = False
    X = np.column_stack([np.ones(n), beta])
    out = bn.neutralise_row(w, X, on)
    assert np.allclose(out[~on], 0.0)
    assert abs(out[on].sum()) < 1e-12 and abs(out[on] @ beta[on]) < 1e-12


def test_neutralise_row_is_the_least_squares_closest_neutral_book():
    rng = np.random.default_rng(4)
    n = 9
    w = rng.normal(0, 0.1, n)
    X = np.column_stack([np.ones(n), rng.normal(1, 0.4, n)])
    on = np.ones(n, bool)
    out = bn.neutralise_row(w, X, on)
    # second method: minimise |v - w|^2 subject to X'v = 0 through its KKT system
    k = X.shape[1]
    K = np.block([[np.eye(n), X], [X.T, np.zeros((k, k))]])
    v = np.linalg.solve(K, np.concatenate([w, np.zeros(k)]))[:n]
    assert np.allclose(out, v, atol=1e-12)
    # a book already neutral comes back unchanged
    assert np.allclose(bn.neutralise_row(out, X, on), out, atol=1e-12)


def test_neutralise_row_with_family_dummies_is_the_within_family_demeaning():
    w = np.array([0.3, -0.1, 0.2, 0.05, -0.4, 0.25, 0.1])
    fam = np.array(["a", "a", "a", "b", "b", "c", "a"])
    on = np.array([True, True, True, True, True, True, False])
    X = np.column_stack([(fam == g).astype(float) for g in ("a", "b", "c")])
    out = bn.neutralise_row(w, X, on)
    want = np.zeros_like(w)
    for g in ("a", "b", "c"):
        s = on & (fam == g)
        want[s] = w[s] - w[s].mean()
    assert np.allclose(out, want, atol=1e-12)
    assert out[5] == pytest.approx(0.0, abs=1e-15)     # a family with one allowed name carries nothing
    assert out[6] == 0.0                                # off the mask


def test_neutralise_row_refuses_a_missing_exposure_on_an_allowed_name():
    w = np.array([0.1, -0.1, 0.2])
    X = np.column_stack([np.ones(3), [1.0, np.nan, 0.8]])
    with pytest.raises(ValueError, match="not finite"):
        bn.neutralise_row(w, X, np.ones(3, bool))
    # the same name taken out of `on` is allowed: the caller has said what it does with it
    out = bn.neutralise_row(w, X, np.array([True, False, True]))
    assert out[1] == 0.0 and abs(out[0] + out[2]) < 1e-12
    with pytest.raises(ValueError, match="NaN or infinite weight"):
        bn.neutralise_row(np.array([0.1, np.nan, 0.2]), np.ones((3, 1)), np.ones(3, bool))


def test_neutralise_row_handles_a_rank_deficient_exposure_set():
    w = np.array([0.3, -0.2, 0.1, -0.4])
    ones = np.ones(4)
    X = np.column_stack([ones, ones])                   # the same column twice
    out = bn.neutralise_row(w, X, np.ones(4, bool))
    assert np.allclose(out, w - w.mean(), atol=1e-12)
