"""Tests for close_auction_forecast.py: the cross rule, the quote reductions, the window boundaries,
the alignment of every forecast feature (no lookahead), the cap arithmetic and the public-output
check. Synthetic inputs only; nothing opens a socket (the root conftest blocks the network).

Run from the private repository's root, with a CreditTrading checkout on PYTHONPATH (desk/README.md):
    PYTHONPATH=<CreditTrading checkout> python3 -m pytest -p no:cacheprovider \
        desk/scripts/cef/close_auction_forecast_2026-11-24/test_close_auction_forecast.py
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # the module under test; CreditTrading comes from PYTHONPATH
import close_auction_forecast as caf  # noqa: E402

NS = 10**9


def P(x, c, s, p=10.0, t="2026-11-20T21:00:03Z"):
    return dict(x=x, c=c, s=s, p=p, t=t)


# ------------------------------------------------------------------------------- the cross rule

def test_round_cross_is_the_primary_6_size():
    r = caf.classify_close([P("N", "6", 5149, 22.14), P("N", "M", 5149, 22.14), P("P", "M", 100, 22.22)], ("N",))
    assert r["kind"] == "round" and r["shares"] == 5149 and r["usd"] == pytest.approx(5149 * 22.14)


def test_no_6_with_an_M_is_no_round_lot_cross_and_never_reads_the_M_size():
    # SCD 2026-11-20: two primary M prices, no 6 -> no round-lot cross; price = the earliest M
    r = caf.classify_close([P("N", "M", 1400, 14.348, "2026-11-20T21:00:02Z"),
                            P("N", "M", 300, 14.28, "2026-11-20T21:10:00Z")], ("N",))
    assert r["kind"] == "no_round" and r["shares"] == 0 and r["usd"] == 0
    assert r["price"] == 14.348


def test_two_primary_6_prints_on_one_code_are_never_sized():
    r = caf.classify_close([P("N", "6", 100), P("N", "6", 200)], ("N",))
    assert r["kind"] == "multi6" and math.isnan(r["usd"])


def test_another_listing_venues_6_is_not_the_cross_but_is_flagged():
    r = caf.classify_close([P("A", "6", 900), P("P", "M", 50)], ("N",))
    assert r["kind"] == "no_primary" and r["other_listing_6"] is True


def test_t_is_not_a_primary_for_an_nyse_listing():
    # T's 6 prints on NYSE names (446 in the data) are not the cross, and not a listing move either
    r = caf.classify_close([P("T", "6", 900), P("N", "M", 120)], ("N",))
    assert r["kind"] == "no_round" and r["other_listing_6"] is False


def test_nasdaq_listing_reads_its_cross_on_t():
    # the old rule read Q alone and scored this day as no round-lot cross (review, 2026-11-24)
    r = caf.classify_close([P("T", "6", 1117, 20.0), P("P", "M", 100, 20.01)], ("Q", "T"))
    assert r["kind"] == "round" and r["shares"] == 1117 and r["six_venues"] == "T"


def test_the_same_cross_on_t_and_q_is_counted_once_and_a_disagreement_is_never_sized():
    same = caf.classify_close([P("T", "6", 759, 9.5), P("Q", "6", 759, 9.5), P("Q", "M", 759, 9.5)], ("Q", "T"))
    assert same["kind"] == "round" and same["shares"] == 759 and same["six_venues"] == "QT"
    differ = caf.classify_close([P("T", "6", 759, 9.5), P("Q", "6", 700, 9.5)], ("Q", "T"))
    assert differ["kind"] == "multi6"


def test_primary_must_be_a_tuple():
    with pytest.raises(TypeError):
        caf.classify_close([P("N", "6", 1)], "N")


def test_primary_codes_give_nasdaq_its_alias():
    U = pd.DataFrame({"ticker": ["X", "Y"], "exchange": ["NYSE", "NASDAQ"], "book": "c4"})
    ex = {"N": "New York Stock Exchange", "Q": "NASDAQ OMX", "T": "NASDAQ Int", "A": "NYSE American (AMEX)"}
    assert caf.primary_codes(U, ex) == {"X": ("N",), "Y": ("Q", "T")}


def _panel_case():
    """Two NYSE names over four sessions; the venue's M era starts on day 2; day 3 has no M for any
    name (Alpaca's M data missing); on day 4 only name B has an M."""
    days = ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]
    rows = []
    for d in days:
        rows.append(dict(date=d, ticker="A", x="N", c="6", p=10.0, s=500.0, t=f"{d}T21:00:03Z"))
    for d in (days[1], days[3]):
        rows.append(dict(date=d, ticker="B", x="N", c="M", p=20.0, s=100.0, t=f"{d}T21:00:03Z"))
    rows.append(dict(date=days[1], ticker="A", x="N", c="M", p=10.0, s=500.0, t=f"{days[1]}T21:00:03Z"))
    A = pd.DataFrame(rows)
    raw = pd.DataFrame([dict(ticker=t, date=d, o=1.0, c=1.0, v=1000.0, vw=1.0, n=1) for t in "AB" for d in days])
    U = pd.DataFrame({"ticker": ["A", "B"], "book": "core", "exchange": "NYSE"})
    return caf.cross_panel(A, {"A": ("N",), "B": ("N",)}, raw, U, days).set_index(["ticker", "date"]).kind, days


def test_no_print_means_zero_only_before_the_m_era_and_missing_inside_it():
    kind, days = _panel_case()
    assert kind[("B", days[0])] == "no_round"            # before the venue's M era: no round-lot cross
    assert kind[("B", days[1])] == "no_round"            # an M and no 6
    assert kind[("B", days[2])] == "missing_M_session"   # no M for any name that session: missing data
    assert kind[("A", days[3])] == "round"
    assert kind[("B", days[3])] == "no_round"


def test_a_name_alone_without_any_print_in_the_m_era_is_missing():
    days = ["2021-01-04", "2021-01-05"]
    A = pd.DataFrame([dict(date=d, ticker="A", x="N", c="M", p=10.0, s=1.0, t=f"{d}T21:00:03Z") for d in days])
    raw = pd.DataFrame([dict(ticker=t, date=d, o=1.0, c=1.0, v=1000.0, vw=1.0, n=1) for t in "AB" for d in days])
    U = pd.DataFrame({"ticker": ["A", "B"], "book": "core", "exchange": "NYSE"})
    kind = caf.cross_panel(A, {"A": ("N",), "B": ("N",)}, raw, U, days).set_index(["ticker", "date"]).kind
    assert kind[("B", days[1])] == "missing_print"


def test_live_sessions_needed_is_80_percent_power_not_t_of_two():
    from scripts.research.inference import power_of_z
    sd, nb, I = 10.0, 35, 5.0
    N = caf.live_sessions_needed(I, sd, nb)
    assert power_of_z(I / (sd * math.sqrt(1 / N + 1 / nb))) >= 0.8
    assert power_of_z(I / (sd * math.sqrt(1 / (N - 1) + 1 / nb))) < 0.8       # the smallest such N
    # the old criterion (expected t of 2) gives a far smaller N, at about 50% power
    k_old = 2.0
    N_old = math.ceil(sd ** 2 / ((I / k_old) ** 2 - sd ** 2 / nb))
    assert N_old < N and power_of_z(2.0) < 0.55
    assert caf.live_sessions_needed(1.0, sd, nb) is None                         # unreachable on this baseline
    assert caf.live_sessions_needed(I, sd, None) == math.ceil((2.8015817 * sd / I) ** 2)


# ------------------------------------------------------------------------------- quotes

def test_time_weighted_half_spread_excludes_invalid_intervals_and_carries_the_prior_quote_in():
    lo, hi = 100 * NS, 200 * NS
    t = np.array([90, 150, 170], dtype=np.int64) * NS        # quote at 90 prevails at lo
    b = np.array([9.99, 9.98, 10.00])
    a = np.array([10.01, 10.02, 9.99])                        # the last is crossed: excluded
    v, cov = caf.tw_half_spread(t, b, a, lo, hi)
    hs1 = 0.02 / 20.0 * 1e4
    hs2 = 0.04 / 20.0 * 1e4
    assert cov == pytest.approx(70.0)                         # 50 s + 20 s; the crossed 30 s are not covered
    assert v == pytest.approx((hs1 * 50 + hs2 * 20) / 70)


def test_quote_at_returns_none_when_the_prevailing_quote_is_crossed():
    t = np.array([10, 20], dtype=np.int64) * NS
    b, a = np.array([9.99, 10.02]), np.array([10.01, 10.00])
    assert caf.quote_at(t, b, a, 15 * NS)[:2] == (9.99, 10.01)
    assert caf.quote_at(t, b, a, 25 * NS) is None             # never the stale earlier quote
    assert caf.quote_at(t, b, a, 5 * NS) is None


# ------------------------------------------------------------------------------- windows

def test_window_bars_take_only_bars_lying_wholly_inside():
    close = caf.close_ts(dt.date(2026, 11, 20), "16:00")
    starts = [close - pd.Timedelta(minutes=m) for m in (125, 120, 15, 10, 5)]
    B = pd.DataFrame({"t": starts, "v": [1, 2, 3, 4, 5]})
    got = caf.window_bars(B, close, 120, 10)
    assert list(got.v) == [2, 3]        # 14:00 and 15:45 bars; the 15:50 bar ends after the window


def test_bars5_daily_am_volume_ends_at_the_cap_time(tmp_path):
    (tmp_path / "bars5").mkdir()
    d = dt.date(2026, 11, 20)
    close = caf.close_ts(d, "16:00")
    t = [close - pd.Timedelta(minutes=m) for m in (390, 25, 20, 15, 10)]
    B = pd.DataFrame({"ticker": "X", "t": [x.strftime("%Y-%m-%dT%H:%M:%SZ") for x in t], "c": 10.0,
                      "v": [100.0, 10.0, 20.0, 1000.0, 5000.0], "vw": 10.0, "n": 1})
    B.to_parquet(tmp_path / "bars5" / "2026-11.parquet", index=False)
    D = caf.bars5_daily(tmp_path, {"2026-11-20": "16:00"})
    row = D.iloc[0]
    # 09:30, 15:35 and 15:40 bars end by 15:45; the 15:45 bar ends at 15:50, after the cap time
    assert row.v_am == 130.0
    assert row.am_last_end_mbc == pytest.approx(15.0)
    assert row.v_120_10 == 1030.0       # 15:35, 15:40, 15:45 bars
    assert row.v_full == 6130.0


def test_a_window_without_bars_on_a_traded_day_is_zero_not_missing(tmp_path):
    (tmp_path / "bars5").mkdir()
    d = dt.date(2026, 11, 20)
    close = caf.close_ts(d, "16:00")
    t = [close - pd.Timedelta(minutes=m) for m in (390, 200)]            # 09:30 and 12:40 only
    B = pd.DataFrame({"ticker": "X", "t": [x.strftime("%Y-%m-%dT%H:%M:%SZ") for x in t], "c": 10.0,
                      "v": [100.0, 50.0], "vw": 10.0, "n": 1})
    B.to_parquet(tmp_path / "bars5" / "2026-11.parquet", index=False)
    row = caf.bars5_daily(tmp_path, {"2026-11-20": "16:00"}).iloc[0]
    assert row.usd_30_10 == 0.0 and row.usd_120_10 == 0.0 and row.usd_am == 1500.0
    assert np.isnan(row.vwap_30_10)


# ------------------------------------------------------------------------------- alignment

def test_causal_features_do_not_see_the_current_row():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    a = caf.causal_rolling_median(x, 3, 1)
    y = x.copy()
    y.iloc[4] = 1e9
    b = caf.causal_rolling_median(y, 3, 1)
    assert a.iloc[4] == b.iloc[4] == 3.0
    e1 = caf.causal_ewm_mean(x, 2, 1)
    e2 = caf.causal_ewm_mean(y, 2, 1)
    assert e1.iloc[4] == e2.iloc[4]


def _synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-05", periods=n)
    ds = [d.date().isoformat() for d in days]
    X = pd.DataFrame({"date": ds, "ticker": "X", "kind": "round", "shares": rng.integers(100, 5000, n).astype(float),
                      "price": 10.0, "day_v": 1e5, "day_c": 10.0, "day_usd": rng.uniform(5e5, 2e6, n),
                      "book": "core", "other_listing_6": False})
    X["usd"] = X.shares * X.price
    X.loc[7, ["kind", "shares", "usd"]] = ["no_round", 0.0, 0.0]
    X["share_of_day"] = X.shares / X.day_v
    D5 = pd.DataFrame({"date": ds, "ticker": "X", "usd_am": rng.uniform(1e5, 1e6, n)})
    cal = caf.calendar_flags([d.date() for d in days], {d.date(): "16:00" for d in days})
    cal.index = ds
    cal.index.name = "date"
    E = pd.DataFrame({"ticker": ["X"], "ex_date": [ds[30]]})
    return X, D5, cal, E


def test_build_features_row_d_does_not_change_when_d_and_later_outcomes_change():
    X, D5, cal, E = _synthetic()
    F = caf.build_features(X, D5, cal, E, {"core": "2026-12-31"})
    i = 60
    X2 = X.copy()
    X2.loc[i:, "usd"] *= 50.0                  # D's own cross and everything after it
    X2.loc[i:, "day_usd"] *= 50.0
    D52 = D5.copy()
    D52.loc[i + 1:, "usd_am"] *= 50.0          # later days' intraday volume
    F2 = caf.build_features(X2, D52, cal, E, {"core": "2026-12-31"})
    for c in ("med20", "med60", "ewm10", "naive", "zero_rate60", "u_am", "u_day1", "u_x1"):
        a, b = F.loc[i, c], F2.loc[i, c]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), c


def test_truncation_check_catches_a_feature_that_reads_the_current_row(monkeypatch):
    X, D5, cal, E = _synthetic()
    F = caf.build_features(X, D5, cal, E, {"core": "2026-12-31"})
    assert caf.truncation_check(F, X, D5, cal, E, {"core": "2026-12-31"}, n=10) > 0
    # the old, wrong behaviour: a rolling median that includes row D itself
    monkeypatch.setattr(caf, "causal_rolling_median", lambda x, n, mp: x.rolling(n, min_periods=mp).median())
    F_bad = caf.build_features(X, D5, cal, E, {"core": "2026-12-31"})
    with pytest.raises(caf.StudyError, match="lookahead"):
        caf.truncation_check(F_bad, X, D5, cal, E, {"core": "2026-12-31"}, n=10)


def test_no_round_day_is_minus_infinity_and_counts_below_every_quantile():
    X, D5, cal, E = _synthetic()
    F = caf.build_features(X, D5, cal, E, {"core": "2026-12-31"})
    assert np.isneginf(F.loc[7, "ly"]) and pd.isna(F.loc[7, "ly_pos"])
    e = np.array([-np.inf] * 15 + list(np.linspace(-1, 1, 85)))
    assert np.isneginf(caf.empirical_quantile(e, 0.10))
    assert np.isfinite(caf.empirical_quantile(e, 0.25))


def test_pre_ex_uses_the_exchange_calendar_not_the_names_next_row():
    X, D5, cal, E = _synthetic()
    X = X.drop(index=31).reset_index(drop=True)    # the name does not trade on the ex-date
    F = caf.build_features(X, D5, cal, E, {"core": "2026-12-31"})
    assert bool(F.loc[F.date == cal.index[29], "pre_ex"].iloc[0]) is True
    assert bool(F.loc[F.date == cal.index[30], "pre_ex"].iloc[0]) is False


# ------------------------------------------------------------------------------- calendar

def test_calendar_flags():
    days = [dt.date(2026, 3, 18), dt.date(2026, 3, 19), dt.date(2026, 3, 20), dt.date(2026, 3, 31),
            dt.date(2026, 4, 1), dt.date(2026, 4, 2), dt.date(2026, 4, 6), dt.date(2026, 11, 25),
            dt.date(2026, 11, 27), dt.date(2026, 11, 30)]
    closes = {d: "16:00" for d in days}
    closes[dt.date(2026, 11, 27)] = "13:00"
    C = caf.calendar_flags(days, closes)
    assert C.loc[dt.date(2026, 3, 20)].quad and C.loc[dt.date(2026, 3, 20)].opex
    assert C.loc[dt.date(2026, 3, 31)].quarter_end and C.loc[dt.date(2026, 3, 31)].month_end
    assert not C.loc[dt.date(2026, 3, 19)].month_end
    # Good Friday 2026-04-03 is a holiday but not the third Friday (04-17): Thursday 04-02 is no expiry
    assert not C.loc[dt.date(2026, 4, 2)].opex
    assert C.loc[dt.date(2026, 11, 27)].half_day and C.loc[dt.date(2026, 11, 25)].pre_half_day
    assert caf.third_friday(2026, 4) == dt.date(2026, 4, 17)


def test_opex_moves_to_thursday_when_the_third_friday_is_a_holiday():
    # 2022-04-15 (Good Friday) was the third Friday of April 2022; expiry was Thursday 04-14
    days = [dt.date(2022, 4, 13), dt.date(2022, 4, 14), dt.date(2022, 4, 18)]
    C = caf.calendar_flags(days, {d: "16:00" for d in days})
    assert C.loc[dt.date(2022, 4, 14)].opex and not C.loc[dt.date(2022, 4, 13)].opex


# ------------------------------------------------------------------------------- the cap

def test_cap_split():
    moc, rest = caf.cap_split(np.array([1000.0, 1000.0, 50.0]), np.array([20000.0, 0.0, 20000.0]), 0.10)
    assert list(moc) == [1000.0, 0.0, 50.0] and list(rest) == [0.0, 1000.0, 0.0]
    moc, rest = caf.cap_split(np.array([5000.0]), np.array([20000.0]), 0.20)
    assert moc[0] == 4000.0 and rest[0] == 1000.0
    with pytest.raises(caf.StudyError):
        caf.cap_split(np.array([1.0]), np.array([np.nan]), 0.1)


def test_weighted_median():
    assert caf.weighted_median([1, 2, 3], [1, 1, 10]) == 3
    assert caf.weighted_median([1, 2, 3], [1, 1, 1]) == 2


# ------------------------------------------------------------------------------- the public output

def test_check_aggregates_refuses_symbols_outside_the_core_market_block_and_ids():
    out = {"auction_size": {"market_by_name_core": {"PDI": {"med_usd": 1}}, "x": 1},
           "afternoon_pool": {"market_by_name_core": {"PDI": {}}}, "cap": {"a": 1}}
    caf.check_aggregates(out, ["PDI", "ADX"], [{"id": "abc-123", "client_order_id": "cef-xyz"}])
    bad = dict(out, cap={"PDI": 1})
    with pytest.raises(caf.StudyError):
        caf.check_aggregates(bad, ["PDI"], [])
    bad2 = dict(out, cap={"note": "cef-xyz"})
    with pytest.raises(caf.StudyError):
        caf.check_aggregates(bad2, ["PDI"], [{"id": "abc-123", "client_order_id": "cef-xyz"}])
