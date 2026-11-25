"""Render the card's tables from results.json, so no figure is copied by hand.

    python3 desk/results/ideas/levered_etf_close/render.py > /tmp/tables.md

Every number printed here is read from results.json beside this file (written by explore.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(v, p=".2f"):
    return "n/a" if v is None else format(v, p)


def book_rows(o: dict, names, leg="leg1") -> list:
    out = ["| book | events | Sharpe (NW t) | return / vol a year | max DD | per event, mean bp (t) | median bp | hit | half-spread at entry, median bp |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        if n not in o:
            continue
        s = o[n][leg]["full"]
        e = o[n]["per_event_leg1" if leg == "leg1" else "per_event_leg2"]
        out.append(f"| {n} | {e.get('events', 0):,} | {f(s['sharpe'])} ({f(s['nw_t'])}) | {f(s['ann_ret'], '.1%')} / {f(s['ann_vol'], '.1%')} | "
                   f"{f(s['max_drawdown'], '.1%')} | {f(e.get('mean_bp'), '.1f')} ({f(e.get('nw_t_daily_mean'))}) | {f(e.get('median_bp'), '.1f')} | "
                   f"{f(e.get('hit'), '.0%')} | {f(e.get('median_quoted_half_spread_bp'), '.1f')} |")
    return out


def split_rows(o: dict, names, leg="leg1") -> list:
    keys = ["full", "early (to 2022)", "late (2023 on)", "2016-19", "2020-22", "2023-25", "2026",
            "stressed (SPY 21d vol >= 20%)=True", "stressed (SPY 21d vol >= 20%)=False",
            "uptrend (SPY above 200d average)=True", "uptrend (SPY above 200d average)=False"]
    lab = ["full", "2016–22", "2023–25", "2016–19", "2020–22", "2023–25", "2026", "stressed", "calm", "uptrend", "downtrend"]
    have = [(k, l) for k, l in zip(keys, lab) if any(o[n][leg].get(k) for n in names if n in o)]
    out = ["| book | " + " | ".join(l for _, l in have) + " |", "|---|" + "---:|" * len(have)]
    for n in names:
        if n not in o:
            continue
        cells = []
        for k, _ in have:
            s = o[n][leg].get(k)
            cells.append("" if not s else f"{f(s['sharpe'])} ({f(s['nw_t'], '.1f')})")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    return out


def main() -> int:
    R = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "results.json")).read_text())
    print(f"panel last session {R['panel_last_session']}; CRSP last {R['crsp_last']}; entries scored {R.get('entries_scored')}\n")
    for lab in [k for k in R if isinstance(R[k], dict) and "T" in R[k]]:
        o = R[lab]
        thr = o["thresholds"]
        print(f"\n### {lab} ({o['first']} to {o['last']}); point-in-time S* from {thr['S']['first_session']}, by year {thr['S']['by_year_mean']}; "
              f"T {o['T_events']:,} events in {o['T_underlyings']} names\n")
        names = ["T", "T-BIG", "T-cont", "T-SA", "C-mom", "C-mom-Tnames", "C-big", "C-open", "P-low"] + [k for k in o if k.startswith("P-none")]
        print("Leg 1, entry close − 30 min to the closing cross:\n")
        print("\n".join(book_rows(o, names, "leg1")))
        print("\nLeg 2, closing cross to the next opening cross (side reversed):\n")
        print("\n".join(book_rows(o, names, "leg2")))
        print("\nSplits and regimes, leg 1, Sharpe (NW t):\n")
        print("\n".join(split_rows(o, ["T", "T-BIG", "T-cont", "C-mom", "C-mom-Tnames", "C-big", "P-low"] + [k for k in o if k.startswith("P-none")], "leg1")))
        print("\nSplits and regimes, leg 2:\n")
        print("\n".join(split_rows(o, ["T", "C-big", "P-low"], "leg2")))
        print("\nBook on book, leg 1 (daily P&L, HAC, 5 lags; t at the Newey-West rule's lags beside):\n")
        print("| regression | alpha bp a day (t) | t at rule lags | beta (t) | R² |\n|---|---:|---:|---:|---:|")
        for k, v in o["book_regressions_leg1"].items():
            print(f"| {k} | {v['alpha_bp_per_day']:.2f} ({v['alpha_t']:.2f}) | {v['alpha_t_at_nw_rule_lags']:.2f} (L={v['nw_rule_lags']}) | {v['beta']:.2f} ({v['beta_t']:.1f}) | {v['r2']:.2f} |")
        tc = o.get("threshold_comparison")
        if tc:
            print("\nThreshold both ways (each from the first session its point-in-time threshold exists):\n")
            print("| book | threshold | from | events | Sharpe (NW t) | to 2022 | 2023 on | same-names control | alpha t vs it, 5 lags / rule lags |\n|---|---|---|---:|---:|---:|---:|---:|---:|")
            for bk, v in tc.items():
                for lab_, x in v.items():
                    if lab_ == "from":
                        continue
                    print(f"| {bk} | {lab_} | {v['from']} | {x['events']:,} | {f(x['sharpe'])} ({f(x['nw_t'])}) | {f(x['to_2022'])} | {f(x['2023_on'])} | "
                          f"{f(x['same_names_control_sharpe'])} | {f(x['alpha_t_on_same_names_5_lags'])} / {f(x['alpha_t_on_same_names_rule_lags'])} |")
        dec = o.get("leg2_gap_us_decomposition")
        if dec:
            print("\nLeg 2 taken apart: " + "; ".join(f"{k}: {v}" if isinstance(v, int) else
                  f"{k}: {f(v.get('mean_bp'), '.1f')}bp (t {f(v.get('nw_t_daily_mean'))}, n {v.get('events')})" for k, v in dec.items()))
        print("\nPooled name-days, date-clustered (y in bp; |r|, S, SA in percent):\n")
        print("| model | n | const (t) | |r| (t) | S (t) | SA (t) |\n|---|---:|---:|---:|---:|---:|")
        for k, v in o["panel_regressions"].items():
            if not isinstance(v, dict):
                continue
            c = lambda x: f"{v[x + '_coef']:.3f} ({v[x + '_t']:.2f})" if x + "_coef" in v else ""  # noqa: E731
            print(f"| {k} | {v['n']:,} | {c('const')} | {c('absr')} | {c('S')} | {c('SA')} |")
        print("\nDose, big-move days with a fund alive (per event, bp (t)):\n")
        print("| bucket | events | leg 1 | leg 2 |\n|---|---:|---:|---:|")
        for k, v in o["dose_big_move_days_with_a_fund"].items():
            a, b = v["leg1"], v["leg2"]
            print(f"| {k} | {a.get('events', 0):,} | {f(a.get('mean_bp'), '.1f')} ({f(a.get('nw_t_daily_mean'))}) | {f(b.get('mean_bp'), '.1f')} ({f(b.get('nw_t_daily_mean'))}) |")
        print("\nT by year (per event):\n")
        print("| year | events | leg 1 bp (t) | half-spread bp | leg 2 bp (t) |\n|---|---:|---:|---:|---:|")
        for y, v in o["T_by_year"].items():
            a, b = v["leg1"], v["leg2"]
            if a.get("events", 0) >= 3:
                print(f"| {y} | {a['events']:,} | {f(a.get('mean_bp'), '.1f')} ({f(a.get('nw_t_daily_mean'))}) | {f(a.get('median_quoted_half_spread_bp'), '.1f')} | {f(b.get('mean_bp'), '.1f')} ({f(b.get('nw_t_daily_mean'))}) |")
        sp = o.get("session_number_split_leg1")
        if sp:
            print("\nSharpe by session number (mod 2 = the neighbours' split; mod 3 = the headline fetch's processes): "
                  + "; ".join(f"{n} mod2 {[f(x) for x in v['mod2']]} mod3 {[f(x) for x in v['mod3']]}" for n, v in sp.items()))
        c = o["T_concentration"]
        print(f"\nConcentration: top three {c['top3']} carry {c['top3_share_of_events']:.0%} of T's events; without them leg 1 "
              f"Sharpe {f(c['without_top3_leg1']['sharpe'])} (t {f(c['without_top3_leg1']['nw_t'])}), per event "
              f"{f(c['without_top3_per_event_leg1'].get('mean_bp'), '.1f')}bp; share of names with 30+ events and a positive mean {f(c['share_of_names_positive_leg1'], '.0%')}")
        for n in ("T", "C-big", "P-low"):
            d = o[n].get("per_event_t1_open_to_close_fade_diagnostic", {})
            print(f"diagnostic, fade t+1 open to close, {n}: {f(d.get('mean_bp'), '.1f')}bp (t {f(d.get('nw_t_daily_mean'))}), n {d.get('events')}")
        L3 = o.get("leg3_next_session_fade_added_after_interim_read")
        if L3:
            print(f"\nLeg 3 (added after the interim read): fade the whole day's move from t+1's opening cross to its closing cross; S3* from {L3['S3_threshold_first_session']}\n")
            print("| book | events | Sharpe (NW t) | return / vol | 2016–22 | 2023–25 | 2026 | stressed | calm | per event bp (t) | hit |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for n in ("T3", "T3-BIG", "C3-all", "C3-big", "C3-Tnames", "P3-low"):
                v = L3[n]; b_ = v["book"]; e = v["per_event"]
                g = lambda k: "" if not b_.get(k) else f"{f(b_[k]['sharpe'])}"  # noqa: E731
                print(f"| {n} | {v['events']:,} | {f(b_['full']['sharpe'])} ({f(b_['full']['nw_t'])}) | {f(b_['full']['ann_ret'], '.1%')} / {f(b_['full']['ann_vol'], '.1%')} | "
                      f"{g('early (to 2022)')} | {g('late (2023 on)')} | {g('2026')} | {g('stressed (SPY 21d vol >= 20%)=True')} | {g('stressed (SPY 21d vol >= 20%)=False')} | "
                      f"{f(e.get('mean_bp'), '.1f')} ({f(e.get('nw_t_daily_mean'))}) | {f(e.get('hit'), '.0%')} |")
            for k, v in L3.items():
                if " on " in k:
                    print(f"- {k}: alpha {v['alpha_bp_per_day']:.2f}bp a day (t {v['alpha_t']:.2f}; {v['alpha_t_at_nw_rule_lags']:.2f} at {v['nw_rule_lags']} lags), beta {v['beta']:.2f}, R² {v['r2']:.2f}")
            x = o.get("exposures_T3")
            if x:
                b = x["daily_betas"]
                print(f"- T3 exposures: SPY {b['SPY']['beta']:.3f} (t {b['SPY']['t']:.1f}), IEF {b['IEF']['beta']:.3f}, HYG {b['HYG']['beta']:.3f}; "
                      + "; ".join(f"{kk[9:]} {vv['corr']:.3f}" for kk, vv in x.items() if kk.startswith("corr_with_")))
        print("\nNeighbours (leg 1):\n")
        print("| entry, book | Sharpe (NW t) | per event bp (t) | half-spread bp |\n|---|---:|---:|---:|")
        for k, v in o["neighbours"].items():
            e = v["per_event_leg1"]
            print(f"| {k} | {f(v['leg1']['sharpe'])} ({f(v['leg1']['nw_t'])}) | {f(e.get('mean_bp'), '.1f')} ({f(e.get('nw_t_daily_mean'))}) | {f(e.get('median_quoted_half_spread_bp'), '.1f')} |")
        print("\nExposures:\n")
        for k in ("exposures_T_leg1", "exposures_T_leg2", "exposures_C-big_leg1"):
            x = o[k]
            b = x["daily_betas"]
            cs = "; ".join(f"{kk[9:]} {vv['corr']:.3f} ({vv['days']} days to {vv['last']})" for kk, vv in x.items() if kk.startswith("corr_with_"))
            print(f"- {k}: SPY {b['SPY']['beta']:.3f} (t {b['SPY']['t']:.1f}), IEF {b['IEF']['beta']:.3f} (t {b['IEF']['t']:.1f}), HYG {b['HYG']['beta']:.3f} "
                  f"(t {b['HYG']['t']:.1f}); in-window beta to SPY {x['beta_to_spy_same_window_active_days']['beta']:.3f} "
                  f"(t {x['beta_to_spy_same_window_active_days']['t']:.1f}); correlation {cs}")
        print("\nCapacity, T:\n")
        print("| capital | positions | median % of ADV | max % ADV | share > 1% ADV | median % of close cross | max % cross | share > 5% cross | both legs, median % cross | both legs, share > 5% |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in o["capacity_T"].items():
            print(f"| {k} | {v['positions']:,} | {v['median_pct_adv']:.2f} | {v['max_pct_adv']:.1f} | {v['share_above_1pct_adv']:.0%} | {v['median_pct_close_cross']:.1f} | "
                  f"{v['max_pct_close_cross']:.0f} | {v['share_above_5pct_cross']:.0%} | {v['both_legs_one_close_order_median_pct_cross']:.1f} | {v['both_legs_share_above_5pct_cross']:.0%} |")
    print("\nexclusions:", json.dumps(R["exclusions"], indent=0))
    print("prints:", R["prints"]); print("factor check:", R["distribution_factor_check"]); print("assets check:", R["assets_crsp_vs_proshares"])
    print("ProShares share of 2025 k, all index:", round(R["proshares_share_of_2025_k_all_index"], 3))
    print("leveraged assets by year:", R["leveraged_assets_by_year"])
    print("truncation:", R.get("truncation_check_index_T")); print("held sleeves:", R["held_sleeves"])
    print("g_t check:", R.get("g_t_check_index_T"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
