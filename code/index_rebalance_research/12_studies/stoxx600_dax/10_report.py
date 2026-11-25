"""Step 10: summary statistics and table CSVs (table_*.csv) + report_tables.md, both families.

t-stats: mean / cluster-robust SE, clusters = review (common.clt). Spread t: across reviews.
Robustness: median, 10% trimmed mean, and the mean with each CAR floored at -100% (CAR is a sum of daily ARs, so a
collapsing stock can sum to below -100% although no long can lose, and no short gain, more than 100%).
Tax split: buy taxed at formation (UK, IE, FR from Aug 2012, IT from Mar 2013, ES from Jan 2021, BE unconfirmed) vs
untaxed; drag = the buy-leg rate (one buy per round trip). 'taxgroup' splits the taxed side into UK and the rest.
Hit split: predicted names that did / did not change at that review (table_hit_split_*.csv).
"""
import numpy as np, pandas as pd
from scipy.stats import trim_mean
from common import OUT, clt, cell

md = []
SEG = [("H6", "H3", "H6→H3"), ("H3", "C", "H3→C"), ("C", "A", "C→A"), ("A", "Tm1", "A→T−1"),
       ("Tm1", "T", "T−1→T (trade day)"), ("T", "Tp1", "T→T+1"), ("T", "Tp20", "T→T+20"), ("T", "Tp60", "T→T+60")]
FAM = {"stoxx": dict(sides=[("add", "Adds (long)"), ("del", "Deletes (short leg; negative = profit for a short)")],
                     extra=["mem", "v0"], label="STOXX Europe 600"),
       "dax": dict(sides=[("up", "Promotions (long)"), ("down", "Demotions (short leg; negative = profit for a short)")],
                   extra=["mem"], label="DAX family")}


def acc(df, pc, ac):
    tp = (df[pc] & df[ac]).sum(); fp = (df[pc] & ~df[ac]).sum(); fn = (~df[pc] & df[ac]).sum()
    return pd.Series(dict(n_pred=int(tp + fp), n_actual=int(tp + fn), hits=int(tp), precision=tp / max(tp + fp, 1),
                          recall=tp / max(tp + fn, 1)))


# ======================================================== accuracy
ps = pd.read_csv(OUT / "predictions_stoxx.csv", dtype={"gvkey": str})
pdx = pd.read_csv(OUT / "predictions_dax.csv", dtype={"gvkey": str})
ac = []
for h in ["H6", "H3", "C"]:
    x = ps[ps.horizon == h]
    for model, sfx in [("rules", ""), ("rules + memory", "_mem"), ("rules + M&A", "_ma"), ("v0 (no eligibility proxies)", "_v0"),
                       ("naive size", None)]:
        if sfx == "_v0" and h == "H6":
            continue
        for side, a in [("add", "act_add"), ("delete", "act_del")]:
            pc = f"pred_{side[:3]}{sfx}" if sfx is not None else f"naive_{side[:3]}"
            r = acc(x, pc, a); r["family"] = "STOXX 600"; r["horizon"] = h; r["model"] = model; r["side"] = side
            ac.append(r)
    x = pdx[pdx.horizon == h]
    for model, pre, suf in [("rules", "pred", ""), ("rules + memory", "pred", "_mem"), ("naive size", "naive", "")]:
        for side in ["up", "down"]:
            r = acc(x, f"{pre}_{side}{suf}", f"act_{side}"); r["family"] = "DAX family"; r["horizon"] = h; r["model"] = model
            r["side"] = "promotion" if side == "up" else "demotion"; ac.append(r)
    for nm in ["DAX", "MDAX", "SDAX", "TecDAX"]:
        for side in ["add", "del"]:
            r = acc(x, f"pred_{side}_{nm}", f"act_{side}_{nm}"); r["family"] = "DAX family"; r["horizon"] = h
            r["model"] = f"rules, {nm}"; r["side"] = "add" if side == "add" else "delete"; ac.append(r)
ac = pd.DataFrame(ac)[["family", "horizon", "model", "side", "n_pred", "n_actual", "hits", "precision", "recall"]]
ac[["precision", "recall"]] = ac[["precision", "recall"]].round(3)
ac.to_csv(OUT / "table_accuracy.csv", index=False)
md.append("### Prediction accuracy (pooled)\n\n" + ac.to_markdown(index=False) + "\n")
ay = []
for fam, df, sides in [("STOXX 600", ps, [("add", "pred_add", "act_add"), ("delete", "pred_del", "act_del")]),
                       ("DAX family", pdx, [("promotion", "pred_up", "act_up"), ("demotion", "pred_down", "act_down")])]:
    for (h, y), x in df.groupby(["horizon", "year"]):
        for side, pc, a in sides:
            r = acc(x, pc, a); r["family"] = fam; r["horizon"] = h; r["year"] = y; r["side"] = side; ay.append(r)
ay = pd.DataFrame(ay)
ay.to_csv(OUT / "table_accuracy_by_year.csv", index=False)
ay["period"] = np.where(ay.year <= 2015, "2008–2015", "2016–2026")
ap = ay.groupby(["family", "horizon", "side", "period"])[["n_pred", "n_actual", "hits"]].sum()
ap["precision"] = (ap.hits / ap.n_pred).round(3); ap["recall"] = (ap.hits / ap.n_actual).round(3)
ap = ap.reset_index(); ap.to_csv(OUT / "table_accuracy_by_period.csv", index=False)
md.append("### Accuracy by period (rules model)\n\n" + ap.to_markdown(index=False) + "\n")
era = pdx.groupby(["horizon", "era"]).apply(lambda x: pd.concat([acc(x, "pred_up", "act_up").add_prefix("up_"),
                                                                 acc(x, "pred_down", "act_down").add_prefix("down_")]),
                                            include_groups=False).reset_index()
era.to_csv(OUT / "table_accuracy_dax_by_era.csv", index=False)
fs = ps[ps.horizon == "C"].groupby("ff_src").apply(lambda x: pd.concat([acc(x, "pred_add", "act_add").add_prefix("add_"),
                                                                       acc(x, "pred_del", "act_del").add_prefix("del_")]),
                                                   include_groups=False).reset_index()
fs.to_csv(OUT / "table_accuracy_stoxx_by_float_source.csv", index=False)
md.append("### DAX family accuracy by rule era\n\n" + era.round(3).to_markdown(index=False) + "\n")
md.append("### STOXX accuracy at C by float source of the name\n\n" + fs.round(3).to_markdown(index=False) + "\n")

# ======================================================== CARs
for fam, cfg in FAM.items():
    ce = pd.read_csv(OUT / f"car_events_{fam}.csv", dtype={"gvkey": str})
    ce["period"] = np.where(ce.year <= 2015, "2008–2015", "2016–2026")
    ce["taxed"] = np.where(ce.tax > 0, "buy taxed", "untaxed")
    ce["taxgroup"] = np.where(ce.cty == "GBR", "UK (taxed)", np.where(ce.tax > 0, "other taxed (FR/IT/ES/IE/BE)", "untaxed"))
    for side, sname in cfg["sides"]:
        for pref, lab in [("car", "matched-control"), ("cari", "index-adjusted")]:
            cols = {}
            specs = [("Hindsight (actual), formed H6", "H6", f"hind_{side}"), ("Predicted at H6", "H6", f"pred_{side}"),
                     ("Predicted at H3", "H3", f"pred_{side}"), ("Predicted at C (cut-off)", "C", f"pred_{side}")]
            if "mem" in cfg["extra"]:
                specs += [("Rules+memory at H3", "H3", f"mem_{side}"), ("Rules+memory at C", "C", f"mem_{side}")]
            if "v0" in cfg["extra"]:
                specs += [("Rules without proxies (v0) at H3", "H3", f"v0_{side}"), ("v0 at C", "C", f"v0_{side}")]
            specs += [("Naive at C", "C", f"naive_{side}")]
            for name, form, port in specs:
                x = ce[(ce.form == form) & (ce.portfolio == port)]
                c = {"n names": f"{len(x)}", "n reviews": f"{x.review.nunique()}", "horizon→T": cell(x[f"{pref}_{form}_T"], x.review)}
                for a, b, l in SEG:
                    c[l] = cell(x[f"{pref}_{a}_{b}"], x.review)
                cols[name] = c
            t = pd.DataFrame(cols)
            t.to_csv(OUT / f"table_car_{fam}_{side}_{pref}.csv")
            md.append(f"### {cfg['label']}: CAR by segment, {sname}, {lab} (% mean, cluster t by review)\n\n" + t.to_markdown() + "\n")
    # periods, tax regime
    bp = []
    s1, s2 = cfg["sides"][0][0], cfg["sides"][1][0]
    plist = [(f"Hindsight {s1} (H6)", "H6", f"hind_{s1}"), (f"Pred {s1} H6", "H6", f"pred_{s1}"), (f"Pred {s1} H3", "H3", f"pred_{s1}"),
             (f"Pred {s1} C", "C", f"pred_{s1}"), (f"Hindsight {s2} (H6)", "H6", f"hind_{s2}"), (f"Pred {s2} H6", "H6", f"pred_{s2}"),
             (f"Pred {s2} H3", "H3", f"pred_{s2}"), (f"Pred {s2} C", "C", f"pred_{s2}")]
    plist += [(f"Rules+memory {s1} H3", "H3", f"mem_{s1}"), (f"Rules+memory {s1} C", "C", f"mem_{s1}"),
              (f"Rules+memory {s2} H3", "H3", f"mem_{s2}"), (f"Rules+memory {s2} C", "C", f"mem_{s2}")]
    for split in ["period", "taxed", "taxgroup"] if fam == "stoxx" else ["period"]:
        for k, g0 in ce.groupby(split):
            for name, form, port in plist:
                x = g0[(g0.form == form) & (g0.portfolio == port)]
                r = {"split": split, "group": k, "portfolio": name, "n": len(x), "horizon→T": cell(x[f"car_{form}_T"], x.review)}
                if split in ("taxed", "taxgroup"):
                    r["mean buy tax %"] = round(x.tax.mean() * 100, 2) if len(x) else np.nan
                for a, b, l in SEG:
                    r[l] = cell(x[f"car_{a}_{b}"], x.review)
                bp.append(r)
    bp = pd.DataFrame(bp)
    bp.to_csv(OUT / f"table_car_splits_{fam}.csv", index=False)
    md.append(f"### {cfg['label']}: segment CARs by period" + (" and tax regime" if fam == "stoxx" else "") +
              " (matched-control, % mean, cluster t)\n\n" + bp.to_markdown(index=False) + "\n")
    # robustness
    rob = []
    for form in ["H6", "H3", "C"]:
        for port in [f"pred_{s1}", f"pred_{s2}", f"hind_{s1}", f"hind_{s2}", f"mem_{s1}", f"mem_{s2}", f"v0_{s1}", f"v0_{s2}"]:
            x = ce[(ce.form == form) & (ce.portfolio == port)]
            for w in [f"{form}_T", "T_Tp1", "T_Tp20", "T_Tp60"]:
                v = x[f"car_{w}"].dropna()
                if len(v) < 5:
                    continue
                vf = v.clip(lower=-1.0)
                rob.append(dict(form=form, portfolio=port, window=w.replace("_", "→").replace("Tp", "T+"), n=len(v),
                                mean=v.mean() * 100, t_cluster=clt(v, x.loc[v.index, "review"]), median=v.median() * 100,
                                trim10=trim_mean(v, 0.1) * 100, mean_floor100=vf.mean() * 100,
                                t_floor100=clt(vf, x.loc[v.index, "review"]), n_below_m100=int((v < -1).sum()),
                                share_pos=(v > 0).mean()))
    rob = pd.DataFrame(rob).round(2)
    rob.to_csv(OUT / f"table_robustness_{fam}.csv", index=False)
    md.append(f"### {cfg['label']}: robustness (mean vs median vs 10% trimmed vs mean with CAR floored at -100%, "
              f"matched-control CAR %)\n\n" + rob.to_markdown(index=False) + "\n")
    # predicted names that did / did not change at the review
    hs = []
    for form in ["H6", "H3", "C"]:
        for port in [f"pred_{s1}", f"pred_{s2}"]:
            x = ce[(ce.form == form) & (ce.portfolio == port)]
            for k, g in [("all", x), ("changed (hit)", x[x.hit]), ("did not change", x[~x.hit])]:
                hs.append({"form": form, "portfolio": port, "group": k, "n": len(g),
                           "horizon→T": cell(g[f"car_{form}_T"], g.review),
                           "median horizon→T": round(g[f"car_{form}_T"].median() * 100, 2) if len(g) else np.nan})
    hs = pd.DataFrame(hs)
    hs.to_csv(OUT / f"table_hit_split_{fam}.csv", index=False)
    md.append(f"### {cfg['label']}: predicted names that changed vs did not (matched-control CAR %, cluster t)\n\n" +
              hs.to_markdown(index=False) + "\n")
    # spread per review
    sp = []
    for form in ["H6", "H3", "C"]:
        for kind, (pa, pdl) in {"rules": (f"pred_{s1}", f"pred_{s2}"), "hindsight": (f"hind_{s1}", f"hind_{s2}"),
                                "naive": (f"naive_{s1}", f"naive_{s2}"), "rules+memory": (f"mem_{s1}", f"mem_{s2}"),
                                "v0 (no proxies)": (f"v0_{s1}", f"v0_{s2}")}.items():
            for w in [f"{form}_T", "T_Tp20"]:
                a = ce[(ce.form == form) & (ce.portfolio == pa)].groupby("review")[f"car_{w}"].mean()
                d = ce[(ce.form == form) & (ce.portfolio == pdl)].groupby("review")[f"car_{w}"].mean()
                s = (a - d).dropna()
                if s.empty:
                    continue
                yrs = pd.Series(s.index.str[:4].astype(int), index=s.index)
                for per, msk in [("2008–2026", yrs > 0), ("2008–2015", yrs <= 2015), ("2016–2026", yrs >= 2016)]:
                    v = s[msk]
                    sp.append(dict(model=kind, form=form, window=w.replace("_", "→").replace("Tp", "T+"), period=per,
                                   n_reviews=len(v), mean_spread=v.mean() * 100,
                                   t=v.mean() / (v.std() / np.sqrt(len(v))) if len(v) > 2 else np.nan,
                                   median=v.median() * 100, share_pos=(v > 0).mean()))
    sp = pd.DataFrame(sp).round(2)
    sp.to_csv(OUT / f"table_spread_{fam}.csv", index=False)
    md.append(f"### {cfg['label']}: long {s1} / short {s2} spread per review (%; t across reviews)\n\n" + sp.to_markdown(index=False) + "\n")
    # per-year + profitable years
    spec = [("H6", f"pred_{s1}", "car_H6_T"), ("H6", f"pred_{s2}", "car_H6_T"), ("H3", f"pred_{s1}", "car_H3_T"),
            ("H3", f"pred_{s2}", "car_H3_T"), ("C", f"pred_{s1}", "car_C_T"), ("C", f"pred_{s2}", "car_C_T"),
            ("C", f"hind_{s1}", "car_T_Tp20"), ("C", f"hind_{s2}", "car_T_Tp20")]
    py, pos = {}, {}
    for f, p_, c in spec:
        x = ce[(ce.form == f) & (ce.portfolio == p_)]
        t = x.groupby("year")[c].agg(["mean", "size"])
        lab = f"{p_} {c[4:].replace('_', '→').replace('Tp', 'T+')}"
        py[lab] = (t["mean"].mul(100).round(1).map("{:+.1f}".format) + " (" + t["size"].astype(str) + ")")
        sign = 1 if p_.endswith(s1) else -1
        if c.endswith("Tp20"):
            sign = -sign
        m = t["mean"].dropna()
        pos[lab] = f"{int((sign * m > 0).sum())} of {len(m)}"
    py = pd.DataFrame(py)
    py.loc["profitable years"] = pd.Series(pos)
    py.index.name = "year"
    py.to_csv(OUT / f"table_per_year_{fam}.csv")
    md.append(f"### {cfg['label']}: per-year mean CAR % (n names); last row = years in which the trade made money "
              f"(long {s1}/short {s2} before T; the reversal trade after T)\n\n" + py.to_markdown() + "\n")
    if fam == "dax":
        mv = []
        ce["mv"] = ce.move.fillna("")
        for form in ["C", "H3"]:
            x = ce[(ce.form == form) & ce.portfolio.str.startswith("hind")]
            for k, g in x.groupby("mv"):
                if len(g) < 8:
                    continue
                r = {"form": form, "move": k, "n": len(g), "C→T" if form == "C" else "H3→T": cell(g[f"car_{form}_T"], g.review),
                     "C→A": cell(g.car_C_A, g.review), "A→T−1": cell(g.car_A_Tm1, g.review), "T−1→T": cell(g.car_Tm1_T, g.review),
                     "T→T+20": cell(g.car_T_Tp20, g.review)}
                mv.append(r)
        mv = pd.DataFrame(mv)
        mv.to_csv(OUT / "table_dax_by_move.csv", index=False)
        md.append("### DAX family: actual moves by type (matched-control CAR %, cluster t)\n\n" + mv.to_markdown(index=False) + "\n")
        # predicted promotions by target index (where the months-ahead return sits)
        nm_ = {0: "none", 1: "SDAX", 2: "MDAX", 3: "DAX"}
        tg = []
        for form in ["H6", "H3", "C"]:
            x = ce[(ce.form == form) & (ce.portfolio == "pred_up")].merge(
                pdx[pdx.horizon == form][["review", "col", "lvl_known", "lvl_pred"]], on=["review", "col"], how="left")
            x["target"] = np.where(x.lvl_pred > x.lvl_known, x.lvl_known.map(nm_) + "→" + x.lvl_pred.map(nm_), "TecDAX in")
            for (k, per), g in x.groupby(["target", "period"]):
                tg.append({"form": form, "target": k, "period": per, "n": len(g), "hit rate": round(g.hit.mean(), 2),
                           "median mcap €bn": round(g.mcap.median() / 1e9, 2),
                           "horizon→T": cell(g[f"car_{form}_T"], g.review),
                           "index-adjusted": cell(g[f"cari_{form}_T"], g.review)})
        tg = pd.DataFrame(tg)
        tg.to_csv(OUT / "table_dax_promotions_by_target.csv", index=False)
        md.append("### DAX family: predicted promotions by target index (matched-control CAR %, cluster t)\n\n" +
                  tg.to_markdown(index=False) + "\n")

open(OUT / "report_tables.md", "w").write("\n".join(md))
print("report_tables.md:", len(md), "tables")
