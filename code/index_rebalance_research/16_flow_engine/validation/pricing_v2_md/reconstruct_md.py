"""Recover the market data of an engine sheet whose md was not saved, from the sheet itself (paper research).

The sheet stores adv_sh, close_base_sh, eur_per_ccy, cad_per_ccy, price at 6 significant digits (%.6g), and also
columns derived from them (pct_adv, pct_adv_visible, close_mult, exp_close_print_mult, notional_*). Each stored string
bounds the true value to an interval; intersecting the intervals of a quantity and of the columns derived from it gives
a value consistent with every stored cell. Demand (net_sh, ...) comes from re-running the engine on the same inputs
(full precision, harness .pkl). Output: an md CSV for engine.py --md-from (full precision).

python3 -I reconstruct_md.py <backup_sheet.csv> <engine_run.pkl> <out_md.csv>
"""
import math
import sys
import numpy as np
import pandas as pd

sheet_p, pkl_p, out_p = sys.argv[1:4]
S = pd.read_csv(sheet_p, dtype=str, keep_default_na=False)
R = pd.read_pickle(pkl_p).set_index("key")


def iv(sv):
    """interval of values that print as sv under %.6g"""
    if sv in ("", "nan", "inf", "-inf"):
        return None
    v = float(sv)
    if v == 0:
        return (0.0, 0.0)
    e = math.floor(math.log10(abs(v)))
    half = 0.5 * 10 ** (e - 5)
    return (v - half, v + half)


def inv(num, sv):
    """interval for x given that num / x prints as sv (num != 0)"""
    i = iv(sv)
    if i is None or num == 0 or not np.isfinite(num):
        return None
    lo, hi = i
    a, b = (num / hi if hi != 0 else np.inf), (num / lo if lo != 0 else np.inf)
    return (min(a, b), max(a, b))


def mul(fac, sv):
    """interval for x given that fac * x prints as sv"""
    i = iv(sv)
    if i is None or fac == 0 or not np.isfinite(fac):
        return None
    a, b = i[0] / fac, i[1] / fac
    return (min(a, b), max(a, b))


def pick(name, base, *cons):
    if base is None:
        return np.nan, ""
    lo, hi = base
    for c in cons:
        if c is not None:
            lo, hi = max(lo, c[0]), min(hi, c[1])
    stored = float(np.mean(base))
    if lo > hi:
        return stored, f"{name}: empty intersection, kept stored value"
    if lo <= stored <= hi:          # the stored value is consistent with every derived cell: keep it
        return stored, ""
    return (lo + hi) / 2, ""


out, warn = [], []
for r in S.itertuples(index=False):
    d = r._asdict()
    k = d["key"]
    g = R.loc[k]
    net, vis, phi = abs(g.net_sh), abs(g.net_sh_visible), g.phi
    adv, w1 = pick("adv", iv(d["adv_sh"]), inv(net, d["pct_adv"]), inv(vis, d["pct_adv_visible"]))
    cm = iv(d["exp_close_print_mult"])
    cmi = None if cm is None else (phi * net / (cm[1] - 1), phi * net / (cm[0] - 1)) if cm[0] > 1 else None
    cb, w2 = pick("close_base", iv(d["close_base_sh"]), inv(net, d["close_mult"]), cmi)
    nl, gnl, enl = g.notional_local, g.gross_notional_local, g.exp_net_sh * g.price
    eur, w3 = pick("eur", iv(d["eur_per_ccy"]), mul(nl, d["notional_eur"]), mul(gnl, d["gross_notional_eur"]),
                   mul(enl, d["exp_notional_eur"]))
    cad, w4 = pick("cad", iv(d["cad_per_ccy"]), mul(nl, d["notional_cad"]))
    warn += [f"{k} {w}" for w in (w1, w2, w3, w4) if w]
    out.append(dict(key=k, country=d["country"], ccy=d["ccy"], price=float(d["price"]) if d["price"] else np.nan,
                    adv_sh=adv, adv_basis=d["adv_basis"], close_base_sh=cb, close_basis=d["close_basis"],
                    close_n_days=float(d["close_n_days"]) if d["close_n_days"] else np.nan, eur_per_ccy=eur,
                    cad_per_ccy=cad))
pd.DataFrame(out).to_csv(out_p, index=False, float_format="%.17g")
print(out_p, len(out), "rows;", len(warn), "warnings")
for w in warn[:20]:
    print("  ", w)
