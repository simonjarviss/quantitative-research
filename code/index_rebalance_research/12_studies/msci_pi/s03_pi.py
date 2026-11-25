"""Step 3: visible passive share pi_visible of MSCI Canada and MSCI Europe (by country), Standard and Small Cap,
per quarter-end 2017-12 .. 2025-12, from the holdings of every visible MSCI tracker in Thomson s12.

pi_visible(segment, market) = sum over tracker funds f of k_f, where k_f is the fraction of each member's index float
that fund f holds. For a cap-weighted tracker, AUM_f x (market weight in f's index) / (market's MSCI float MV) = k_f,
i.e. the task's definition. We measure it name by name:
    k_f = k_ruler x exp(median_i log(h_f,i / h_ruler,i))        over the segment's names i held by both
    k_ruler = 90th percentile of h_ruler,i / shares outstanding_i (s12 shrout2): names with FIF = 1
Rulers (s02 ref_table): CSCA (MSCI Canada), IMEA (MSCI Europe), WSML (MSCI World Small Cap; SCZ before 2019 for
Europe; no Canadian Small Cap ruler before 2019-03). Which funds count: funds.py (MSCI names + whitelist + the tracker
test). A fund that passes both the Standard and the Small Cap test tracks an IMI index.
"Holders" pi of a segment = all funds that hold the segment's names (Standard + IMI funds for Standard names; Small Cap +
IMI funds for Small Cap names). Those are the funds that trade when a name enters or leaves the segment from outside
the IMI. For a Standard <-> Small Cap migration the IMI funds do not trade: net = (pi_STDholders - pi_SCholders) x F.
Funds that report half-yearly are carried forward one quarter at their last k. Identical duplicate filings
(same quarter, same holdings) are counted once.
Europe: pi per DM-Europe country (CINS first character), then MSCI Europe = ruler-weight-weighted mean of countries.
Outputs: data/pi_funds.parquet (fund x quarter x segment x market/country contributions), data/pi_quarterly.parquet.
"""
import numpy as np
import pandas as pd

from common import DATA
from funds import COV_MIN, IQR_MAX, name_ok


def main():
    A = pd.read_parquet(DATA / "s12_agg.parquet")
    A["q"] = pd.to_datetime(A.q)
    A["rdate"] = pd.to_datetime(A.rdate)
    R = pd.read_parquet(DATA / "ref_names.parquet")
    R["q"] = pd.to_datetime(R.q)
    # no Canadian Small Cap ruler when only SCZ (EAFE Small Cap, no Canada) is available
    R = R[~((R.seg == "SC") & (R.mkt == "CA") & (R.ref == "SCZ"))]
    A = A.merge(R.groupby(["q", "seg", "mkt", "ctry"]).agg(W=("w", "sum"), nref=("c8", "size")).reset_index(),
                on=["q", "seg", "mkt", "ctry"])
    kref = R.groupby(["q", "seg", "mkt"]).agg(k_ref=("own", lambda s: s.quantile(0.9)), ruler=("ref", "first")).reset_index()
    A = A.merge(kref, on=["q", "seg", "mkt"])
    N = pd.read_parquet(DATA / "s12_fnames.parquet")
    N["rdate"] = pd.to_datetime(N.rdate)
    nm = N.sort_values("rdate").groupby("fundno").agg(fundname=("fundname", "last"), fcountry=("country", "last"))
    A = A.merge(nm, on="fundno", how="left")
    A["cov"] = A.wcov / A.W
    A["iqr"] = A.lr_p75 - A.lr_p25
    A["name_ok"] = [name_ok(f, n) for f, n in zip(A.fundno, A.fundname)]
    A["test_ok"] = (A["cov"] >= A.seg.map(COV_MIN)) & (A.iqr <= A.seg.map(IQR_MAX)) & (A.n >= np.minimum(3, A.nref))
    # A fund is accepted for a segment x market if it passes the tracker test in at least half of the quarters it
    # reports; it then counts in every quarter where it holds the segment (coverage >= 0.6), because a failed
    # quarter is usually a report-date mismatch with the ruler, not a change of mandate.
    pr = A[A.name_ok].groupby(["fundno", "seg", "mkt", "q"]).test_ok.max().groupby(["fundno", "seg", "mkt"]).mean()
    acc = set(pr[pr >= 0.5].index)
    A["accepted"] = [(f, s, m) in acc for f, s, m in zip(A.fundno, A.seg, A.mkt)]
    x = A[A.name_ok & A.accepted & (A["cov"] >= 0.6) & (A.iqr <= 2 * A.seg.map(IQR_MAX))].copy()
    x = x.sort_values("rdate").groupby(["fundno", "q", "seg", "mkt", "ctry"]).tail(1)     # one report per quarter
    # identical duplicate filings: same quarter/segment/country, same names held and same median ratio
    x["dupkey"] = x.q.astype(str) + x.seg + x.ctry + x.n.astype(str) + x.lr_med.round(5).astype(str)
    x = x.sort_values("fundno").drop_duplicates("dupkey")
    x["k"] = x.k_ref * np.exp(x.lr_med)
    # carry forward one quarter for funds that skip a quarter (half-yearly reporters)
    qs = sorted(A.q.unique())
    rows = [x.assign(carried=False)]
    have = set(zip(x.fundno, x.q, x.seg, x.ctry))
    for (f, s, c), g in x.groupby(["fundno", "seg", "ctry"]):
        for q in g.q:
            i = qs.index(q)
            if i + 1 < len(qs) and (f, qs[i + 1], s, c) not in have:
                r = g[g.q == q].iloc[[0]].assign(q=qs[i + 1], carried=True)
                rows.append(r)
    F = pd.concat(rows, ignore_index=True)
    # fund type: IMI if it passes both tests in that market in that quarter
    both = F.groupby(["fundno", "q", "mkt"]).seg.nunique().rename("nseg").reset_index()
    F = F.merge(both, on=["fundno", "q", "mkt"])
    F["ftype"] = np.where(F.nseg == 2, "IMI", F.seg)
    F.to_parquet(DATA / "pi_funds.parquet", index=False)
    # quarterly pi per segment, market, country; with the split by fund type
    P = F.groupby(["q", "seg", "mkt", "ctry"]).agg(pi=("k", "sum"), n_funds=("fundno", "nunique"),
                                                   pi_imi=("k", lambda s: s[F.loc[s.index, "ftype"] == "IMI"].sum())).reset_index()
    W = R.groupby(["q", "seg", "mkt", "ctry"]).w.sum().rename("W").reset_index()
    P = P.merge(W, on=["q", "seg", "mkt", "ctry"], how="right").fillna({"pi": 0.0, "pi_imi": 0.0, "n_funds": 0})
    P.to_parquet(DATA / "pi_quarterly.parquet", index=False)
    M = P.groupby(["q", "seg", "mkt"]).apply(lambda g: pd.Series(dict(pi=np.average(g.pi, weights=g.W),
                                                                      pi_imi=np.average(g.pi_imi, weights=g.W)))).reset_index()
    M.to_parquet(DATA / "pi_quarterly_mkt.parquet", index=False)
    pd.set_option("display.width", 250)
    print((M.pivot_table(index="q", columns=["seg", "mkt"], values="pi") * 100).round(3).to_string())
    print((M.pivot_table(index="q", columns=["seg", "mkt"], values="pi_imi") * 100).round(3).tail(4).to_string())


if __name__ == "__main__":
    main()
