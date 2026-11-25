"""Step 7: append the MSCI families to ../net_flow/pi_calibration.csv (the engine's pi file), keeping its columns
exactly: family, date, pi_visible, R, pi_true, source, R_se, R_n, R_window, R_basis.

Rules:
  * Back up the file first (data/backup/pi_calibration.csv.<UTC stamp>), then rewrite it as: every original line
    byte-for-byte (minus any earlier 'msci_pi study' rows, so the step is idempotent) + the new rows. The original
    rows are verified unchanged (SHA-256 of the non-MSCI lines before and after).
  * source = 'msci_pi study' on every new row (detail is in R_window / R_basis and in RESULTS.md).
  * Dates: every quarter-end with s12 holdings (2017-12-31 .. 2025-12-31), every MSCI review trade date T
    (2018-02 .. 2026-05; pi interpolated), and the extension dates to the last Datastream date (2026-07-01).
  * R: the family's market R from s06 (Standard index changes vs unchanged members, event-date FE), by window
    2018-2021 / 2022-2026 when both windows are identified (t >= 2), else the full sample. R is not extrapolated
    backward: the 2017-12-31 rows carry no R. Small Cap families take the market's Standard-event R (their own
    slope is not identified: WSML is sampled, so many 'Small Cap events' are sampling trades; it is reported in
    table_R_by_type.csv as a lower bound).
  * pi_true = R x pi_visible. R includes liquidity-provider round trips and benchmarked-active trading: pi_true is an
    upper bound on index-fund ownership.
"""
import hashlib
import time

import numpy as np
import pandas as pd

from common import DATA, OUT, PI_CAL, reviews
from s05_pi_series import pi_at

SRC = "msci_pi study"
COLS = ["family", "date", "pi_visible", "R", "pi_true", "source", "R_se", "R_n", "R_window", "R_basis"]
BASIS = {"CA": "CA consolidated volume (Datastream 295)", "EU": "primary-venue volume (understates by ~25%; R is a lower bound)"}
MAIN = "{m} MAIN: Standard adds/deletes/migrations vs unchanged members"


def r_table():
    T = pd.read_csv(OUT / "table_R.csv")
    out = {}
    for m in ("CA", "EU"):
        full = T[(T["sample"] == MAIN.format(m=m))].iloc[0]
        wins = {}
        for w in ("2018-2021", "2022-2026"):
            x = T[T["sample"] == MAIN.format(m=m) + f", {w}"]
            if len(x) and x.iloc[0].t >= 2:
                wins[w] = x.iloc[0]
        out[m] = (full, wins if len(wins) == 2 else {})
    return out


def r_for(rt, m, d, sc=False):
    full, wins = rt[m]
    d = pd.Timestamp(d)
    if d.year < 2018:
        return np.nan, np.nan, np.nan, ""
    lab = "Standard index changes vs unchanged members, event-date FE"
    if wins:
        w = "2018-2021" if d.year <= 2021 else "2022-2026"
        r = wins[w]
        txt = f"{m} {lab}, {w} (t {r.t:.1f}, {int(r.n_events)} index changes, {int(r.dates)} reviews)"
    else:
        r = full
        txt = f"{m} {lab}, 2018-2026 (t {r.t:.1f}, {int(r.n_events)} index changes, {int(r.dates)} reviews)"
    if sc:
        txt += "; Small Cap R not identified (WSML sampled): Standard-event R used"
    return float(r.R), float(r.se), int(r.n), txt


def rows():
    S = pd.read_parquet(DATA / "pi_series.parquet")
    rt = r_table()
    RV = reviews()
    out = []
    for fam, g in S.groupby("family"):
        m = "CA" if fam.startswith("MSCI_CANADA") else "EU"
        sc = fam.endswith("_SC")
        dates = set(g.dropna(subset=["pi_visible"]).date)
        dates |= set(RV[(RV["T"] >= g.date.min()) & (RV["T"] <= g.date.max())]["T"])
        for d in sorted(dates):
            pv = pi_at(S, fam, d)
            R, se, n, win = r_for(rt, m, d, sc)
            out.append(dict(family=fam, date=pd.Timestamp(d).date().isoformat(), pi_visible=pv, R=R,
                            pi_true=pv * R if np.isfinite(R) else np.nan, source=SRC, R_se=se,
                            R_n=n if np.isfinite(R) else np.nan, R_window=win, R_basis=BASIS[m]))
    return pd.DataFrame(out)[COLS]


def fmt(v):
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        return repr(v)
    return str(v)


def main():
    raw = PI_CAL.read_text().splitlines(keepends=True)
    header, body = raw[0], raw[1:]
    keep = [l for l in body if f",{SRC}," not in l]
    sha_before = hashlib.sha256("".join(keep).encode()).hexdigest()
    bk = DATA / "backup" / f"pi_calibration.csv.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    bk.parent.mkdir(exist_ok=True)
    bk.write_text("".join(raw))
    new = rows()
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in new.itertuples(index=False):
        w.writerow([fmt(v) if not isinstance(v, (int, np.integer)) else str(v) for v in r])
    text = header + "".join(keep) + buf.getvalue()
    PI_CAL.write_text(text)
    chk = [l for l in PI_CAL.read_text().splitlines(keepends=True)[1:] if f",{SRC}," not in l]
    sha_after = hashlib.sha256("".join(chk).encode()).hexdigest()
    assert sha_before == sha_after, "original rows changed"
    print(f"backup {bk.name}; kept {len(keep)} original rows (sha {sha_before[:12]} unchanged); appended {len(new)} rows")
    new.to_csv(OUT / "pi_rows_appended.csv", index=False)
    last = new.sort_values("date").groupby("family").tail(1)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(last[["family", "date", "pi_visible", "R", "pi_true", "R_se", "R_window"]].to_string(index=False))


if __name__ == "__main__":
    main()
