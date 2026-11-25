#!/usr/bin/env python3
"""Archive Interactive Brokers' public shortable-stock files (borrow fee, rebate, shares available).

Source (tested 2026-10-09): ftp://ftp2.interactivebrokers.com, user "shortstock", empty password.
IBKR KB article "Monitoring Stock Loan Availability" (https://ibkb.interactivebrokers.com/article/2024)
documents the same host and login. ftp3.interactivebrokers.com and ftp.interactivebrokers.com
timed out on 2026-10-09.

Each country file is pipe-delimited with CRLF line endings:
    #BOF|2026.10.09|22:31:39
    #SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|FIGI|
    AAG|CAD|AFTERMATH SILVER LTD|267518964|CA00831V2057|-0.5161|2.7661|4900000|BBG001M2GWD6|
    ...
    #EOF|3308
FEERATE and REBATERATE are annual percentages; AVAILABLE is IBKR's indicative inventory in shares.
A file is grouped by IBKR trading venue country, not issuer country (germany.txt carries many
foreign ISINs), so join on ISIN or FIGI. The <file>.md5 sidecar is the MD5 of the file with LF
line endings (i.e. after stripping CR).

Per run, the script:
  * lists the FTP root and downloads every country file (*.txt), or only --countries;
  * validates each (#BOF/#EOF present, #EOF row count == data rows, MD5 == sidecar), retrying
    because IBKR rewrites the files during the day;
  * writes it gzipped to  <out>/<BOF date>/<country>_<BOFdate>_<BOFtime>.txt.gz
    (the BOF stamp is IBKR's own as-of time; a snapshot already on disk is skipped);
  * appends one row per file to <out>/manifest.csv.
--with-margin also saves the stockmargin_final_dtls.*.dat margin files (about 10 MB each,
eight files; they carry ShortMargin and concentration margins). Off by default.

Run it once a day, or several times a day: each new BOF stamp becomes a new snapshot.
This script installs no scheduler. Example cron line, for reference only:
    15 9,12,15,22 * * 1-5  /usr/bin/python3 /path/to/ibkr_shortstock_archiver.py
Exit code: 0 if every requested file was saved or already present, 1 otherwise.

Standard library only. Python 3.9+.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import ftplib
import gzip
import hashlib
import io
import os
import sys
import time
from pathlib import Path

HOST = "ftp2.interactivebrokers.com"
USER = "shortstock"
PASSWORD = ""
DEFAULT_OUT = Path(__file__).resolve().parent / "ibkr_archive"
MANIFEST_FIELDS = [
    "run_utc", "file", "bof_date", "bof_time", "rows", "bytes", "md5_ok", "status", "path",
]
NET_ERRORS = (ftplib.Error, OSError, EOFError)


class Conn:
    """FTP session that reconnects on demand."""

    def __init__(self, timeout: int):
        self.timeout = timeout
        self.ftp: ftplib.FTP | None = None

    def get(self) -> ftplib.FTP:
        if self.ftp is None:
            ftp = ftplib.FTP(HOST, timeout=self.timeout)
            ftp.login(USER, PASSWORD)
            self.ftp = ftp
        return self.ftp

    def reset(self) -> None:
        if self.ftp is not None:
            try:
                self.ftp.close()
            except NET_ERRORS:
                pass
        self.ftp = None

    def fetch(self, name: str) -> bytes:
        buf = io.BytesIO()
        self.get().retrbinary(f"RETR {name}", buf.write)
        return buf.getvalue()

    def quit(self) -> None:
        if self.ftp is not None:
            try:
                self.ftp.quit()
            except NET_ERRORS:
                pass


def parse_header(raw: bytes, name: str) -> tuple[str, str, int]:
    """Return (bof_date YYYY-MM-DD, bof_time HHMMSS, data_rows); raise ValueError if malformed."""
    lines = [ln for ln in raw.replace(b"\r", b"").decode("utf-8", "replace").split("\n") if ln]
    if not lines or not lines[0].startswith("#BOF|"):
        raise ValueError(f"{name}: no #BOF line")
    if not lines[-1].startswith("#EOF|"):
        raise ValueError(f"{name}: no #EOF line (truncated?)")
    _, d, t = lines[0].split("|")[:3]
    bof_date = dt.datetime.strptime(d.strip(), "%Y.%m.%d").date().isoformat()
    bof_time = t.strip().replace(":", "")
    data_rows = sum(1 for ln in lines if not ln.startswith("#"))
    stated = int(lines[-1].split("|")[1])
    if name.endswith(".txt") and stated != data_rows:
        raise ValueError(f"{name}: #EOF says {stated} rows, found {data_rows}")
    return bof_date, bof_time, data_rows


def append_manifest(out: Path, row: dict) -> None:
    path = out / "manifest.csv"
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def archive_one(conn: Conn, name: str, listing: set[str], out: Path,
                retries: int, run_utc: str) -> bool:
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            raw = conn.fetch(name)
            bof_date, bof_time, rows = parse_header(raw, name)
            want = None
            if f"{name}.md5" in listing:
                want = conn.fetch(f"{name}.md5").decode("ascii", "ignore").split()[0].lower()
            got = hashlib.md5(raw.replace(b"\r", b"")).hexdigest()
            if want is not None and want != got:
                raise ValueError(f"{name}: md5 mismatch (file being rewritten?)")
            stem, ext = name.rsplit(".", 1)
            dest = out / bof_date / f"{stem}_{bof_date.replace('-', '')}_{bof_time}.{ext}.gz"
            status = "exists"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                with gzip.open(tmp, "wb", compresslevel=6) as gz:
                    gz.write(raw)
                os.replace(tmp, dest)
                status = "saved"
            append_manifest(out, dict(run_utc=run_utc, file=name, bof_date=bof_date,
                                      bof_time=bof_time, rows=rows, bytes=len(raw),
                                      md5_ok="" if want is None else "True", status=status,
                                      path=str(dest.relative_to(out))))
            print(f"{status:7s} {name:40s} {bof_date} {bof_time} rows={rows}")
            return True
        except ValueError as exc:  # bad content: retry on the same session
            last_err = str(exc)
        except NET_ERRORS as exc:  # network trouble: reconnect, then retry
            last_err = f"{type(exc).__name__}: {exc}"
            conn.reset()
        print(f"retry   {name}: attempt {attempt}/{retries}: {last_err}", file=sys.stderr)
        if attempt < retries:
            time.sleep(min(30, 5 * attempt))
    append_manifest(out, dict(run_utc=run_utc, file=name, bof_date="", bof_time="", rows="",
                              bytes="", md5_ok="", status=f"failed: {last_err}"[:200], path=""))
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Archive IBKR public shortable-stock files.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="archive root (default: %(default)s)")
    ap.add_argument("--countries", nargs="+", default=None,
                    help="file stems to fetch, e.g. canada british germany (default: every *.txt)")
    ap.add_argument("--with-margin", action="store_true",
                    help="also archive stockmargin_final_dtls.*.dat (about 10 MB each)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=90, help="FTP socket timeout, seconds")
    args = ap.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    run_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = Conn(args.timeout)
    try:
        listing = set(conn.get().nlst())
    except NET_ERRORS as exc:
        print(f"cannot connect to {HOST}: {exc}", file=sys.stderr)
        append_manifest(out, dict(run_utc=run_utc, file="*", bof_date="", bof_time="", rows="",
                                  bytes="", md5_ok="", status=f"failed: connect: {exc}"[:200], path=""))
        return 1

    ok = True
    names = sorted(n for n in listing if n.endswith(".txt"))
    if args.countries:
        wanted = {c.lower().removesuffix(".txt") for c in args.countries}
        missing = sorted(wanted - {n[:-4] for n in names})
        if missing:
            print(f"not on server: {missing}", file=sys.stderr)
            ok = False
        names = [n for n in names if n[:-4] in wanted]
    if args.with_margin:
        names += sorted(n for n in listing if n.startswith("stockmargin_") and n.endswith(".dat"))

    for name in names:
        ok &= archive_one(conn, name, listing, out, args.retries, run_utc)
    conn.quit()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
