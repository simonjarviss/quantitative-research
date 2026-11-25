"""Polite HTTP GET shared by the msci_pi study and the msci_nov2026 shadow.

Neutral browser User-Agent, no cookies, no identifiers, no accounts. A process-wide AND cross-process gap of
>= MIN_GAP seconds between requests (a lock file holds the time of the last request, so two scripts running at once
still respect the gap). Every caller caches responses on disk; this module only fetches.
"""
from __future__ import annotations

import fcntl
import os
import time
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MIN_GAP = 1.6
LOCK = Path(os.environ.get("POLITE_LOCK", str(Path(__file__).resolve().parent / "data" / ".polite.lock")))
LOG = Path(__file__).resolve().parent / "data" / "fetch_log.tsv"


def get(url: str, timeout: int = 60, retries: int = 2, accept: str = "*/*") -> tuple[int, bytes]:
    """Return (status, body). Never raises on HTTP errors (status 0 = network failure)."""
    status, body = 0, b""
    for k in range(retries + 1):
        with open(LOCK, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            try:
                last = float(fh.read().strip() or 0)
            except ValueError:
                last = 0.0
            wait = MIN_GAP - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    status, body = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, body = e.code, b""
            except Exception:  # noqa: BLE001
                status, body = 0, b""
            fh.seek(0); fh.truncate(); fh.write(f"{time.time():.3f}"); fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)
        try:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG, "a") as lg:
                lg.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{status}\t{len(body)}\t{url}\n")
        except OSError:
            pass
        if status == 200 or status in (403, 404, 410):
            break
        time.sleep(3 * (k + 1))
    return status, body
