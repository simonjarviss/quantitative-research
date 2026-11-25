"""A GET-only view of the desk's Alpaca client, for the levered_etf_close fetch.

WHY A WRAPPER. `quantt.broker.alpaca.AlpacaClient` is the runner's client and holds
`submit_order`. A research fetch must not be able to reach it, even by mistake, so the
client is held under a name-mangled attribute and only `_get` is forwarded, and only to
the read-only paths listed below (the pattern of `scripts/research/reopen_c5/
fetch_2023_refs.QuotesOnly`). `assert_get_only` refuses the object before the first
request if any attribute name looks like an order or write path.

KEYS. Built only by `AlpacaClient.from_environment("cef")`, which needs QUANTT_ENV_FILE.
This module never names, opens or prints a key file; an unset variable raises first.
The cef book's key is used for market data only (the data plan is per account, and no
account endpoint is called).

PATHS ALLOWED (every one a GET that changes nothing):
  data host     /v2/stocks/bars, /v2/stocks/quotes, /v2/stocks/auctions,
                /v2/stocks/meta/exchanges
  trading host  /v2/calendar, /v2/assets/<symbol>   (reference data, no account state)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # ct_root, beside this file
from ct_root import credittrading_root  # noqa: E402
REPO = credittrading_root()   # CreditTrading, named on PYTHONPATH (ct_root.py says why)

ORDERISH = ("submit", "order", "cancel", "replace", "post", "delete", "patch", "put", "position", "account")
DATA_PATHS = {"/v2/stocks/bars", "/v2/stocks/quotes", "/v2/stocks/auctions", "/v2/stocks/meta/exchanges"}
TRADING_PATHS = {"/v2/calendar"}
ASSET_PATH = re.compile(r"^/v2/assets/[A-Z][A-Z0-9.\-]{0,9}$")
REQ_SLEEP_S = 0.31          # stay under 200 requests a minute (the Basic data plan's limit)


class FetchError(RuntimeError):
    """The fetch refused, naming what was wrong. Never a default."""


class DataOnly:
    """Forwards read-only GETs on the allowed paths; raises on anything else."""

    def __init__(self, client, data_base: str, trading_base: str):
        self.__client = client
        self._data_base = data_base
        self._trading_base = trading_base
        self.requests = 0
        self._last = 0.0
        self.min_interval = REQ_SLEEP_S

    def _get(self, base: str, path: str, params: dict | None = None):
        ok = ((base == self._data_base and path in DATA_PATHS)
              or (base == self._trading_base and (path in TRADING_PATHS or ASSET_PATH.match(path))))
        if not ok:
            raise FetchError(f"DataOnly refuses GET {base}{path}: not a listed read-only path")
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        self.requests += 1
        return self.__client._get(base, path, params)


def assert_get_only(obj) -> None:
    """Raise if `obj` exposes any attribute whose name looks like an order or account path."""
    bad = [n for n in dir(obj) if not n.startswith("__") and any(w in n.lower() for w in ORDERISH)]
    if bad:
        raise FetchError(f"{type(obj).__name__} exposes order-like attributes {bad}; refusing to fetch with it")


def make_client() -> DataOnly:
    """The only constructor. Refuses, before any work, when QUANTT_ENV_FILE is unset."""
    if not os.environ.get("QUANTT_ENV_FILE"):
        raise FetchError("QUANTT_ENV_FILE is not set: it must name the file holding the cef book's Alpaca keys")
    from quantt.broker.alpaca import DATA_BASE, PAPER_BASE, AlpacaClient
    c = DataOnly(AlpacaClient.from_environment("cef"), DATA_BASE, PAPER_BASE)
    assert_get_only(c)
    return c


def paged(c: DataOnly, path: str, params: dict, key: str, max_pages: int = 2000) -> dict:
    """Every page of a multi-symbol market-data GET -> {symbol: [rows]}."""
    from quantt.broker.alpaca import DATA_BASE
    out, token, pages = {}, None, 0
    while True:
        p = dict(params)
        if token:
            p["page_token"] = token
        body = c._get(DATA_BASE, path, p)
        if not isinstance(body, dict) or key not in body:
            raise FetchError(f"GET {path}: response has no {key!r} key")
        for sym, rows in (body[key] or {}).items():
            if not isinstance(rows, list):
                raise FetchError(f"GET {path}: {key}[{sym}] is not a list")
            out.setdefault(sym, []).extend(rows)
        pages += 1
        nxt = body.get("next_page_token")
        if not nxt:
            return out
        if nxt == token or pages > max_pages:
            raise FetchError(f"GET {path}: paging did not terminate after {pages} pages")
        token = nxt
