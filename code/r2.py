
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

# Repo root = parent of src/ (this file is src/data/r2.py). Absolute, so the
# answer does not depend on the caller's working directory.
# STANDALONE COPY (R2_export, 2026-09-29): the default env file is the
# r2.env beside this file, not <repo>/config/.env. Otherwise identical to
# src/data/r2.py at the time of copying.
REPO = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = REPO / "r2.env"
ENV_FILE_VAR = "QUANTT_ENV_FILE"

# Exactly these names; no aliases (see module docstring, third fallback).
REQUIRED_VARS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

# S3 bucket-name rule (3-63 chars of lowercase letters, digits, dot, hyphen).
# Checked because the bucket is interpolated into every SQL path string; a
# value outside this set is either a typo or something that should not be in
# a SQL literal.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$")
# WRDS schema/table names on the mirror, and option underlyings. Same reason:
# they go into a SQL string, so they are validated rather than escaped.
_IDENT_RE = re.compile(r"^[a-z0-9_]+$")
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class R2ConfigError(RuntimeError):
    """The R2 configuration is missing or malformed. Names what, never a value."""


@dataclass(frozen=True)
class R2Config:
    """The four values a connection needs. The repr never shows the key or secret."""

    endpoint_host: str
    bucket: str
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    source: Path = field(default=DEFAULT_ENV_FILE)

    def __repr__(self) -> str:  # explicit, so a future field cannot leak by default
        return (f"R2Config(endpoint_host={self.endpoint_host!r}, bucket={self.bucket!r}, "
                f"access_key_id=<redacted>, secret_access_key=<redacted>, "
                f"source={str(self.source)!r})")


def env_file_path(environ: "dict | os._Environ | None" = None) -> Path:
    """Tver looked at, and a missing file raises rather than moving
    on to another.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_FILE_VAR)
    path = Path(raw).expanduser() if raw else DEFAULT_ENV_FILE
    if not path.is_absolute():
        # A relative path would resolve against the CWD -- the exact
        # where-am-I-standing dependence this rebuild removed.
        raise R2ConfigError(f"{ENV_FILE_VAR} must be an absolute path; got a relative one")
    return path


def _read_env(path: Path) -> dict:
    """key=value pairs from `path`, NOT exported to os.environ."""
    if not path.exists():
        raise R2ConfigError(f"env file {path} does not exist (set {ENV_FILE_VAR} or create it)")
    from dotenv import dotenv_values  # imported here: keep module import cheap
    return {k: v for k, v in dotenv_values(path).items() if v}


def _endpoint_host(endpoint: str) -> str:
    """R2_ENDPOINT -> the bare host DuckDB's ENDPOINT wants.

    Accepts `https://<acct>.r2.cloudflarestorage.com` or the bare host. Refuses
    `http://` (credentials over plaintext) and any path component (DuckDB
    would build URLs under it and every read would 404 with a message that
    does not mention the endpoint). Error messages never echo the value.
    """
    s = endpoint.strip()
    if "://" in s:
        parts = urlsplit(s)
        if parts.scheme != "https":
            raise R2ConfigError("R2_ENDPOINT must use https:// (or be a bare host)")
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise R2ConfigError("R2_ENDPOINT must be a host only, with no path or query")
        host = parts.netloc
    else:
        host = s.rstrip("/")
        if "/" in host:
            raise R2ConfigError("R2_ENDPOINT must be a host only, with no path")
    if not host or "'" in host or " " in host:
        raise R2ConfigError("R2_ENDPOINT does not look like a host name")
    return host


def load_config(environ: "dict | os._Environ | None" = None) -> R2Config:
    """Build the config from the env file, or raise naming every missing variable."""
    path = env_file_path(environ)
    env = _read_env(path)
    missing = [n for n in REQUIRED_VARS if not env.get(n)]
    if missing:
        raise R2ConfigError(
            f"{', '.join(missing)} not set in {path}. All of {', '.join(REQUIRED_VARS)} "
            "are required; there are no defaults and no alias names.")
    bucket = env["R2_BUCKET"].strip()
    if not _BUCKET_RE.match(bucket):
        raise R2ConfigError("R2_BUCKET is not a valid S3 bucket name")
    return R2Config(endpoint_host=_endpoint_host(env["R2_ENDPOINT"]), bucket=bucket,
                    access_key_id=env["R2_ACCESS_KEY_ID"].strip(),
                    secret_access_key=env["R2_SECRET_ACCESS_KEY"].strip(), source=path)


def check_keys(environ: "dict | os._Environ | None" = None) -> dict[str, bool]:
    """Which REQUIRED_VARS are set in the env file -- booleans only, no network.

    The safe way to test a key: print whether it is set, never its value.
    """
    env = _read_env(env_file_path(environ))
    return {n: bool(env.get(n)) for n in REQUIRED_VARS}


def _sql_str(v: str) -> str:
    """A SQL string literal. DuckDB's CREATE SECRET takes no bound parameters."""
    return "'" + v.replace("'", "''") + "'"


def connect(config: R2Config | None = None):
    """An in-memory DuckDB connection that can read `s3://<bucket>/...`.

    httpfs is LOADed, not INSTALLed: INSTALL would download an extension from
    the network as a side effect of opening a connection. If it is not
    installed, this raises and says so; installing it is a deliberate step.
    """
    import duckdb
    cfg = config if config is not None else load_config()
    con = duckdb.connect()
    try:
        con.execute("LOAD httpfs")
    except duckdb.Error as exc:
        con.close()
        raise R2ConfigError(
            "DuckDB httpfs extension is not installed; run once, deliberately: "
            "python3 -c \"import duckdb; duckdb.connect().execute('INSTALL httpfs')\"") from exc
    # TEMPORARY: in memory only, never written to ~/.duckdb. SCOPE: applies to
    # our bucket and nothing else. URL_STYLE path + REGION auto: what R2's S3
    # API needs (Cloudflare R2 docs, S3 compatibility).
    con.execute(
        "CREATE TEMPORARY SECRET quantt_r2 ("
        " TYPE s3,"
        f" KEY_ID {_sql_str(cfg.access_key_id)},"
        f" SECRET {_sql_str(cfg.secret_access_key)},"
        f" ENDPOINT {_sql_str(cfg.endpoint_host)},"
        " REGION 'auto',"
        " URL_STYLE 'path',"
        f" SCOPE {_sql_str('s3://' + cfg.bucket)})")
    return con


def r2_path(schema: str, table: str, config: R2Config | None = None) -> str:
    """`s3://<bucket>/wrds/<schema>/<table>.parquet` for a mirrored WRDS table."""
    for name, v in (("schema", schema), ("table", table)):
        if not _IDENT_RE.match(v):
            raise ValueError(f"{name} {v!r} is not a lowercase WRDS identifier")
    cfg = config if config is not None else load_config()
    return f"s3://{cfg.bucket}/wrds/{schema}/{table}.parquet"


def options_path(symbol: str, month: str, config: R2Config | None = None) -> str:
    """`s3://<bucket>/options/<SYM>/tick/<YYYY-MM>.parquet` for one month of ticks."""
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(f"symbol {symbol!r} is not an upper-case ticker")
    if not _MONTH_RE.match(month):
        raise ValueError(f"month {month!r} is not YYYY-MM")
    cfg = config if config is not None else load_config()
    return f"s3://{cfg.bucket}/options/{symbol}/tick/{month}.parquet"


def q(con, sql: str, params: list | None = None):
    """Run SQL (with optional bound parameters) and return a pandas DataFrame."""
    return (con.execute(sql, params) if params is not None else con.execute(sql)).df()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2 (WRDS mirror) access; read-only.")
    ap.add_argument("--check-keys", action="store_true",
                    help="print only whether each R2_* variable is SET; no network")
    args = ap.parse_args(argv)
    if not args.check_keys:
        ap.print_help()
        return 2
    print(f"env file: {env_file_path()}")
    for name, is_set in check_keys().items():
        print(f"{name:<22} {'SET' if is_set else 'NOT SET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
