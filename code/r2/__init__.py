
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

# This package's folder (r2/). Absolute, so the answer does not depend on the
# caller's working directory. The default env file and catalog/ live here.
# Derived from the repo's src/data/r2.py, which reads <repo>/config/.env instead.
REPO = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = REPO / "r2.env"
ENV_FILE_VAR = "QUANTT_ENV_FILE"
# Written by `python3 -m r2.build_catalog`. Only the name lookups below need it; plain
# connect/r2_path/q work without it.
CATALOG_DIR = REPO / "catalog"

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

# Data_Jayden/<name>/<name>_<YYYY>.csv, exposed as desk.<name>. Types are
# pinned: futures.dv01_usd_per_bp is empty in every file, so type detection
# would make it VARCHAR, and a new file with a renamed column should fail
# loudly rather than read as NULLs.
DESK_TABLES = {
    "rates": {"observation_date": "DATE", "source": "VARCHAR", "series_id": "VARCHAR",
              "maturity": "VARCHAR", "rate_bps": "INTEGER"},
    "futures": {"observation_date": "DATE", "source": "VARCHAR", "instrument_id": "VARCHAR",
                "settlement_price": "DOUBLE", "dv01_usd_per_bp": "DOUBLE"},
    "contract_risk": {"observation_date": "DATE", "instrument_id": "VARCHAR", "dv01_usd_per_bp": "DOUBLE",
                      "rate_sensitivity_sign": "INTEGER", "dv01_method": "VARCHAR"},
}

# `schema.table` tokens in a query; sql() keeps the ones the catalog knows.
_REF_RE = re.compile(r"\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\b")
# DuckDB errors quote the full object URL; these hide the bucket and the
# account-specific endpoint host, and leave other links (DuckDB docs) alone.
_REDACT = ((re.compile(r"s3://[a-z0-9.\-]+/"), "s3://<bucket>/"),
           (re.compile(r"https?://[a-z0-9\-]+\.r2\.cloudflarestorage\.com/[a-z0-9.\-]+/"),
            "https://<endpoint>/<bucket>/"),
           (re.compile(r"[a-z0-9\-]+\.r2\.cloudflarestorage\.com"), "<endpoint>"))


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
    """The one env file credentials may come from.

    `$QUANTT_ENV_FILE` when set and non-empty (the laptop prod clone and the
    session runner set it, docs/RUNNER.md), else `r2.env` in this package's folder.
    This is the path the task contract names explicitly, not a search:
    exactly one candidate is ever looked at, and a missing file raises rather
    than moving on to another.
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


def redact(message: str) -> str:
    """`message` with the bucket name and endpoint host replaced by placeholders."""
    for pattern, placeholder in _REDACT:
        message = pattern.sub(placeholder, message)
    return message


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
    # Footers are re-read on every query otherwise: a repeat query on a big
    # table drops from seconds to milliseconds. The progress bar writes
    # carriage-return noise into logs and captured output.
    con.execute("SET enable_http_metadata_cache = true")
    con.execute("SET parquet_metadata_cache = true")
    con.execute("SET enable_progress_bar = false")
    # TIMESTAMPTZ columns (options `ts`, TRACE action times, CRSP intraday
    # index) are stored in UTC. Without this, `ts::DATE` follows the machine's
    # zone and gives different trading dates on a laptop and on a UTC server.
    con.execute("SET TimeZone = 'America/New_York'")
    return con


@contextmanager
def session(config: R2Config | None = None):
    """`with r2.session() as con:` -- connect() that always closes, secret included."""
    con = connect(config)
    try:
        yield con
    finally:
        con.close()


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


def options_paths(con, symbol: str, first_month: str, last_month: str,
                  config: R2Config | None = None) -> list[str]:
    """The month files of `symbol` that exist between two YYYY-MM months, inclusive.

    One LIST request. Months outside the archive are simply absent from the
    result; an empty result raises, since it always means a wrong symbol or range.
    """
    for m in (first_month, last_month):
        if not _MONTH_RE.match(m):
            raise ValueError(f"month {m!r} is not YYYY-MM")
    cfg = config if config is not None else load_config()
    pattern = options_path(symbol, "2000-01", cfg).rsplit("/", 1)[0] + "/*.parquet"
    files = [f for (f,) in con.execute("SELECT file FROM glob(?) ORDER BY file", [pattern]).fetchall()
             if first_month <= f.rsplit("/", 1)[1].removesuffix(".parquet") <= last_month]
    if not files:
        raise ValueError(f"no {symbol} option ticks between {first_month} and {last_month}")
    return files


def source(name: str, config: R2Config | None = None) -> str:
    """The SQL table expression behind a logical table name.

    `<dataset>.<table>`  -> the WRDS file, e.g. `crsp_a_stock.dsf_v2`
    `options.<SYM>`      -> every month of that symbol's ticks
    `desk.<name>`        -> the Data_Jayden CSVs (rates, futures, contract_risk), types pinned
    """
    schema, _, table = name.partition(".")
    cfg = config if config is not None else load_config()
    if schema == "options":
        pattern = options_path(table.upper(), "2000-01", cfg).rsplit("/", 1)[0] + "/*.parquet"
        return f"read_parquet({_sql_str(pattern)})"
    if schema == "desk":
        if table not in DESK_TABLES:
            raise ValueError(f"desk table {table!r} is not one of {', '.join(DESK_TABLES)}")
        types = ", ".join(f"{_sql_str(c)}: {_sql_str(t)}" for c, t in DESK_TABLES[table].items())
        pattern = f"s3://{cfg.bucket}/Data_Jayden/{table}/*.csv"
        return f"read_csv({_sql_str(pattern)}, header = true, union_by_name = true, types = {{{types}}})"
    return f"read_parquet({_sql_str(r2_path(schema, table, cfg))})"


def register(con, *names: str, config: R2Config | None = None) -> None:
    """Create a view per logical name, so SQL can say `FROM crsp_a_stock.dsf_v2`.

    Creating a view reads that file's footer (one round trip, ~1-3 s cold), so
    register what a query uses rather than the whole bucket.
    """
    cfg = config if config is not None else load_config()
    for name in names:
        expr = source(name, cfg)  # validates both parts before they reach SQL
        schema, _, table = name.partition(".")
        if schema == "options":
            table = table.upper()
        _execute(con, f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        _execute(con, f'CREATE OR REPLACE VIEW "{schema}"."{table}" AS SELECT * FROM {expr}')


@lru_cache(maxsize=1)
def _known_names() -> dict[str, str]:
    """lower-case name -> canonical name, for every table catalog/tables.csv lists."""
    path = CATALOG_DIR / "tables.csv"
    if not path.exists():
        raise R2ConfigError(f"{path} does not exist; run `python3 -m r2.build_catalog` to create it")
    names = {f"desk.{t}": f"desk.{t}" for t in DESK_TABLES}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["area"] == "wrds":
                names[f"{row['dataset']}.{row['table']}"] = f"{row['dataset']}.{row['table']}"
            elif row["area"] == "options":
                names[f"options.{row['table'].lower()}"] = f"options.{row['table']}"
    return names


def referenced_tables(query: str) -> list[str]:
    """Catalog tables named as `schema.table` in `query`, in canonical form.

    Raises KeyError for a name whose dataset exists but whose table does not
    (a typo, or a table newer than the catalog), instead of letting DuckDB
    report a missing schema.
    """
    known = _known_names()
    datasets = {n.split(".", 1)[0] for n in known}
    found, unknown = set(), set()
    for a, b in _REF_RE.findall(query):
        name = f"{a}.{b}".lower()
        if name in known:
            found.add(known[name])
        elif a.lower() in datasets:
            unknown.add(f"{a}.{b}")
    if unknown:
        raise KeyError(f"{', '.join(sorted(unknown))} not in the catalog. Check `python3 -m r2 --tables "
                       f"<dataset>`; for a table added since the catalog was last built, call "
                       f"r2.register(con, name) first.")
    return sorted(found)


def _execute(con, sql: str, params: list | None = None):
    import duckdb
    try:
        return con.execute(sql, params) if params is not None else con.execute(sql)
    except duckdb.Error as exc:
        # Same exception class, bucket and endpoint removed; `from None` so the
        # original message does not ride along in the traceback.
        raise type(exc)(redact(str(exc))) from None


def q(con, sql: str, params: list | None = None):
    """Run SQL (with optional bound parameters) and return a pandas DataFrame."""
    return _execute(con, sql, params).df()


def sql(con, query: str, params: list | None = None, config: R2Config | None = None):
    """Run a read-only query that names tables directly, and return a DataFrame.

        r2.sql(con, "SELECT dlycaldt, dlyret FROM crsp_a_stock.dsf_v2 WHERE permno = ?", [14593])

    Every catalog table the query names (`<dataset>.<table>`, `options.<SYM>`,
    `desk.<name>`) is registered as a view first. Only SELECT-type statements
    (including WITH, DESCRIBE, SUMMARIZE) and EXPLAIN are accepted: this is
    the one entry point that takes arbitrary SQL, and the bucket is read-only.
    """
    import duckdb
    for statement in duckdb.extract_statements(query):
        if statement.type not in (duckdb.StatementType.SELECT, duckdb.StatementType.EXPLAIN):
            raise ValueError(f"r2.sql runs read-only queries; got a {statement.type.name} statement")
    have = {f"{s}.{v}".lower() for s, v in con.execute(
        "SELECT schema_name, view_name FROM duckdb_views() WHERE NOT internal").fetchall()}
    need = [n for n in referenced_tables(query) if n.lower() not in have]
    if need:
        register(con, *need, config=config)
    return q(con, query, params)


def tables(pattern: str = ""):
    """Catalog rows (outside _recycle_bin) whose `dataset.table` or domain matches a regex."""
    import pandas as pd
    t = pd.read_csv(_catalog_file("tables.csv"))
    t = t[t["area"] != "_recycle_bin"]
    rx = re.compile(pattern, re.IGNORECASE)
    hay = t["dataset"] + "." + t["table"] + " " + t["domain"].fillna("")
    return t[hay.map(lambda h: bool(rx.search(h)))].reset_index(drop=True)


def columns(name: str):
    """The columns of one table (`crsp_a_stock.dsf_v2`, `options.SPY`, `desk.rates`)."""
    import pandas as pd
    dataset, _, table = name.partition(".")
    table = table.upper() if dataset == "options" else table
    c = pd.read_csv(_catalog_file("columns.csv"))
    out = c[(c["dataset"] == dataset) & (c["table"] == table)]
    if out.empty:
        raise KeyError(f"{name} is not in {CATALOG_DIR / 'columns.csv'}")
    return out.drop(columns=["area", "dataset", "table"]).reset_index(drop=True)


def find_column(pattern: str):
    """Every (dataset, table, column, type) whose column name matches a regex."""
    import pandas as pd
    rx = re.compile(pattern, re.IGNORECASE)
    c = pd.read_csv(_catalog_file("columns.csv"))
    return c[c["column"].map(lambda h: bool(rx.search(h)))][
        ["dataset", "table", "column", "type"]].reset_index(drop=True)


def _catalog_file(name: str) -> Path:
    path = CATALOG_DIR / name
    if not path.exists():
        raise R2ConfigError(f"{path} does not exist; run `python3 -m r2.build_catalog` to create it")
    return path


def ping(config: R2Config | None = None) -> str:
    """Prove the credentials work: one LIST and one footer read. Returns a summary line."""
    cfg = config if config is not None else load_config()
    with session(cfg) as con:
        prefix = r2_path("ff_all", "factors_daily", cfg).rsplit("/", 1)[0]
        listed = _execute(con, "SELECT count(*) FROM glob(?)", [prefix + "/*.parquet"]).fetchone()[0]
        rows = _execute(con, "SELECT num_rows FROM parquet_file_metadata(?)",
                        [r2_path("ff_all", "factors_daily", cfg)]).fetchone()[0]
    return f"OK: listed {listed} objects under wrds/ff_all/, read footer of ff_all.factors_daily ({rows:,} rows)"


def _diagnose(exc: BaseException) -> str:
    """One line saying which part of the access chain failed."""
    text = redact(str(exc))
    if isinstance(exc, R2ConfigError):
        return f"CONFIG: {text}"
    if "403" in text or "SignatureDoesNotMatch" in text or "InvalidAccessKeyId" in text:
        return f"CREDENTIALS REJECTED (check R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY): {text}"
    if "404" in text or "NoSuchBucket" in text:
        return f"NOT FOUND (check R2_BUCKET / R2_ENDPOINT): {text}"
    if "Could not establish connection" in text or "resolve" in text.lower() or "timed out" in text.lower():
        return f"NETWORK (endpoint unreachable): {text}"
    return f"{type(exc).__name__}: {text}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m r2", description="R2 (WRDS mirror) access; read-only.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-keys", action="store_true",
                   help="print only whether each R2_* variable is SET; no network")
    g.add_argument("--ping", action="store_true", help="prove the credentials work (one LIST, one footer read)")
    g.add_argument("--sql", metavar="QUERY", help="run a read-only query; tables named as dataset.table")
    g.add_argument("--tables", metavar="REGEX", nargs="?", const="", help="list catalog tables (no network)")
    g.add_argument("--columns", metavar="TABLE", help="columns of one table from the catalog (no network)")
    g.add_argument("--find-column", metavar="REGEX", help="tables having a matching column (no network)")
    ap.add_argument("--out", metavar="FILE", help="with --sql: write .csv or .parquet instead of printing")
    ap.add_argument("--max-rows", type=int, default=50, help="with --sql/--tables: rows to print (default 50)")
    args = ap.parse_args(argv)

    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)

    if args.check_keys:
        print(f"env file: {env_file_path()}")
        for name, is_set in check_keys().items():
            print(f"{name:<22} {'SET' if is_set else 'NOT SET'}")
        return 0
    if args.ping:
        try:
            print(ping())
            return 0
        except Exception as exc:  # the point of --ping is to name the failure
            print(_diagnose(exc), file=sys.stderr)
            return 1
    if args.sql is not None:
        with session() as con:
            df = sql(con, args.sql)
        if args.out:
            out = Path(args.out)
            df.to_parquet(out, index=False) if out.suffix == ".parquet" else df.to_csv(out, index=False)
            print(f"wrote {len(df):,} rows to {out}")
        else:
            print(df.head(args.max_rows).to_string(index=False))
            if len(df) > args.max_rows:
                print(f"... {len(df):,} rows in total (use --out FILE for all)")
        return 0
    if args.tables is not None:
        cols = ["dataset", "table", "domain", "role", "rows", "size_mb", "date_col", "date_min", "date_max", "grain"]
        df = tables(args.tables)[cols]
        print(df.head(args.max_rows).to_string(index=False))
        if len(df) > args.max_rows:
            print(f"... {len(df):,} tables match (raise --max-rows)")
        return 0
    if args.columns is not None:
        print(columns(args.columns).to_string(index=False))
        return 0
    print(find_column(args.find_column).to_string(index=False))
    return 0

