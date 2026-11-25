"""Tests for the r2 package. `pytest r2` is offline; `R2_LIVE=1 pytest r2` also hits the bucket."""
import os
import re
from pathlib import Path

import pytest

import r2

FAKE = r2.R2Config(endpoint_host="acct123.r2.cloudflarestorage.com", bucket="test-bucket",
                   access_key_id="AKID-FAKE-VALUE", secret_access_key="SECRET-FAKE-VALUE")
HAVE_CATALOG = (r2.CATALOG_DIR / "tables.csv").exists()
live = pytest.mark.skipif(os.environ.get("R2_LIVE") != "1", reason="set R2_LIVE=1 to query the bucket")


def write_env(tmp_path: Path, **values) -> dict:
    path = tmp_path / "r2.env"
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return {r2.ENV_FILE_VAR: str(path)}


# --- configuration: one file, no defaults, nothing echoed ---------------------------------

def test_missing_env_file_names_the_path(tmp_path):
    with pytest.raises(r2.R2ConfigError, match="does not exist"):
        r2.load_config({r2.ENV_FILE_VAR: str(tmp_path / "absent.env")})


def test_relative_env_path_is_refused():
    with pytest.raises(r2.R2ConfigError, match="absolute"):
        r2.env_file_path({r2.ENV_FILE_VAR: "config/.env"})


def test_missing_variables_are_named_and_values_are_not(tmp_path):
    environ = write_env(tmp_path, R2_ACCESS_KEY_ID="AKID-FAKE-VALUE", R2_SECRET_ACCESS_KEY="SECRET-FAKE-VALUE")
    with pytest.raises(r2.R2ConfigError) as err:
        r2.load_config(environ)
    assert "R2_ENDPOINT" in str(err.value) and "R2_BUCKET" in str(err.value)
    assert "FAKE-VALUE" not in str(err.value)


@pytest.mark.parametrize("endpoint", ["http://acct.r2.cloudflarestorage.com",
                                      "https://acct.r2.cloudflarestorage.com/bucket"])
def test_bad_endpoint_is_refused(tmp_path, endpoint):
    environ = write_env(tmp_path, R2_ENDPOINT=endpoint, R2_ACCESS_KEY_ID="a", R2_SECRET_ACCESS_KEY="b",
                        R2_BUCKET="test-bucket")
    with pytest.raises(r2.R2ConfigError):
        r2.load_config(environ)


def test_repr_and_check_keys_never_show_secrets(tmp_path):
    assert "FAKE-VALUE" not in repr(FAKE)
    environ = write_env(tmp_path, R2_ENDPOINT="https://acct.r2.cloudflarestorage.com",
                        R2_ACCESS_KEY_ID="AKID-FAKE-VALUE", R2_SECRET_ACCESS_KEY="SECRET-FAKE-VALUE",
                        R2_BUCKET="test-bucket")
    assert r2.check_keys(environ) == dict.fromkeys(r2.REQUIRED_VARS, True)
    assert r2.load_config(environ).endpoint_host == "acct.r2.cloudflarestorage.com"


# --- names and paths ---------------------------------------------------------------------

def test_paths():
    assert r2.r2_path("crsp_a_stock", "dsf_v2", FAKE) == "s3://test-bucket/wrds/crsp_a_stock/dsf_v2.parquet"
    assert r2.options_path("SPY", "2026-06", FAKE) == "s3://test-bucket/options/SPY/tick/2026-06.parquet"


@pytest.mark.parametrize("call", [
    lambda: r2.r2_path("crsp_a_stock", "dsf'; --", FAKE),
    lambda: r2.r2_path("CRSP", "dsf", FAKE),
    lambda: r2.options_path("spy", "2026-06", FAKE),
    lambda: r2.options_path("SPY", "2026-13", FAKE),
    lambda: r2.source("desk.unknown", FAKE),
])
def test_bad_names_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_source_expressions():
    assert r2.source("crsp_a_stock.dsf_v2", FAKE) == \
        "read_parquet('s3://test-bucket/wrds/crsp_a_stock/dsf_v2.parquet')"
    assert r2.source("options.spy", FAKE) == "read_parquet('s3://test-bucket/options/SPY/tick/*.parquet')"
    desk = r2.source("desk.futures", FAKE)
    assert "s3://test-bucket/Data_Jayden/futures/*.csv" in desk
    assert "'dv01_usd_per_bp': 'DOUBLE'" in desk  # empty column must not come back as VARCHAR


@pytest.mark.skipif(not HAVE_CATALOG, reason="catalog/ not built")
def test_referenced_tables():
    query = ("SELECT * FROM CRSP_A_STOCK.DSF_V2 d JOIN crsp_a_ccm.ccmxpf_lnkhist l ON l.lpermno = d.permno "
             "JOIN options.spy o ON 1=1 JOIN desk.rates r ON 1=1 WHERE d.dlycaldt > 2025.12")
    assert r2.referenced_tables(query) == ["crsp_a_ccm.ccmxpf_lnkhist", "crsp_a_stock.dsf_v2",
                                           "desk.rates", "options.SPY"]
    with pytest.raises(KeyError, match="crsp_a_stock.dsf_v3"):
        r2.referenced_tables("SELECT * FROM crsp_a_stock.dsf_v3")


@pytest.mark.parametrize("query", ["CREATE TABLE t AS SELECT 1", "SELECT 1; DROP TABLE t",
                                   "COPY (SELECT 1) TO 'x.csv'", "SET threads = 1"])
def test_sql_refuses_non_read_statements(query):
    with pytest.raises(ValueError, match="read-only"):
        r2.sql(None, query, config=FAKE)  # refused before the connection is touched


def test_redact():
    msg = ("HTTP GET error reading 's3://quantt-data/wrds/x.parquet' via "
           "https://abc123.r2.cloudflarestorage.com/quantt-data/wrds/x.parquet; see https://duckdb.org/docs/")
    out = r2.redact(msg)
    assert "quantt-data" not in out and "abc123" not in out
    assert "s3://<bucket>/wrds/x.parquet" in out and "https://duckdb.org/docs/" in out


def test_module_never_writes_to_the_bucket():
    code = Path(r2.__file__).read_text()
    assert not re.search(r"\bCOPY\b[^\n]*\bTO\b", code)
    assert not re.search(r"\b(PERSISTENT SECRET|INSERT INTO|DELETE FROM)\b", code)


# --- live: R2_LIVE=1 -----------------------------------------------------------------------

@live
def test_ping():
    assert r2.ping().startswith("OK:")


@live
def test_sql_end_to_end():
    with r2.session() as con:
        ff = r2.sql(con, "SELECT count(*) AS n FROM ff_all.factors_daily WHERE date >= ?", ["2026-01-01"])
        assert ff["n"].iloc[0] > 0
        desk = r2.sql(con, "SELECT typeof(dv01_usd_per_bp) AS t FROM desk.futures LIMIT 1")
        assert desk["t"].iloc[0] == "DOUBLE"
        assert r2.sql(con, "SELECT current_setting('TimeZone') AS tz")["tz"].iloc[0] == "America/New_York"
        files = r2.options_paths(con, "SPY", "2026-06", "2026-07")
        assert [f.rsplit("/", 1)[1] for f in files] == ["2026-06.parquet", "2026-07.parquet"]


@live
def test_errors_are_redacted():
    cfg = r2.load_config()
    with r2.session(cfg) as con:
        with pytest.raises(Exception) as err:
            r2.q(con, f"SELECT 1 FROM read_parquet('{r2.r2_path('crsp_a_stock', 'no_such_table', cfg)}')")
    assert cfg.bucket not in str(err.value) and "<bucket>" in str(err.value)
