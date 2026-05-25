"""
DuckDB database wrapper for the Multimodal Earnings Call Intelligence System.

Provides:
    init_database(db_path)          — create schema and return connection
    insert_segments(conn, df)       — insert/upsert into segments table
    insert_text_features(conn, df)  — insert/upsert into text_features table
    insert_audio_features(conn, df) — insert/upsert into audio_features table
    insert_market_data(conn, df)    — insert/upsert into market_data table
    query_call(conn, call_id)       — return all segments for a call_id
    validate_schema(conn)           — check all tables exist with correct columns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

# Path to the SQL init script, relative to this file
_SQL_INIT = Path(__file__).resolve().parent.parent.parent / "scripts" / "init_db.sql"


# ---------------------------------------------------------------------------
# Connection / Init
# ---------------------------------------------------------------------------


def init_database(db_path: Union[str, Path]) -> duckdb.DuckDBPyConnection:
    """
    Create (or open) the DuckDB database and apply the full schema.

    Args:
        db_path: Path to the .db file. Use ':memory:' for an in-memory DB.

    Returns:
        An open DuckDB connection with all four tables created.
    """
    db_path = Path(db_path) if db_path != ":memory:" else db_path
    conn = duckdb.connect(str(db_path))

    if not _SQL_INIT.exists():
        raise FileNotFoundError(
            f"SQL init script not found: {_SQL_INIT}\n"
            "Make sure scripts/init_db.sql exists in the project root."
        )

    sql = _SQL_INIT.read_text()
    conn.execute(sql)
    logger.info("Database initialized at: %s", db_path)
    return conn


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def _upsert_from_polars(
    conn: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
    table: str,
    primary_key: str,
) -> int:
    """
    Generic upsert: insert rows from a Polars DataFrame, ignoring conflicts
    on the primary key (INSERT OR IGNORE semantics via DELETE + INSERT).

    Returns the number of rows inserted.
    """
    if df.is_empty():
        logger.warning("Empty DataFrame passed for table '%s'. Nothing inserted.", table)
        return 0

    # Validate identifiers to prevent SQL injection (only allow [a-zA-Z0-9_])
    import re
    _IDENT_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    if not _IDENT_PATTERN.match(table):
        raise ValueError(f"Invalid table name: {table!r}")
    if not _IDENT_PATTERN.match(primary_key):
        raise ValueError(f"Invalid primary key name: {primary_key!r}")

    # Register the Polars DataFrame as a temporary view
    conn.register("_staging", df.to_arrow())
    try:
        # Delete existing rows that share a primary key, then insert fresh
        conn.execute(
            f"DELETE FROM {table} WHERE {primary_key} IN "
            f"(SELECT {primary_key} FROM _staging)"
        )
        conn.execute(f"INSERT INTO {table} SELECT * FROM _staging")
        n = df.height
        logger.info("Inserted %d rows into '%s'.", n, table)
        return n
    finally:
        conn.unregister("_staging")


def insert_segments(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """
    Insert or replace rows into the segments table (Contract A).

    Expected columns: call_id, segment_id, speaker_role, speaker_name,
                      segment_type, text, start_time, end_time, audio_path.
    """
    _validate_columns(df, "segments", [
        "call_id", "segment_id", "speaker_role", "segment_type", "text",
    ])
    return _upsert_from_polars(conn, df, "segments", "segment_id")


def insert_text_features(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """
    Insert or replace rows into the text_features table (Contract B).

    Expected columns: segment_id, sentiment_score, uncertainty_score,
                      forward_looking_score, hedging_frequency,
                      specificity_score, linguistic_complexity.
    """
    _validate_columns(df, "text_features", [
        "segment_id", "sentiment_score", "uncertainty_score",
    ])
    return _upsert_from_polars(conn, df, "text_features", "segment_id")


def insert_audio_features(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """
    Insert or replace rows into the audio_features table (Contract C).

    Expected columns: segment_id, pitch_mean, pitch_variance, speech_rate,
                      pause_duration_total, energy_variance, voice_stability,
                      opensmile_vector, wav2vec2_embedding.
    """
    _validate_columns(df, "audio_features", ["segment_id"])
    return _upsert_from_polars(conn, df, "audio_features", "segment_id")


def insert_market_data(conn: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """
    Insert or replace rows into the market_data table (Contract D).

    Expected columns: call_id, ticker, call_date, close_t0, close_t1,
                      close_t5, return_1d, return_5d, realized_vol_1d,
                      realized_vol_5d, earnings_surprise.
    """
    _validate_columns(df, "market_data", [
        "call_id", "ticker", "call_date",
    ])
    return _upsert_from_polars(conn, df, "market_data", "call_id")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def query_call(
    conn: duckdb.DuckDBPyConnection,
    call_id: str,
    include_features: bool = False,
) -> pl.DataFrame:
    """
    Return all segments for a given call_id as a Polars DataFrame.

    Args:
        conn: Open DuckDB connection.
        call_id: Earnings call identifier (e.g., 'AAPL_2024Q1').
        include_features: If True, LEFT JOIN text_features into the result.

    Returns:
        Polars DataFrame with segment rows for the requested call.
    """
    if include_features:
        sql = """
            SELECT s.*, tf.*
            FROM segments s
            LEFT JOIN text_features tf USING (segment_id)
            WHERE s.call_id = ?
            ORDER BY s.segment_id
        """
    else:
        sql = "SELECT * FROM segments WHERE call_id = ? ORDER BY segment_id"

    result = conn.execute(sql, [call_id]).pl()

    if result.is_empty():
        logger.warning("No segments found for call_id='%s'.", call_id)

    return result


def query_calls_by_ticker(
    conn: duckdb.DuckDBPyConnection,
    ticker: str,
) -> pl.DataFrame:
    """Return all market_data rows for a given ticker."""
    return conn.execute(
        "SELECT * FROM market_data WHERE ticker = ? ORDER BY call_date",
        [ticker],
    ).pl()


def list_calls(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Return a summary table of all calls: call_id, segment count, types."""
    return conn.execute("""
        SELECT
            call_id,
            COUNT(*) AS n_segments,
            COUNT(*) FILTER (WHERE segment_type = 'prepared_remarks')  AS n_prepared,
            COUNT(*) FILTER (WHERE segment_type = 'analyst_question')  AS n_questions,
            COUNT(*) FILTER (WHERE segment_type = 'management_answer') AS n_answers
        FROM segments
        GROUP BY call_id
        ORDER BY call_id
    """).pl()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_schema(conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Check that all four required tables exist.

    Returns True if all tables exist, raises RuntimeError otherwise.
    """
    required = {"segments", "text_features", "audio_features", "market_data"}
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
    }
    missing = required - existing
    if missing:
        raise RuntimeError(
            f"Database schema is incomplete. Missing tables: {missing}"
        )
    logger.info("Schema validation passed. All 4 tables present.")
    return True


def _validate_columns(df: pl.DataFrame, table_name: str, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing from a DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame for '{table_name}' is missing required columns: {missing}\n"
            f"Got columns: {df.columns}"
        )
