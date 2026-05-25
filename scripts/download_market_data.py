#!/usr/bin/env python3
"""
Download market data from Yahoo Finance for earnings call dates.

Reads call_ids from segments.parquet (or a call manifest) and pulls
OHLCV data from yfinance to produce market_data.parquet (Contract D).

Usage:
    python scripts/download_market_data.py                              # Default
    python scripts/download_market_data.py --segments data/processed/segments.parquet
    python scripts/download_market_data.py --lookback-days 30 --forward-days 10
"""

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

# Add project root to path so 'src' can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import yfinance as yf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market data utilities
# ---------------------------------------------------------------------------


def get_next_business_day(date: str, prices: pl.DataFrame, offset: int = 1) -> str | None:
    """Get the closing price `offset` business days after `date`."""
    future = prices.filter(pl.col("Date") > pl.lit(date)).sort("Date")
    if len(future) >= offset:
        return future.row(offset - 1)
    return None


def compute_realized_volatility(prices: pl.DataFrame, date: str, window: int) -> float | None:
    """Compute realized volatility over `window` trading days after `date`."""
    future = prices.filter(pl.col("Date") > pl.lit(date)).sort("Date").head(window)
    if len(future) < 2:
        return None

    closes = future["Close"].to_list()
    log_returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    if not log_returns:
        return None
    return float(np.std(log_returns, ddof=1))


def fetch_ticker_prices(ticker: str, start_date: str, end_date: str) -> pl.DataFrame | None:
    """Download historical prices for a ticker from yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date, end=end_date, auto_adjust=True)
        if hist.empty:
            return None

        hist = hist.reset_index()
        df = pl.from_pandas(hist[["Date", "Open", "High", "Low", "Close", "Volume"]])

        # Ensure Date is a string for consistent handling
        df = df.with_columns(
            pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date")
        )
        return df
    except Exception as e:
        logger.warning("Failed to fetch prices for %s: %s", ticker, e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download market data for earnings calls and produce market_data.parquet",
    )
    parser.add_argument(
        "--segments", type=str, default="data/processed/segments.parquet",
        help="Path to segments.parquet to extract call_ids and dates from.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Output directory for market_data.parquet.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=30,
        help="Days before earliest call to start downloading prices. Default: 30.",
    )
    parser.add_argument(
        "--forward-days", type=int, default=15,
        help="Days after latest call to download prices. Default: 15.",
    )
    parser.add_argument(
        "--rate-limit-sleep", type=float, default=0.5,
        help="Seconds to sleep between yfinance requests. Default: 0.5.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    segments_path = project_root / args.segments
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load call metadata from segments.parquet
    # -----------------------------------------------------------------------
    if not segments_path.exists():
        logger.error("Segments file not found: %s", segments_path)
        logger.error("Run scripts/download_transcripts.py first.")
        sys.exit(1)

    segments = pl.read_parquet(segments_path)
    logger.info("Loaded %d segments from %s", len(segments), segments_path)

    # Extract unique calls with ticker and date info
    calls = (
        segments
        .select("call_id")
        .unique()
        .with_columns([
            pl.col("call_id").str.extract(r"^([A-Z]+)_", 1).alias("ticker"),
            pl.col("call_id").str.extract(r"_(\d{4})Q", 1).alias("year"),
            pl.col("call_id").str.extract(r"Q(\d)$", 1).alias("quarter"),
        ])
        .filter(pl.col("ticker").is_not_null())
        .sort("call_id")
    )
    logger.info("Found %d unique calls across %d tickers",
                len(calls), calls["ticker"].n_unique())

    # -----------------------------------------------------------------------
    # Step 2: Download prices per ticker
    # -----------------------------------------------------------------------
    unique_tickers = calls["ticker"].unique().sort().to_list()
    logger.info("Downloading prices for %d tickers...", len(unique_tickers))

    # We need a date mapping from call_id to actual call_date.
    # Since the HuggingFace dataset has a `date` field, we should use it.
    # But segments.parquet may not have it. So we approximate from year/quarter:
    # Q1 → ~Jan-Mar, Q2 → ~Apr-Jun, Q3 → ~Jul-Sep, Q4 → ~Oct-Dec
    # We'll download a wide window and match later.

    # Determine date range
    years = calls["year"].cast(pl.Int64).to_list()
    min_year = min(years) if years else 2023
    max_year = max(years) if years else 2024

    global_start = f"{min_year - 1}-10-01"  # Buffer before first call
    global_end = f"{max_year + 1}-04-01"    # Buffer after last call

    # Cache prices per ticker
    price_cache: dict[str, pl.DataFrame] = {}
    failed_tickers = []

    for ticker in tqdm(unique_tickers, desc="Downloading prices"):
        prices = fetch_ticker_prices(ticker, global_start, global_end)
        if prices is not None and len(prices) > 0:
            price_cache[ticker] = prices
        else:
            failed_tickers.append(ticker)
        time.sleep(args.rate_limit_sleep)

    logger.info("Downloaded prices for %d/%d tickers (%d failed)",
                len(price_cache), len(unique_tickers), len(failed_tickers))
    if failed_tickers:
        logger.warning("Failed tickers: %s", failed_tickers)

    # -----------------------------------------------------------------------
    # Step 3: Build market_data records
    # -----------------------------------------------------------------------
    # Quarter → approximate earnings call month (earnings typically reported
    # in the month after quarter-end)
    quarter_to_months = {
        "1": [4, 5],   # Q1 earnings reported in Apr-May
        "2": [7, 8],   # Q2 earnings reported in Jul-Aug
        "3": [10, 11], # Q3 earnings reported in Oct-Nov
        "4": [1, 2],   # Q4 earnings reported in Jan-Feb (next year)
    }

    market_records = []
    skipped = 0

    for row in tqdm(calls.iter_rows(named=True), total=len(calls), desc="Computing market data"):
        call_id = row["call_id"]
        ticker = row["ticker"]
        year = int(row["year"])
        quarter = row["quarter"]

        if ticker not in price_cache:
            skipped += 1
            continue

        prices = price_cache[ticker]

        # Find the most likely call date: look for earnings report dates
        # in the expected months
        months = quarter_to_months.get(quarter, [1, 2, 3])
        target_year = year + 1 if quarter == "4" else year

        # Find trading days in the expected month range
        candidate_dates = prices.filter(
            (pl.col("Date").str.slice(0, 4).cast(pl.Int64) == target_year)
            & (pl.col("Date").str.slice(5, 2).cast(pl.Int64).is_in(months))
        ).sort("Date")

        if len(candidate_dates) == 0:
            # Fallback: try the quarter-end month itself
            fallback_months = {
                "1": [3, 4], "2": [6, 7], "3": [9, 10], "4": [12, 1],
            }
            fb_months = fallback_months.get(quarter, [1])
            candidate_dates = prices.filter(
                (pl.col("Date").str.slice(5, 2).cast(pl.Int64).is_in(fb_months))
            ).sort("Date")

        if len(candidate_dates) == 0:
            skipped += 1
            continue

        # Use the last trading day of the first expected month as proxy for call_date
        # (most earnings calls happen in the last week of the reporting month)
        call_date = candidate_dates["Date"][-1]

        # Get close prices
        all_prices = prices.sort("Date")
        call_idx_df = all_prices.filter(pl.col("Date") <= pl.lit(call_date))
        if len(call_idx_df) == 0:
            skipped += 1
            continue

        close_t0 = float(call_idx_df["Close"][-1])

        # Next-day and 5-day closes
        future_prices = all_prices.filter(pl.col("Date") > pl.lit(call_date)).sort("Date")

        close_t1 = float(future_prices["Close"][0]) if len(future_prices) >= 1 else None
        close_t5 = float(future_prices["Close"][4]) if len(future_prices) >= 5 else None

        # Realized volatility
        realized_vol_1d = compute_realized_volatility(all_prices, call_date, window=2)
        realized_vol_5d = compute_realized_volatility(all_prices, call_date, window=6)

        # Returns
        return_1d = (close_t1 / close_t0 - 1) if close_t1 else None
        return_5d = (close_t5 / close_t0 - 1) if close_t5 else None

        market_records.append({
            "call_id": call_id,
            "ticker": ticker,
            "call_date": call_date,
            "close_t0": close_t0,
            "close_t1": close_t1,
            "close_t5": close_t5,
            "return_1d": return_1d,
            "return_5d": return_5d,
            "realized_vol_1d": realized_vol_1d,
            "realized_vol_5d": realized_vol_5d,
            "earnings_surprise": None,  # Requires EPS estimates — set to None for now
        })

    logger.info("Built market data for %d calls (%d skipped)", len(market_records), skipped)

    if not market_records:
        logger.error("No market data records produced. Check ticker coverage.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 4: Save as Parquet
    # -----------------------------------------------------------------------
    df = pl.DataFrame(market_records)

    output_path = output_dir / "market_data.parquet"
    df.write_parquet(output_path)
    logger.info("Saved market data to: %s", output_path)

    # -----------------------------------------------------------------------
    # Step 5: Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total records:     %d", len(df))
    logger.info("Unique tickers:    %d", df["ticker"].n_unique())
    logger.info("Date range:        %s to %s",
                df["call_date"].min(), df["call_date"].max())
    logger.info("Records with t+1:  %d", df.filter(pl.col("close_t1").is_not_null()).height)
    logger.info("Records with t+5:  %d", df.filter(pl.col("close_t5").is_not_null()).height)
    logger.info("Records with vol1: %d", df.filter(pl.col("realized_vol_1d").is_not_null()).height)

    # Also cache raw prices for future use
    price_cache_path = output_dir / "price_cache"
    price_cache_path.mkdir(parents=True, exist_ok=True)
    for ticker, prices in price_cache.items():
        prices.write_parquet(price_cache_path / f"{ticker}.parquet")
    logger.info("Cached raw prices for %d tickers in %s", len(price_cache), price_cache_path)


if __name__ == "__main__":
    main()
