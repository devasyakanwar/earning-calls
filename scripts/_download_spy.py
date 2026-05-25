"""Download SPY price data for abnormal return computation."""
import yfinance as yf
import polars as pl
from pathlib import Path

spy = yf.Ticker("SPY")
hist = spy.history(start="2019-10-01", end="2026-04-01", auto_adjust=True)
hist = hist.reset_index()
df = pl.from_pandas(hist[["Date", "Open", "High", "Low", "Close", "Volume"]])
df = df.with_columns(pl.col("Date").dt.strftime("%Y-%m-%d").alias("Date"))

output = Path("data/processed/price_cache/SPY.parquet")
output.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(output)
print(f"SPY data: {len(df)} rows, {df['Date'].min()} to {df['Date'].max()}")
