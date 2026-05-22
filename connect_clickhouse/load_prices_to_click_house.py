from pathlib import Path
import os

import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PRICE_CSV = BASE_DIR / "data" / "clean" / "Data_500_stocks_clean_ver2.csv"

client = clickhouse_connect.get_client(
    host=os.getenv("CLICKHOUSE_HOST"),
    port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
    username=os.getenv("CLICKHOUSE_USER"),
    password=os.getenv("CLICKHOUSE_PASSWORD"),
    database=os.getenv("CLICKHOUSE_DATABASE", "default"),
    secure=os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true",
)

client.command("""
CREATE DATABASE IF NOT EXISTS stock
""")

client.command("""
CREATE TABLE IF NOT EXISTS stock.stock_prices
(
    symbol String,
    date DateTime,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64
)
ENGINE = MergeTree
ORDER BY (symbol, date)
""")

df = pd.read_csv(CLEAN_PRICE_CSV)

df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["symbol", "date"])

client.insert_df(
    table="stock.stock_prices",
    df=df
)

print(f"Uploaded {len(df)} rows to stock.stock_prices")