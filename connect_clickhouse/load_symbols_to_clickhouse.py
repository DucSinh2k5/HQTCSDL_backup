from pathlib import Path
import os

import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = Path(r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks_1\data\company_info.csv")

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
CREATE TABLE IF NOT EXISTS stock.stock_symbols
(
    symbol String,
    company_name String,
    listed_at DateTime,
    listed_date Date
)
ENGINE = MergeTree
ORDER BY symbol
""")

df = pd.read_csv(CSV_PATH)

df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
df["company_name"] = df["company_name"].astype(str).str.strip()
df["listed_at"] = pd.to_datetime(df["listed_date"])
df["listed_date"] = pd.to_datetime(df["listed_date"]).dt.date

df = df[["symbol", "company_name", "listed_at", "listed_date"]]

client.insert_df(
    table="stock.stock_symbols",
    df=df
)

print(f"Uploaded {len(df)} rows to stock.stock_symbols")