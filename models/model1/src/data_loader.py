import clickhouse_connect
import pandas as pd

from src.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_TABLE,
    CLICKHOUSE_USER,
)


def load_data(path=None):
    if path is not None:
        print("[model1] Ignoring CSV path; loading data from ClickHouse.")

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )

    query = f"""
        SELECT *
        FROM {CLICKHOUSE_TABLE}
        ORDER BY symbol, trading_date
    """
    df = client.query_df(query)
    df.columns = df.columns.str.strip()
    return df
