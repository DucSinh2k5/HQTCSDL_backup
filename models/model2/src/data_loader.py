import clickhouse_connect
import pandas as pd

from .config import (
    CLICKHOUSE_HOST,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_SECURE,
    SOURCE_TABLE
)


def get_clickhouse_client():
    return clickhouse_connect.get_client(
        # host=CLICKHOUSE_HOST,
        # port=CLICKHOUSE_PORT,
        # username=CLICKHOUSE_USER,
        # password=CLICKHOUSE_PASSWORD,
        # database=CLICKHOUSE_DATABASE,
        # secure=CLICKHOUSE_SECURE
        host='cvzq3t560s.ap-southeast-1.aws.clickhouse.cloud',
        port=8443,
        username='default',
        password='ze_1268BkMgWP',
        secure=True
    )


def load_stock_data():
    client = get_clickhouse_client()

    query = f"""
        SELECT *
        FROM {SOURCE_TABLE}
        ORDER BY symbol, trading_date
    """

    df = client.query_df(query)

    if df.empty:
        raise ValueError("Không lấy được dữ liệu từ ClickHouse.")

    df["trading_date"] = pd.to_datetime(df["trading_date"])

    return df

client = get_clickhouse_client()