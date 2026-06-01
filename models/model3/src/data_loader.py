from src.config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_FEATURES_TABLE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_USERNAME,
)


def quote_identifier(identifier):
    return "`" + str(identifier).replace("`", "``") + "`"


def load_data_from_clickhouse():
    if not CLICKHOUSE_HOST:
        raise ValueError("CLICKHOUSE_HOST is required")

    if not CLICKHOUSE_PASSWORD:
        raise ValueError("CLICKHOUSE_PASSWORD is required")

    try:
        import clickhouse_connect
    except ImportError as exc:
        raise ImportError(
            "clickhouse-connect is required for ClickHouse loading. "
            "Install it with: pip install clickhouse-connect"
        ) from exc

    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USERNAME,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )

    table_name = (
        f"{quote_identifier(CLICKHOUSE_DATABASE)}."
        f"{quote_identifier(CLICKHOUSE_FEATURES_TABLE)}"
    )
    query = f"SELECT * FROM {table_name} ORDER BY symbol, trading_date"
    df = client.query_df(query)
    df.columns = df.columns.str.strip()
    return df


def load_data(path=None):
    print(
        "[model3] Loading data from ClickHouse table "
        f"{CLICKHOUSE_DATABASE}.{CLICKHOUSE_FEATURES_TABLE}"
    )
    return load_data_from_clickhouse()
