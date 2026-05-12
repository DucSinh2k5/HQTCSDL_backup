import os
from dotenv import load_dotenv
import clickhouse_connect

load_dotenv()


def get_clickhouse_client():
    host = os.getenv("CLICKHOUSE_HOST")
    port = int(os.getenv("CLICKHOUSE_PORT", "8443"))
    username = os.getenv("CLICKHOUSE_USER")
    password = os.getenv("CLICKHOUSE_PASSWORD")
    database = os.getenv("CLICKHOUSE_DATABASE", "default")
    secure = os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true"

    if not host or not username or not password:
        raise ValueError("Missing ClickHouse connection variables in .env")

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        secure=secure,
    )

    return client