import clickhouse_connect
import pandas as pd

from config import (
    CLICKHOUSE_HOST,
    CLICKHOUSE_PORT,
    CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_TABLE,
    CLICKHOUSE_SECURE
)


class StockDataLoader:
  def __init__(self):
      self.client = clickhouse_connect.get_client(
          host=CLICKHOUSE_HOST,
          port=CLICKHOUSE_PORT,
          username=CLICKHOUSE_USER,
          password=CLICKHOUSE_PASSWORD,
          database=CLICKHOUSE_DATABASE,
          secure=CLICKHOUSE_SECURE
      )

  def load_data(self) -> pd.DataFrame:

      query = f"""
      SELECT *
      FROM {CLICKHOUSE_TABLE}
      ORDER BY symbol, trading_date
      """

      df = self.client.query_df(query)

      return df


if __name__ == "__main__":
  loader = StockDataLoader()
  df = loader.load_data()

  print("=" * 50)
  print("DATA SHAPE")
  print(df.shape)

  print("=" * 50)
  print("HEAD")
  print(df.head())

  print("=" * 50)
  print("COLUMNS")
  print(df.columns.tolist())