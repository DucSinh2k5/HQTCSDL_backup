import duckdb

DB_PATH = "stock_analytics.duckdb"
CSV_PATH = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\company_info.csv"
clean_csv_path = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\clean\Data_500_stocks_cleaned.csv"
con = duckdb.connect(DB_PATH)

con.execute(f"""
CREATE OR REPLACE TABLE stock_symbols AS
SELECT
    CAST(symbol AS VARCHAR) AS symbol,
    CAST(company_name AS VARCHAR) AS company_name,
    CAST(listed_date AS TIMESTAMP) AS listed_at,
    CAST(listed_date AS DATE) AS listed_date
FROM read_csv_auto('{CSV_PATH}', header = true);
""")
#xoa bang cu neu da ton tai
con.execute(f"""
CREATE OR REPLACE TABLE stock_prices AS
SELECT
    CAST(symbol AS VARCHAR) AS symbol,
    CAST(date AS DATE) AS date, 
    CAST(open AS DOUBLE) AS open,
    CAST(high AS DOUBLE) AS high,
    CAST(low AS DOUBLE) AS low,
    CAST(close AS DOUBLE) AS close,
    CAST(volume AS BIGINT) AS volume
FROM read_csv_auto('{clean_csv_path}', header = true);
""")
df = con.execute("""
SELECT *
FROM stock_prices;
""").fetchdf()

print(df)

con.close()