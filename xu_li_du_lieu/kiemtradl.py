import pandas as pd

CSV_PATH = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\clean\Data_500_stocks_cleaned.csv"
OUTPUT_PATH = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\clean\duplicate_symbol_date.csv"

df = pd.read_csv(CSV_PATH)
dup_mask = df.duplicated(subset=["symbol", "date"], keep=False)
dup_rows = df.loc[dup_mask].sort_values(["symbol", "date"])

print(f"Total rows: {len(df)}")
print(f"Duplicate rows (symbol+date): {dup_mask.sum()}")

if dup_rows.empty:
	print("No duplicates found for symbol+date.")
else:
	print(dup_rows.head(20))
	dup_rows.to_csv(OUTPUT_PATH, index=False)
	print(f"Saved duplicates to: {OUTPUT_PATH}")
