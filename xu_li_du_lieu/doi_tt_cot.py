
import pandas as pd

path = r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\clean\Data_500_stocks_cleaned.csv"
cols = ["symbol", "date", "open", "high", "low", "close", "volume"]

df = pd.read_csv(path, usecols=cols)
df = df[cols]
df.to_csv(path, index=False, chunksize=2000)
print(df.head())