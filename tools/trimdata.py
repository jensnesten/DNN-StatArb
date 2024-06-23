import pandas as pd

df = pd.read_csv('./SP500_DJIA_merged.csv', index_col='Date', parse_dates=True)


df = df[~df.index.duplicated(keep='first')]

cleaned_file_path = 'clean.csv'
df.to_csv(cleaned_file_path, index=True, header=True)