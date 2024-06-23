import pandas as pd

data = pd.read_csv('pricestest3.csv', index_col=0, parse_dates=True)
sp500 = pd.DataFrame(data)

djia = pd.read_csv('pricestest2.csv', index_col=0, parse_dates=True)
DJIA = pd.DataFrame(djia)

sp500.sort_index(inplace=True)
djia.sort_index(inplace=True)

merged_data = sp500.join(djia[['Close2']], how='left')

merged_data.to_csv("SP500_DJIA_merged.csv")
