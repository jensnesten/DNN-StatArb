import pandas as pd
from datetime import datetime

# Load the CSV file
df = pd.read_csv('../Data/sp500_log_returns.csv')

# Convert the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set a reference date
reference_date = datetime(2000, 1, 1)

# Calculate the number of days since the reference date
df['Days'] = (df['Date'] - reference_date).dt.days

# Save the preprocessed data to a new CSV file
df.to_csv('../Data/sp500_log_returns_preprocessed.csv', index=False)
