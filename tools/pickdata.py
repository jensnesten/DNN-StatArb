import pandas as pd

# Step 2: Load the S&P 500 data
df = pd.read_csv('SP500.csv')

# Step 3: Ensure the date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Step 4: Filter the DataFrame
filtered_df = df[df['Date'] >= '2024-04-01']

# Step 5: Write to a new CSV file
filtered_df.to_csv('SP500_filtered_from_2024.csv', index=False)