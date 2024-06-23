import requests
import os
import json
import matplotlib.pyplot as plt
from lightweight_charts import Chart
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import tkinter as tk
from tkinter import ttk

def get_asset_type():
    user_input = combo.get()
    if user_input == "CfdOnIndex":
        assetType = "CfdOnIndex"
    elif user_input == "CfdOnFutures":
        assetType = "CfdOnFutures"
    elif user_input == "FxSpot":
        assetType = "FxSpot"
    elif user_input == "Stock":
        assetType = "Stock"
    elif user_input == "CfdOnStock":
        assetType = "CfdOnStock"
    elif user_input == "StockIndex":
        assetType = "StockIndex"
    return assetType

def get_period():
    return period_var.get()

def get_uic():
    return uic_var.get()

def get_date():
    return date_var.get()

def main():
    assetType = get_asset_type()
    period = get_period()
    uic = get_uic()
    date = get_date()
    headers = {
        "Authorization": "Bearer eyJhbGciOiJFUzI1NiIsIng1dCI6IjI3RTlCOTAzRUNGMjExMDlBREU1RTVCOUVDMDgxNkI2QjQ5REEwRkEifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiTTZNSnltdmxwcG1PYjF8VTI2QktDdz09IiwiY2lkIjoiTTZNSnltdmxwcG1PYjF8VTI2QktDdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiZTNkYjFlM2E0NTVlNDJmOWE5OTQ0N2M1ZGFiZGE0OGQiLCJkZ2kiOiI4NCIsImV4cCI6IjE3MjQ5NDI2MjQiLCJvYWwiOiIxRiIsImlpZCI6ImQ1M2ExM2MzZjhjODQzM2FhMmUwMDhkYzYwZjUwZDQzIn0.bmJmSaDkDHZ8IbxMHOqc2KRTw7rk83ZDlstcN3oa6wbDdfFiECpX0TrQvv5Y3IxuuGHvZBWhQg_sQ9VypJI6Mg"
    }
    response = get_data(headers, assetType)
    
    formatted_response = format_json(response)
    
    description = [data['Description'] for data in formatted_response['Data']]
    identifier = [data['Identifier'] for data in formatted_response['Data']]
    exchangeID = [data['ExchangeId'] for data in formatted_response['Data']]
    symbol = [data['Symbol'] for data in formatted_response['Data']]

    result_text.delete(1.0, tk.END)  # Clear previous results
    for i in range(len(description)):
        result_text.insert(tk.END, f"{description[i].ljust(20)} Symbol: {symbol[i].ljust(20)} Exchange ID: {exchangeID[i].ljust(20)} UIC: {identifier[i]}\n")

    # Get prices and save to CSV
    get_prices(headers, assetType=assetType, uic=uic, period=period, date=date)

def get_prices(headers, assetType, uic, period, date):
    base_url = "https://gateway.saxobank.com/sim/openapi"
    endpoint = f"/chart/v1/charts?AssetType={assetType}&Horizon={period}&Mode=From&Time={date}&Uic={uic}"
    response = requests.get(base_url + endpoint, headers=headers)
    formatted_response = format_json(response.json())

    # Check if 'Data' key exists in the response
    if 'Data' not in formatted_response:
        return

    data_list = formatted_response['Data']

    # Initialize lists to store the data
    close_ask_values = []
    time_values = []
    high_ask_values = []
    low_ask_values = []
    open_ask_values = []

    # Extract data with error handling
    for data in data_list:
        try:
            close_ask_values.append(data['CloseAsk'])
            time_values.append(datetime.strptime(data['Time'], '%Y-%m-%dT%H:%M:%S.%fZ'))
            high_ask_values.append(data['HighAsk'])
            low_ask_values.append(data['LowAsk'])
            open_ask_values.append(data['OpenAsk'])
        except KeyError as e:
            print(f"Warning: Missing key {e} in data entry {data}")

    # Check if we have any data to write to CSV
    if not time_values:
        print("Error: No valid data to write to CSV")
        return

    # Create a pandas dataframe with the data and save it to a CSV file
    df = pd.DataFrame({
        "Date": time_values,
        "Open": open_ask_values,
        "High": high_ask_values,
        "Low": low_ask_values,
        "Close": close_ask_values
    })
    df.to_csv("prices.csv", index=False)

def get_data(headers, assetType):
    endpoint = f"https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?AssetTypes={assetType}"
    response = requests.get(endpoint, headers=headers)
    return response.json()

def format_json(json_data):
    return json.loads(json.dumps(json_data, indent=2))

#------GUI------
root = tk.Tk()
root.title("Asset Type Selector")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

label = ttk.Label(frame, text="Select Asset Type:")
label.grid(row=0, column=0, sticky=tk.W)

combo = ttk.Combobox(frame, values=["CfdOnIndex", "CfdOnFutures", "FxSpot", "Stock", "CfdOnStock", "StockIndex"])
combo.grid(row=1, column=0, sticky=(tk.W, tk.E))
combo.current(0)

# Period Selector
period_var = tk.StringVar()
period_selector = ttk.Combobox(frame, textvariable=period_var, width=4)
period_selector['values'] = (1, 5, 15, 30, 60, 1440)
period_selector.grid(row=1, column=1, sticky=tk.W)  # Adjust the row and column as needed
period_selector.current(0)
label1 = ttk.Label(frame, text="Period:")
label1.grid(row=0, column=1, sticky=tk.W)

# Text Result when getdata() is called
result_text = tk.Text(frame, wrap=tk.WORD, width=100, height=40)
result_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# UIC selector
label_uic = ttk.Label(frame, text="UIC:")
label_uic.grid(row=0, column=2, sticky=tk.W)  # Place the label in column 2
uic_var = tk.StringVar()
uic_entry = ttk.Entry(frame, textvariable=uic_var, width=8)
uic_entry.grid(row=1, column=2, sticky=tk.W)  # Place the UIC entry in column 2

# Date selector
label_date = ttk.Label(frame, text="Date (yyyy-mm-dd):")
label_date.grid(row=0, column=3, sticky=tk.W)  # Place the label in column 2
date_var = tk.StringVar()
date_entry = ttk.Entry(frame, textvariable=date_var, width=8)
date_entry.grid(row=1, column=3, sticky=tk.W)  # Place the UIC entry in column 2

# Get Data Button
button = ttk.Button(frame, text="Get Data", command=main, width=10)
button.grid(row=2, column=0, sticky=tk.W)

# Download Button
button = ttk.Button(frame, text="Download", command=main)
button.grid(row=2, column=2, sticky=tk.W)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
frame.grid_rowconfigure(3, weight=1)
frame.grid_columnconfigure(0, weight=1)

root.mainloop()
