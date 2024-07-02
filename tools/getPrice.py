import requests
import os
import json
import matplotlib.pyplot as plt
from lightweight_charts import Chart
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

apikey = os.getenv('API_KEY')

def main():

    print("Available asset types are: ")
    print("1. CfdOnIndex")
    print("2. CfdOnFutures")
    print("3. FxSpot")
    print("4. Stock")
    print("5. CfdOnStock")
    print("6. StockIndex")

    user_input = input("Which asset type are you looking for: ")
    
    while user_input not in ["1", "2", "3", "4", "5", "6"]:
        print("Invalid input. Enter a corresponding number. Please try again.")
        user_input = input("Which asset type are you looking for: ")
    if user_input == "1":
        assetType = "CfdOnIndex"
    elif user_input == "2":
        assetType = "CfdOnFutures"
    elif user_input == "3":
        assetType = "FxSpot"
    elif user_input == "4":
        assetType = "Stock"
    elif user_input == "5":
        assetType = "CfdOnStock"
    elif user_input == "6":
        assetType = "StockIndex"

    load_dotenv()
    

    headers = {
        "Authorization": f"Bearer {apikey}"
    }

    response = get_data(headers, assetType)
    formatted_response = format_json(response)
    description = [data['Description'] for data in formatted_response['Data']]
    identifier = [data['Identifier'] for data in formatted_response['Data']]
    symbol = [data['Symbol'] for data in formatted_response['Data']]

    for i in range(len(description)):
        print(f"Instrument: {description[i]}, Symbol: {symbol[i]}, UIC: {identifier[i]}")

    uic_input = input("Enter the UIC of the asset you are interested in: ")
    period = input("Enter the period for the historical prices (1, 5, 15, 30, 60, 1440): ")
    

    response = get_historical_prices(headers, assetType, uic_input, period=period)
    formatted_response = format_json(response)

    #Get all the necessary data from the response
    close_ask_values = [data['CloseAsk'] for data in formatted_response['Data']]
    time_values = [datetime.strptime(data['Time'], '%Y-%m-%dT%H:%M:%S.%fZ') for data in formatted_response['Data']]
    high_ask_values = [data['HighAsk'] for data in formatted_response['Data']]
    low_ask_values = [data['LowAsk'] for data in formatted_response['Data']]
    open_ask_values = [data['OpenAsk'] for data in formatted_response['Data']]
    chart = Chart()

    #Then we make a pandas datafram with the data, into a csv file
    df = pd.DataFrame({"Date": time_values, "Open": open_ask_values, "High": high_ask_values, "Low": low_ask_values, "Close": close_ask_values})
    df.to_csv("prices.csv", index=False)

    df = pd.read_csv("prices.csv")
    chart.set(df)
    chart.show(block=True)


    #print(formatted_response)

    """ 
    for i in range(len(close_ask_values)):
        print(f"Time: {time_values[i]} CloseAsk: {close_ask_values[i]}")

    chart = Chart()
    """

    """
    plt.figure(figsize=(13, 8))
    plt.plot(time_values, close_ask_values)
    plt.xlabel('Time')
    plt.ylabel('CloseAsk')
    plt.title('Time vs CloseAsk')
    plt.show()
    """


def get_data(headers, assetType):
    endpoint = "https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?AssetTypes=" + assetType
    response = requests.get(endpoint, headers=headers)
    return response.json()

def get_historical_prices(headers, assetType, uic, period):
    base_url = "https://gateway.saxobank.com/sim/openapi"
    endpoint = f"/chart/v1/charts?AssetType={assetType}&Horizon={period}&Uic={uic}"
    response = requests.get(base_url + endpoint, headers=headers)
    return response.json()

def format_json(json_data):
    return json.loads(json.dumps(json_data, indent=4))

if __name__ == "__main__":
    main()