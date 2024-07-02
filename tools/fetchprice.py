import requests
import os
import json
import matplotlib.pyplot as plt
from lightweight_charts import Chart
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

load_dotenv()

api_key = os.getenv('API_KEY')


def main():

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    assetType = "CfdOnIndex"
    period = 1
    uic = 4913

    df = pd.DataFrame()

    for month in range(1, 2):
        print("Fetching price data....")
        for day in range(1, 31):
            response = get_historical_prices(headers, assetType=assetType, uic=uic, period=period, day=day, month=month)
            formatted_response = format_json(response)
            #Get all the necessary data from the response
            close_ask_values = [data['CloseAsk'] for data in formatted_response['Data']]
            time_values = [datetime.strptime(data['Time'], '%Y-%m-%dT%H:%M:%S.%fZ') for data in formatted_response['Data']]
            high_ask_values = [data['HighAsk'] for data in formatted_response['Data']]
            low_ask_values = [data['LowAsk'] for data in formatted_response['Data']]
            open_ask_values = [data['OpenAsk'] for data in formatted_response['Data']]
            #Then we make a pandas datafram with the data, into a csv file
            new_data = pd.DataFrame({"Date": time_values, "Open": open_ask_values, "High": high_ask_values, "Low": low_ask_values, "Close": close_ask_values})
            df = pd.concat([df, new_data])
        df.to_csv("prices.csv", index=False)
        


def get_historical_prices(headers, assetType, uic, period, day, month):
    base_url = "https://gateway.saxobank.com/sim/openapi"
    endpoint = f"/chart/v1/charts?AssetType={assetType}&Horizon={period}&Mode=From&Time=2022-{month}-{day}&Uic={uic}"
    response = requests.get(base_url + endpoint, headers=headers)
    return response.json()


def format_json(json_data):
    return json.loads(json.dumps(json_data, indent=4))

if __name__ == "__main__":
    main()