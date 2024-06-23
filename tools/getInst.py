import json
import requests

print("Available asset types are: ")
print("1. CfdOnIndex")
print("2. CfdOnFutures")
print("3. FxSpot")
print("4. Stock")
print("5. CfdOnStock")
print("6. StockIndex")
#Ask user for input and save to variable
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

def main():
    headers = {
        "Authorization": "Bearer eyJhbGciOiJFUzI1NiIsIng1dCI6IjI3RTlCOTAzRUNGMjExMDlBREU1RTVCOUVDMDgxNkI2QjQ5REEwRkEifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiTTZNSnltdmxwcG1PYjF8VTI2QktDdz09IiwiY2lkIjoiTTZNSnltdmxwcG1PYjF8VTI2QktDdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiZTNkYjFlM2E0NTVlNDJmOWE5OTQ0N2M1ZGFiZGE0OGQiLCJkZ2kiOiI4NCIsImV4cCI6IjE3MjQ5NDI2MjQiLCJvYWwiOiIxRiIsImlpZCI6ImQ1M2ExM2MzZjhjODQzM2FhMmUwMDhkYzYwZjUwZDQzIn0.bmJmSaDkDHZ8IbxMHOqc2KRTw7rk83ZDlstcN3oa6wbDdfFiECpX0TrQvv5Y3IxuuGHvZBWhQg_sQ9VypJI6Mg"
    }
    response = get_data(headers, assetType)
    
    formatted_response = format_json(response)
    
    description = [data['Description'] for data in formatted_response['Data']]
    identifier = [data['Identifier'] for data in formatted_response['Data']]
    exhangeID = [data['ExchangeId'] for data in formatted_response['Data']]
    symbol = [data['Symbol'] for data in formatted_response['Data']]

    for i in range(len(description)):
        print(f"Instrument: {description[i]}, Symbol: {symbol[i]}, UIC: {identifier[i]}")


def get_data(headers, assetType):
    endpoint = "https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?AssetTypes=" + assetType
    response = requests.get(endpoint, headers=headers)
    return response.json()

def format_json(json_data):
    return json.loads(json.dumps(json_data, indent=2))


if __name__ == "__main__":
    main()