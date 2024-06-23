import websocket
import json
import uuid
import requests
import urllib.parse

def is_websockets_supported():
    # In Python, we assume WebSockets are supported if the websocket-client library is installed
    try:
        import websocket
        return True
    except ImportError:
        return False

def create_connection(access_token, context_id, streamer_url):
    # Construct the WebSocket URL
    ws_url = f"{streamer_url}?authorization=BEARER%20{access_token}&contextId={context_id}"
    print(f"WebSocket URL: {ws_url}")

    # Check if WebSockets are supported
    if not is_websockets_supported():
        print("This environment doesn't support WebSockets.")
        raise Exception("This environment doesn't support WebSockets.")

    try:
        # Create the WebSocket connection
        connection = websocket.WebSocketApp(ws_url,
                                            on_message=on_message,
                                            on_error=on_error,
                                            on_close=on_close)
        connection.on_open = on_open
        print(f"Connection created. ReadyState: {connection.sock and connection.sock.connected}.")
        return connection
    except Exception as error:
        print(f"Error creating websocket. {error}")

def on_message(ws, message):
    print(f"Received message: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    if close_status_code == 1000 or close_status_code == 1001:
        print(f"Streaming disconnected with code {close_status_code}.")
    else:
        print(f"Streaming disconnected with code {close_status_code}.")

def on_open(ws):
    print("Streaming connected.")
    # Start listener once the WebSocket connection is open
    start_listener(api_url, access_token, context_id, reference_id, account_key, uic, asset_type, amount, field_groups)
    # Subscribe to prices
    subscribe_to_prices(api_url, access_token, context_id, reference_id, asset_type, uic, refresh_rate)

def start_listener(api_url, access_token, context_id, reference_id, account_key, uic, asset_type, amount, field_groups):
    # Define the subscription payload
    data = {
        "ContextId": context_id,
        "ReferenceId": reference_id,
        "Arguments": {
            "AccountKey": account_key,
            "Uic": uic,
            "AssetType": asset_type,
            "Amount": amount,
            "FieldGroups": field_groups
        }
    }

    # Define the headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "streaming.saxobank.com"
    }

    print(f"Subscription payload: {json.dumps(data, indent=4)}")
    print(f"Headers: {headers}")

    # Make the subscription request
    response = requests.post(f"{api_url}/trade/v1/prices/subscriptions", headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 201:
        print("Subscription successful")
        print(response.json())
    else:
        print(f"Failed to subscribe: {response.status_code} - {response.text}")

def subscribe_to_prices(api_url, access_token, context_id, reference_id, asset_type, uic, refresh_rate):
    # Define the subscription payload
    data = {
        "Arguments": {
            "AssetType": asset_type,
            "Uic": uic
        },
        "ContextId": context_id,
        "ReferenceId": reference_id,
        "RefreshRate": refresh_rate
    }

    # Define the headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "streaming.saxobank.com"
    }

    # Make the subscription request
    response = requests.post(f"{api_url}/trade/v1/prices/subscriptions", headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 201:
        print("Subscription successful")
        print(response.json())
    else:
        print(f"Failed to subscribe: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Replace with your actual access token
    access_token = "eyJhbGciOiJFUzI1NiIsIng1dCI6IjI3RTlCOTAzRUNGMjExMDlBREU1RTVCOUVDMDgxNkI2QjQ5REEwRkEifQ.eyJvYWEiOiI3Nzc3NSIsImlzcyI6Im9hIiwiYWlkIjoiMTA5IiwidWlkIjoiTTZNSnltdmxwcG1PYjF8VTI2QktDdz09IiwiY2lkIjoiTTZNSnltdmxwcG1PYjF8VTI2QktDdz09IiwiaXNhIjoiRmFsc2UiLCJ0aWQiOiIyMDAyIiwic2lkIjoiZTNkYjFlM2E0NTVlNDJmOWE5OTQ0N2M1ZGFiZGE0OGQiLCJkZ2kiOiI4NCIsImV4cCI6IjE3MjQ5NDI2MjQiLCJvYWwiOiIxRiIsImlpZCI6ImQ1M2ExM2MzZjhjODQzM2FhMmUwMDhkYzYwZjUwZDQzIn0.bmJmSaDkDHZ8IbxMHOqc2KRTw7rk83ZDlstcN3oa6wbDdfFiECpX0TrQvv5Y3IxuuGHvZBWhQg_sQ9VypJI6Mg"
    
    # Generate a unique context ID
    context_id = "MyApp_" + str(uuid.uuid4().int)[:13]
    print(f"Context ID: {context_id}")
    
    # Define the WebSocket URL
    streamer_url = "wss://streaming.saxobank.com/sim/openapi/streamingws/connect"
    
    # Define the API URL
    api_url = "https://gateway.saxobank.com/sim/openapi"
    
    # Replace with your actual asset type, horizon, and UIC
    asset_type = "FxSpot"
    horizon = 10
    uic = 21
    
    # Define the reference ID, account key, amount, and field groups
    reference_id = "MyPricesEvent"
    account_key = "M6MJymvlppmOb1|U26BKCw=="
    amount = 1000
    field_groups = [
        "Commissions",
        "DisplayAndFormat",
        "Greeks",
        "HistoricalChanges",
        "InstrumentPriceDetails",
        "MarginImpactBuySell",
        "MarketDepth",
        "PriceInfo",
        "PriceInfoDetails",
        "Quote",
        "Timestamps"
    ]

    # Define the refresh rate
    refresh_rate = 1000

    # Create a WebSocket connection
    connection = create_connection(access_token, context_id, streamer_url)
    if connection:
        connection.run_forever()
