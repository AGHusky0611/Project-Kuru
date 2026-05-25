import ccxt

# Paste the keys you generated on testnet.binance.vision
API_KEY = 'YOUR_TESTNET_API_KEY'
SECRET_KEY = 'YOUR_TESTNET_SECRET_KEY'

# Initialize the Binance client in Sandbox (Testnet) mode
exchange = ccxt.binance({
    'apiKey': 'ZBVc7j20BwFto28WfpDon6H699G3q0rHec86ATuccDCTudLowh7yWos5dFfuUim2',
    'secret': 'f7FCayXXY38LntKYHv5TJB0gNe4BGAujTeA2uK538B8OVsHrGpTDcsmmBBbLcfGl',
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True) 

print("Fetching Testnet Balance...")
balance = exchange.fetch_balance()

# Print the free funds Binance gave you
print(f"USDT Balance: {balance['USDT']['free']}")
print(f"BTC Balance:  {balance['BTC']['free']}")
print(f"BNB Balance:  {balance['BNB']['free']}")