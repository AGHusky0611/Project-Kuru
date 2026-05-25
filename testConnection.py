import ccxt

exchange = ccxt.binance({
    'apiKey': 'ZBVc7j20BwFto28WfpDon6H699G3q0rHec86ATuccDCTudLowh7yWos5dFfuUim2',
    'secret': 'f7FCayXXY38LntKYHv5TJB0gNe4BGAujTeA2uK538B8OVsHrGpTDcsmmBBbLcfGl',
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True)

# Test 1 - balance
balance = exchange.fetch_balance()
print("USDT:", balance['USDT']['free'])
print("BTC:", balance['BTC']['free'])

# Test 2 - market data
ticker = exchange.fetch_ticker('BTC/USDT')
print("BTC price:", ticker['last'])

# Test 3 - OHLCV
bars = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=5)
print("OHLCV rows:", len(bars))

print("\nConnection OK!")