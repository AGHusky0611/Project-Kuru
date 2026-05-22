import time
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier

import time
import ccxt
import pandas as pd
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier
from SupportModels.TechnicalIndicators import engineer_live_features

# 1. Initialize Exchange (Use Testnet for safety)
exchange = ccxt.binance({
    'apiKey': 'YOUR_BINANCE_TESTNET_API_KEY',
    'secret': 'YOUR_BINANCE_TESTNET_SECRET',
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True) # Connects to Binance Testnet

SYMBOL = 'BTC/USDT'
TRADE_SIZE = 0.01 # Amount of BTC to buy

def fetch_live_data(timeframe: str) -> pd.DataFrame:
    """Fetches the last 50 candles to calculate 21-EMA and 14-RSI correctly."""
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=50)
    
    df = pd.DataFrame(bars, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df = df.set_index('Open time')
    
    # Mocking Binance Order Book data for live execution
    # In a full production environment, fetch actual taker volumes via WebSocket
    df['Number of trades'] = 1000 
    df['Taker buy base asset volume'] = df['Volume'] * 0.5 
    df['Taker buy quote asset volume'] = df['Volume'] * df['Close'] * 0.5
    
    return df

def execute_trade(direction: str):
    """Sends the actual execution order to the exchange."""
    try:
        if direction == 'LONG':
            print(f"[EXECUTING] Market BUY {TRADE_SIZE} {SYMBOL}")
            exchange.create_market_buy_order(SYMBOL, TRADE_SIZE)
            # Add Stop Loss logic here
    except Exception as e:
        print(f"[API ERROR] Failed to execute trade: {e}")

def run_live_inference(model, timeframe: str):
    print(f"Fetching live {timeframe} data...")
    raw_df = fetch_live_data(timeframe)
    
    # Process features
    live_features = engineer_live_features(raw_df)
    
    # Drop columns not used in training
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    X_live = live_features.drop(columns=[c for c in exclude_cols if c in live_features.columns])
    
    # Predict Probability
    prob_up = model.predict_proba(X_live)
    
    print(f"[{timeframe}] Model Confidence for UP: {prob_up * 100:.2f}%")
    
    # Sniper Execution Logic
    if prob_up > 0.60:
        print(f"[{timeframe}] HIGH CONFIDENCE BULLISH SIGNAL DETECTED.")
        execute_trade('LONG')
    else:
        print(f"[{timeframe}] No trade conditions met. Standing by.")

def load_models():
    BASE_DIR = Path(__file__).resolve().parent
    model_dir = (BASE_DIR / "Models").resolve()
    
    print("Loading Project Kuru AI Models...")
    model_15m, model_1h = None, None
    
    path_15m = model_dir / "kuru_live_model_15m.json"
    if path_15m.exists():
        model_15m = XGBClassifier()
        model_15m.load_model(str(path_15m))
        print("[OK] 15-Minute Model loaded successfully.")
    else:
        print("[WARN] 15-Minute Model not found. Skipping 15m execution.")

    path_1h = model_dir / "kuru_live_model_1h.json"
    if path_1h.exists():
        model_1h = XGBClassifier()
        model_1h.load_model(str(path_1h))
        print("[OK] 60-Minute Model loaded successfully.")
    else:
        print("[WARN] 60-Minute Model not found. Skipping 1h execution.")
        
    return model_15m, model_1h

def execute_15m_model(model):
    if model is None: 
        return
    print(f"[{datetime.now().strftime('%H:%M:%S')}] --> Firing 15-Minute Sniper Model")
    # Live data fetching via CCXT and prediction logic goes here

def execute_60m_model(model):
    if model is None: 
        return
    print(f"[{datetime.now().strftime('%H:%M:%S')}] --> Firing 60-Minute Trend Model")
    # Live data fetching via CCXT and prediction logic goes here

def run_kuru_orchestrator():
    print("Initializing Project Kuru Orchestrator...")
    model_15m, model_1h = load_models()
    
    if model_15m is None and model_1h is None:
        print("[ERROR] No models found. Run train.py first. Exiting.")
        return
        
    print("System active. Waiting for next candle close...")
    
    while True:
        current_time = datetime.now()
        
        # Trigger only at the exact start of a new minute
        if current_time.second == 0:
            
            # The 15-Minute Trigger (:00, :15, :30, :45)
            if current_time.minute % 15 == 0:
                execute_15m_model(model_15m)
                
            # The 60-Minute Trigger (:00)
            if current_time.minute == 0:
                execute_60m_model(model_1h)
                
            # Sleep for 60 seconds to prevent double-firing within the same minute
            time.sleep(60)
        else:
            # Sleep for 1 second and check the clock again
            time.sleep(1)

if __name__ == "__main__":
    run_kuru_orchestrator()