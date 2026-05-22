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
exchange.set_sandbox_mode(True) 

SYMBOL = 'BTC/USDT'
TRADE_SIZE = 0.01 

def fetch_live_data(timeframe: str) -> pd.DataFrame:
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=50)
    
    df = pd.DataFrame(bars, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df = df.set_index('Open time')
    
    df['Number of trades'] = 1000 
    df['Taker buy base asset volume'] = df['Volume'] * 0.5 
    df['Taker buy quote asset volume'] = df['Volume'] * df['Close'] * 0.5
    
    return df

def execute_trade(direction: str):
    try:
        if direction == 'LONG':
            print(f"[EXECUTING] Market BUY {TRADE_SIZE} {SYMBOL}")
            exchange.create_market_buy_order(SYMBOL, TRADE_SIZE)
    except Exception as e:
        print(f"[API ERROR] Failed to execute trade: {e}")

def run_live_inference(model, timeframe: str):
    print(f"Fetching live {timeframe} data...")
    raw_df = fetch_live_data(timeframe)
    
    live_features = engineer_live_features(raw_df)
    
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    X_live = live_features.drop(columns=[c for c in exclude_cols if c in live_features.columns])
    
    prob_up = model.predict_proba(X_live)
    
    print(f"[{timeframe}] Model Confidence for UP: {prob_up * 100:.2f}%")
    
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
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --> Firing 15-Minute Sniper Model")
    run_live_inference(model, '15m')

def execute_60m_model(model):
    if model is None: 
        return
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --> Firing 60-Minute Trend Model")
    run_live_inference(model, '1h')

def run_kuru_orchestrator():
    print("Initializing Project Kuru Orchestrator...")
    model_15m, model_1h = load_models()
    
    if model_15m is None and model_1h is None:
        print("[ERROR] No models found. Run train.py first. Exiting.")
        return
        
    print("System active. Waiting for next candle close...")
    
    while True:
        current_time = datetime.now()
        
        if current_time.second == 0:
            if current_time.minute % 15 == 0:
                execute_15m_model(model_15m)
                
            if current_time.minute == 0:
                execute_60m_model(model_1h)
                
            time.sleep(60)
        else:
            time.sleep(1)

if __name__ == "__main__":
    run_kuru_orchestrator()