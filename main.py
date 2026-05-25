import time
import os
import json
import csv
import logging
import ccxt
import pandas as pd
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier
from SupportModels.TechnicalIndicators import engineer_live_features

logging.basicConfig(
    filename="kuru.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 1. Initialize Exchange (Use Testnet for safety)
api_key = os.getenv("KURU_API_KEY")
api_secret = os.getenv("KURU_API_SECRET")
if not api_key or not api_secret:
    raise ValueError("Missing KURU_API_KEY or KURU_API_SECRET environment variables.")

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})
exchange.set_sandbox_mode(True) 

SYMBOL = 'BTC/USDT'
TRADE_SIZE = 0.01

starting_balance = None
cooldown_candles = 0
entry_price = None
entry_time = None

DATA_DIR = Path("data")
TRADES_CSV = DATA_DIR / "trades.csv"
TRADES_JSONL = DATA_DIR / "trades.jsonl"
SIGNALS_JSONL = DATA_DIR / "signals.jsonl"

def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")

def append_jsonl(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload) + "\n")

def append_trade_csv(payload: dict):
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = TRADES_CSV.exists()
    with TRADES_CSV.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)

def record_trade(payload: dict):
    append_trade_csv(payload)
    append_jsonl(TRADES_JSONL, payload)

def record_signal(timeframe: str, prob_up: float):
    payload = {
        "timestamp": utc_now_iso(),
        "timeframe": timeframe,
        "confidence": round(prob_up * 100, 4)
    }
    append_jsonl(SIGNALS_JSONL, payload)

def safe_fetch(fn, retries=3):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            logging.warning(f"[RETRY {i + 1}] {e}")
            time.sleep(2 ** i)
    return None

def has_sufficient_balance(min_usdt=20):
    balance = safe_fetch(exchange.fetch_balance)
    if not balance:
        return False
    usdt = balance.get('USDT', {}).get('free', 0)
    if usdt < min_usdt:
        logging.info(f"[SKIP] Insufficient balance: {usdt} USDT")
        return False
    return True

def check_drawdown(max_drawdown=0.05):
    global starting_balance
    balance = safe_fetch(exchange.fetch_balance)
    if not balance:
        return False
    current = balance.get('USDT', {}).get('total', 0)
    if starting_balance is None:
        starting_balance = current
        return False
    if starting_balance == 0:
        return False
    drawdown = (starting_balance - current) / starting_balance
    if drawdown > max_drawdown:
        logging.error(f"[KILL SWITCH] Drawdown {drawdown * 100:.1f}% exceeded limit. Shutting down.")
        return True
    return False

def should_skip_due_to_cooldown():
    global cooldown_candles
    if cooldown_candles > 0:
        logging.info(f"[COOLDOWN] Skipping. {cooldown_candles} candles remaining.")
        cooldown_candles -= 1
        return True
    return False

def place_stop_loss(entry_price_value):
    stop_price = entry_price_value * 0.98
    safe_fetch(lambda: exchange.create_order(
        SYMBOL,
        'stop',
        'sell',
        TRADE_SIZE,
        stop_price,
        {'stopPrice': stop_price}
    ))
    logging.info(f"[STOP-LOSS] Placed at {stop_price:.2f}")

def sync_startup_state():
    logging.info("[STARTUP] Syncing position state from exchange...")
    try:
        balance = exchange.fetch_balance()
        btc_held = balance.get('BTC', {}).get('free', 0)
        if btc_held > 0.001:
            logging.info("[STARTUP] Found existing BTC position. Recovering as LONG.")
            return "LONG"
        logging.info("[STARTUP] No open position found. Starting FLAT.")
        return "FLAT"
    except Exception as e:
        logging.warning(f"[STARTUP WARN] Could not sync state: {e}. Defaulting to FLAT.")
        return "FLAT"

def seconds_until_next_candle(interval_minutes):
    now = datetime.now()
    total_seconds = interval_minutes * 60
    elapsed = (now.minute % interval_minutes) * 60 + now.second
    return total_seconds - elapsed

def fetch_live_data(timeframe: str) -> pd.DataFrame:
    bars = safe_fetch(lambda: exchange.fetch_ohlcv(SYMBOL, timeframe, limit=50))
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df = df.set_index('Open time')

    return df

def load_threshold(threshold_path: Path, default_threshold: float = 0.65) -> float:
    try:
        if threshold_path.exists():
            with threshold_path.open('r') as handle:
                payload = json.load(handle)
            return float(payload.get('threshold', default_threshold))
    except Exception as e:
        logging.warning(f"[THRESHOLD] Failed to load {threshold_path.name}: {e}")
    return default_threshold

def execute_trade(direction: str, current_position: str):
    global entry_price, entry_time, cooldown_candles
    try:
        if direction == 'LONG' and current_position == 'FLAT':
            if not has_sufficient_balance():
                return current_position
            logging.info(f"[EXECUTING] Market BUY {TRADE_SIZE} {SYMBOL}")
            order = safe_fetch(lambda: exchange.create_market_buy_order(SYMBOL, TRADE_SIZE))
            if order:
                entry_price = order.get('average') or order.get('price')
                entry_time = utc_now_iso()
                if entry_price:
                    place_stop_loss(entry_price)
                else:
                    logging.warning("[STOP-LOSS] Entry price unavailable; stop-loss not placed.")
            return 'LONG'
        if direction == 'FLAT' and current_position == 'LONG':
            logging.info(f"[EXECUTING] Market SELL {TRADE_SIZE} {SYMBOL}")
            order = safe_fetch(lambda: exchange.create_market_sell_order(SYMBOL, TRADE_SIZE))
            exit_price = None
            if order:
                exit_price = order.get('average') or order.get('price')
            if entry_price and exit_price:
                pnl = (exit_price - entry_price) * TRADE_SIZE
                pnl_pct = (exit_price - entry_price) / entry_price
                payload = {
                    "trade_id": entry_time or utc_now_iso(),
                    "symbol": SYMBOL,
                    "side": "LONG",
                    "size": TRADE_SIZE,
                    "entry_time": entry_time,
                    "entry_price": round(entry_price, 8),
                    "exit_time": utc_now_iso(),
                    "exit_price": round(exit_price, 8),
                    "pnl": round(pnl, 8),
                    "pnl_pct": round(pnl_pct * 100, 6),
                }
                record_trade(payload)
                if exit_price < entry_price:
                    cooldown_candles = 3
            else:
                logging.warning("[TRADE] Missing entry or exit price; trade not recorded.")
            entry_price = None
            entry_time = None
            return 'FLAT'
    except Exception as e:
        logging.error(f"[API ERROR] Failed to execute trade: {e}")
    return current_position

def run_live_inference(model, timeframe: str):
    logging.info(f"Fetching live {timeframe} data...")
    raw_df = fetch_live_data(timeframe)
    if raw_df.empty:
        logging.warning(f"[{timeframe}] No data fetched.")
        return None
    
    live_features = engineer_live_features(raw_df)
    if live_features.empty:
        logging.warning(f"[{timeframe}] No live features available.")
        return None
    
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_live = live_features.drop(columns=[c for c in exclude_cols if c in live_features.columns])
    if X_live.empty:
        logging.warning(f"[{timeframe}] No model inputs available.")
        return None
    
    prob_up = model.predict_proba(X_live)[-1][1]
    
    logging.info(f"[{timeframe}] Model Confidence for UP: {prob_up * 100:.2f}%")
    record_signal(timeframe, prob_up)
    return prob_up

def load_models():
    BASE_DIR = Path(__file__).resolve().parent
    model_dir = (BASE_DIR / "Models").resolve()
    
    logging.info("Loading Project Kuru AI Models...")
    model_15m, model_1h = None, None
    threshold_15m, threshold_1h = 0.65, 0.65
    
    path_15m = model_dir / "kuru_live_model_15m.json"
    if path_15m.exists():
        model_15m = XGBClassifier()
        model_15m.load_model(str(path_15m))
        threshold_15m = load_threshold(path_15m.with_name("kuru_live_model_15m_threshold.json"))
        logging.info("[OK] 15-Minute Model loaded successfully.")
    else:
        logging.warning("[WARN] 15-Minute Model not found. Skipping 15m execution.")

    path_1h = model_dir / "kuru_live_model_1h.json"
    if path_1h.exists():
        model_1h = XGBClassifier()
        model_1h.load_model(str(path_1h))
        threshold_1h = load_threshold(path_1h.with_name("kuru_live_model_1h_threshold.json"))
        logging.info("[OK] 60-Minute Model loaded successfully.")
    else:
        logging.warning("[WARN] 60-Minute Model not found. Skipping 1h execution.")
        
    return model_15m, model_1h, threshold_15m, threshold_1h

def execute_15m_model(model):
    if model is None: 
        return None
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] --> Firing 15-Minute Sniper Model")
    return run_live_inference(model, '15m')

def execute_60m_model(model):
    if model is None: 
        return None
    logging.info(f"[{datetime.now().strftime('%H:%M:%S')}] --> Firing 60-Minute Trend Model")
    return run_live_inference(model, '1h')

def run_kuru_orchestrator():
    logging.info("Initializing Project Kuru Orchestrator...")
    model_15m, model_1h, threshold_15m, threshold_1h = load_models()
    
    if model_15m is None and model_1h is None:
        logging.error("[ERROR] No models found. Run train.py first. Exiting.")
        return
    
    current_position = sync_startup_state()
    last_prob_1h = None
    if model_1h is not None:
        last_prob_1h = execute_60m_model(model_1h)
        
    logging.info("System active. Waiting for next candle close...")
    
    while True:
        if check_drawdown():
            return

        current_position = sync_startup_state()

        sleep_15m = seconds_until_next_candle(15)
        time.sleep(sleep_15m)

        prob_15m = execute_15m_model(model_15m)
        if datetime.now().minute == 0 and model_1h is not None:
            last_prob_1h = execute_60m_model(model_1h)

        if should_skip_due_to_cooldown():
            continue

        if prob_15m is None or last_prob_1h is None:
            continue

        if prob_15m > threshold_15m and last_prob_1h > threshold_1h:
            current_position = execute_trade('LONG', current_position)
        elif prob_15m < 0.35 and current_position == 'LONG':
            current_position = execute_trade('FLAT', current_position)

if __name__ == "__main__":
    run_kuru_orchestrator()