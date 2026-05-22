import time
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier

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