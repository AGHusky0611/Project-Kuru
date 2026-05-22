import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

def engineer_kuru_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, index_col='Open time', parse_dates=True)

    df = df[~df.index.duplicated(keep='last')]

    df['Buyer_Aggression'] = df['Taker buy base asset volume'] / df['Volume']
    df['Avg_Trade_Size'] = df['Volume'] / df['Number of trades']
    
    taker_avg_price = df['Taker buy quote asset volume'] / df['Taker buy base asset volume']
    df['Taker_Price_Premium'] = taker_avg_price - df['Close']

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna()

    # Ensure the output directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    
    return df

BASE_DIR = Path(__file__).resolve().parent
CLEAN_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "kuru_clean_btc_15m.csv").resolve()
FEATURES_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "kuru_features_btc_15m.csv").resolve()

if CLEAN_CSV_PATH.exists():
    features_df = engineer_kuru_features(CLEAN_CSV_PATH, FEATURES_CSV_PATH)
    print(f"Features engineered. Final shape: {features_df.shape}")
else:
    print(f"File not found: {CLEAN_CSV_PATH}")