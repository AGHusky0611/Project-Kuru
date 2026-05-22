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

    df['Return'] = df['Close'].pct_change()
    df['Return_Lag_1'] = df['Return'].shift(1)
    df['Return_Lag_2'] = df['Return'].shift(2)
    df['Rolling_Vol_14'] = df['Return'].rolling(window=14).std()

    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    df['Next_Return'] = df['Return'].shift(-1)
    noise_threshold = 0.002 # Slightly higher noise threshold for 1-hour candles
    df = df[df['Next_Return'].abs() > noise_threshold]
    df['Target'] = (df['Next_Return'] > 0).astype(int)

    df = df.drop(columns=['Return', 'Next_Return'])
    df = df.dropna()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CLEAN_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "Clean" / "kuru_clean_btc_1h.csv").resolve()
    FEATURES_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "Clean" / "kuru_features_btc_1h.csv").resolve()

    if CLEAN_CSV_PATH.exists():
        features_df = engineer_kuru_features(CLEAN_CSV_PATH, FEATURES_CSV_PATH)
        print(f"1-Hour Features engineered. Shape: {features_df.shape}")
    else:
        print(f"File not found: {CLEAN_CSV_PATH}")