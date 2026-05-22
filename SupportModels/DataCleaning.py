import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "btc_15m_data_2018_to_2025.csv").resolve()
CLEAN_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "kuru_clean_btc_15m.csv").resolve()

def preprocess_kuru_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
            
    if 'Ignore' in df.columns:
        df = df.drop(columns=['Ignore'])

    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    df = df.set_index('Open time').sort_index()

    numeric_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Quote asset volume', 'Number of trades', 
        'Taker buy base asset volume', 'Taker buy quote asset volume'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if df.isnull().sum().sum() > 0:
        df = df.ffill().dropna()
        
    # Save the cleaned data to a new file
    df.to_csv(output_path)
    
    return df

if os.path.exists(RAW_CSV_PATH):
    cleaned_df = preprocess_kuru_data(RAW_CSV_PATH, CLEAN_CSV_PATH)
    print(f"Clean data saved to {CLEAN_CSV_PATH}. Shape: {cleaned_df.shape}")
else:
    print(f"File not found: {RAW_CSV_PATH}")