import pandas as pd
from pathlib import Path

def clean_raw_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if 'Ignore' in df.columns:
        df = df.drop(columns=['Ignore'])

    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    df = df.set_index('Open time').sort_index()

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if df.isnull().sum().sum() > 0:
        df = df.ffill().dropna()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return df

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    RAW_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "Raw" / "btc_1h_data_2018_to_2025.csv").resolve()
    CLEAN_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "Clean" / "kuru_clean_btc_1h.csv").resolve()

    if RAW_CSV_PATH.exists():
        clean_df = clean_raw_data(RAW_CSV_PATH, CLEAN_CSV_PATH)
        print(f"1-Hour Data cleaned. Shape: {clean_df.shape}")
    else:
        print(f"File not found: {RAW_CSV_PATH}")