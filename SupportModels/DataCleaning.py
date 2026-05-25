import pandas as pd
from pathlib import Path

def clean_raw_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    
    if 'Ignore' in df.columns:
        df = df.drop(columns=['Ignore'])

    df['Open time'] = pd.to_datetime(df['Open time'])
    df['Close time'] = pd.to_datetime(df['Close time'])
    df = df.set_index('Open time').sort_index()
    df = df[df.index.notna()]

    drop_cols = [
        'Close time',
        'Quote asset volume',
        'Number of trades',
        'Taker buy base asset volume',
        'Taker buy quote asset volume'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if df.isnull().sum().sum() > 0:
        df = df.ffill().dropna()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return df