import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

def tune_and_train_kuru(features_path: Path, results_dir: Path, model_name: str, param_grid: dict, n_iter: int):
    df = pd.read_csv(features_path, index_col='Open time', parse_dates=True)

    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Close time', 'Target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Target']

    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    tscv = TimeSeriesSplit(n_splits=5)
    best_score, best_params = 0, None

    for i, params in enumerate(param_list, 1):
        fold_accuracies = []
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = XGBClassifier(**params, random_state=42, eval_metric='logloss', n_jobs=-1)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            preds = model.predict(X_fold_val)
            fold_accuracies.append(accuracy_score(y_fold_val, preds))
            
        avg_acc = np.mean(fold_accuracies)
        if avg_acc > best_score:
            best_score = avg_acc
            best_params = params

    print(f"\n--- Optimal Hyperparameters Found ---")
    print(best_params)

    final_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss', n_jobs=-1)
    final_model.fit(X_train, y_train, verbose=False)

    probabilities = final_model.predict_proba(X_test)[:, 1]
    CONFIDENCE_THRESHOLD = 0.58 

    take_trade_mask = (probabilities > CONFIDENCE_THRESHOLD) | (probabilities < (1 - CONFIDENCE_THRESHOLD))
    y_test_sniper = y_test[take_trade_mask]
    preds_sniper = (probabilities[take_trade_mask] > 0.5).astype(int)

    if len(y_test_sniper) > 0:
        final_accuracy = accuracy_score(y_test_sniper, preds_sniper)
        print(f"\nFinal High-Confidence Accuracy: {final_accuracy * 100:.2f}%\n")
    else:
        print("\nNo trades met the confidence threshold.")

    model_path = results_dir.parent.parent / "Models" / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    
    return final_model