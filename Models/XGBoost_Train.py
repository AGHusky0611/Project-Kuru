import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

def tune_and_train_kuru(features_path: Path, results_dir: Path):
    df = pd.read_csv(features_path, index_col='Open time', parse_dates=True)

    exclude_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Quote asset volume', 'Number of trades', 
        'Taker buy base asset volume', 'Taker buy quote asset volume', 
        'Close time', 'Target'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['Target']

    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Generate 50 random combinations from the grid
    param_list = list(ParameterSampler(param_grid, n_iter=50, random_state=42))
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_results = []
    best_score = 0
    best_params = None

    print("Initiating Custom Time-Series Hyperparameter Search...\n")

    # Custom Loop to print every run
    for i, params in enumerate(param_list, 1):
        print(f"Run {i}/50 | Hyperparameters: {params}")
        
        fold_accuracies = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = XGBClassifier(**params, random_state=42, eval_metric='logloss', n_jobs=-1)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            
            preds = model.predict(X_fold_val)
            fold_accuracies.append(accuracy_score(y_fold_val, preds))
            
        avg_acc = np.mean(fold_accuracies)
        print(f"--> Result: {avg_acc * 100:.2f}% Accuracy\n")
        
        # Save results for Excel
        run_data = params.copy()
        run_data['Mean_Accuracy'] = avg_acc
        all_results.append(run_data)
        
        # Track the best model
        if avg_acc > best_score:
            best_score = avg_acc
            best_params = params

    results_dir.mkdir(parents=True, exist_ok=True)
    excel_path = results_dir / "hyperparameter_tuning_results.xlsx"
    
    results_df = pd.DataFrame(all_results).sort_values(by='Mean_Accuracy', ascending=False)
    results_df.to_excel(excel_path, index=False)
    print(f"All 50 runs saved to: {excel_path}")

    print("\n--- Optimal Hyperparameters Found ---")
    print(best_params)
    print(f"Cross-Validation Accuracy: {best_score * 100:.2f}%")

    # Train final model on the entire 80% using the best parameters
    print("\nTraining final model and evaluating on unseen 20% Test Set...")
    final_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss', n_jobs=-1)
    final_model.fit(X_train, y_train, verbose=False)

    predictions = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nFinal Live-Environment Accuracy: {final_accuracy * 100:.2f}%\n")
    print(classification_report(y_test, predictions))
    
    return final_model

BASE_DIR = Path(__file__).resolve().parent
FEATURES_CSV_PATH = (BASE_DIR / ".." / "Datasets" / "kuru_features_btc_15m.csv").resolve()
RESULTS_DIR = (BASE_DIR / ".." / "results" / "xlsx").resolve()

if FEATURES_CSV_PATH.exists():
    best_live_model = tune_and_train_kuru(FEATURES_CSV_PATH, RESULTS_DIR)
else:
    print(f"File not found: {FEATURES_CSV_PATH}")