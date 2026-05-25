from pathlib import Path
from SupportModels.DataCleaning import clean_raw_data
from SupportModels.TechnicalIndicators import engineer_kuru_features
from Models.XGBoost_Train import tune_and_train_kuru

class KuruMasterTrainer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.raw_dir = self.base_dir / "Datasets" / "Raw"
        self.clean_dir = self.base_dir / "Datasets" / "Clean"
        self.results_dir = self.base_dir / "results" / "xlsx"
        
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train_timeframe(self, timeframe: str, raw_csv: str, noise_threshold: float, param_grid: dict, n_iter: int):
        print(f"\n{'='*50}\n INITIATING PROJECT KURU PIPELINE: {timeframe.upper()} \n{'='*50}")
        
        raw_path = self.raw_dir / raw_csv
        clean_path = self.clean_dir / f"kuru_clean_btc_{timeframe}.csv"
        features_path = self.clean_dir / f"kuru_features_btc_{timeframe}.csv"
        model_name = f"kuru_live_model_{timeframe}.json"
        
        if not raw_path.exists():
            print(f"[ERROR] Dataset not found: {raw_path}")
            return

        print(f"[1/3] Cleaning Data...")
        clean_raw_data(raw_path, clean_path)
        
        print(f"[2/3] Engineering Features (Noise Filter: {noise_threshold})...")
        engineer_kuru_features(clean_path, features_path, noise_threshold)
        
        print(f"[3/3] Tuning and Training Model...")
        tune_and_train_kuru(features_path, self.results_dir, model_name, param_grid, n_iter)

    def execute_all(self):
        grid_15m = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        }
        self.train_timeframe("15m", "btc_15m_data_2018_to_2025.csv", 0.001, grid_15m, 10)

        grid_60m = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [5, 10, 20],
        "gamma": [0, 0.5, 1, 2],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 2, 5],
    }
        self.train_timeframe("1h", "btc_1h_data_2018_to_2025.csv", 0.002, grid_60m, 100)

if __name__ == "__main__":
    trainer = KuruMasterTrainer()
    trainer.execute_all()