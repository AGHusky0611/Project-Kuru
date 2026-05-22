from pathlib import Path
from SupportModels.60TDataCleaner import clean_raw_data
from SupportModels.60TTechnicalIndicators import engineer_kuru_features
from Models.XGBoost_Train import tune_and_train_kuru

class KuruMasterTrainer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.datasets_dir = self.base_dir / "Datasets"
        self.results_dir = self.base_dir / "results" / "xlsx"
        
        # Ensure directories exist
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train_timeframe(self, timeframe: str, raw_csv: str, noise_threshold: float, param_grid: dict, n_iter: int):
        print(f"\n{'='*50}")
        print(f" INITIATING PROJECT KURU PIPELINE: {timeframe.upper()} ")
        print(f"{'='*50}")
        
        raw_path = self.datasets_dir / raw_csv
        clean_path = self.datasets_dir / f"kuru_clean_btc_{timeframe}.csv"
        features_path = self.datasets_dir / f"kuru_features_btc_{timeframe}.csv"
        model_name = f"kuru_live_model_{timeframe}.json"
        
        if not raw_path.exists():
            print(f"[ERROR] Raw dataset not found: {raw_path}")
            return

        print(f"\n[1/3] Cleaning {timeframe} Data...")
        clean_raw_data(raw_path, clean_path)
        
        print(f"\n[2/3] Engineering {timeframe} Features (Noise Filter: {noise_threshold})...")
        engineer_kuru_features(clean_path, features_path, noise_threshold)
        
        print(f"\n[3/3] Tuning and Training {timeframe} Model...")
        tune_and_train_kuru(features_path, self.results_dir, model_name, param_grid, n_iter)
        
        print(f"\n [OK] {timeframe} Pipeline Complete!")

    def execute_all(self):
        # 15-Minute Configuration
        grid_15m = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
        self.train_timeframe(
            timeframe="15m", 
            raw_csv="btc_15m_data_2018_to_2025.csv", 
            noise_threshold=0.001, 
            param_grid=grid_15m, 
            n_iter=30
        )

        # 60-Minute Configuration (Shallower trees, higher noise threshold)
        grid_60m = {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8],
            "colsample_bytree": [0.6, 0.8],
        }
        self.train_timeframe(
            timeframe="1h", 
            raw_csv="btc_1h_data_2018_to_2025.csv", 
            noise_threshold=0.002, 
            param_grid=grid_60m, 
            n_iter=20
        )
        print("\n ALL MODELS SUCCESSFULLY TRAINED AND DEPLOYED TO /MODELS/")

if __name__ == "__main__":
    trainer = KuruMasterTrainer()
    trainer.execute_all()