import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def main():
    print("Loading 2024 data...")
    # Load data
    df = pd.read_csv("data/cdc_2024/harmonized_cdc_2024.csv", dtype=str)
    
    # 2. Filter Primary Population
    df["is_us_resident"] = pd.to_numeric(df["is_us_resident"], errors="coerce")
    df["is_singleton"] = pd.to_numeric(df["is_singleton"], errors="coerce")
    df["birth_weight_g"] = pd.to_numeric(df["birth_weight_g"], errors="coerce")
    
    # Valid LBW condition: birth weight between 227 and 8165
    valid_bw = (df["birth_weight_g"] >= 227) & (df["birth_weight_g"] <= 8165)
    
    mask = (df["is_us_resident"] == 1) & (df["is_singleton"] == 1) & valid_bw
    df_eval = df[mask].copy()
    
    print(f"Primary LBW population size in 2024: {len(df_eval):,}")
    
    # 3. Target and Features
    df_eval["target_lbw"] = (df_eval["birth_weight_g"] < 2500).astype(int)
    prevalence = df_eval["target_lbw"].mean() * 100
    print(f"Actual LBW prevalence in 2024: {prevalence:.2f}%")
    
    # 4. Load Model
    model_path = Path("models/target_lbw__landmark_strict__lightgbm.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Did you download it from ClearML?")
        
    data = joblib.load(model_path)
    preprocessor = data["preprocessor"]
    model = data["model"]
    features = data["features"]
    
    # Convert numerical features
    for f in features:
        if f != "mother_race":
            df_eval[f] = pd.to_numeric(df_eval[f], errors="coerce")
            
    # 5. Predict Scores
    X_eval = df_eval[features]
    print("Applying preprocessing...")
    X_trans = preprocessor.transform(X_eval)
    
    print("Predicting scores...")
    y_pred_proba = model.predict_proba(X_trans)[:, 1]
    
    # 6. Score Distribution
    mean_score = np.mean(y_pred_proba)
    median_score = np.median(y_pred_proba)
    p90_score = np.percentile(y_pred_proba, 90)
    
    print("\n--- LBW Model Score Distribution in 2024 ---")
    print(f"Mean:   {mean_score:.4f}")
    print(f"Median: {median_score:.4f}")
    print(f"90th %: {p90_score:.4f}")
    
    # The 2022 threshold was 0.118980.
    selected = (y_pred_proba >= 0.118980).mean() * 100
    print(f"Selected at threshold 0.118980: {selected:.2f}%")
    
    # 7. Covariate Drift
    print("\n--- Key Risk Features Covariate Shift (Means) ---")
    numeric_features = [f for f in features if f != "mother_race"]
    
    for nf in numeric_features:
        mean_val = df_eval[nf].mean()
        # Some are binary, some are continuous
        print(f"  {nf:<25}: {mean_val:.4f}")

if __name__ == "__main__":
    main()
