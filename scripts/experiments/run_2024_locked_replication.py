import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from clearml import Task, Dataset

def main():
    # 1. Initialize ClearML task
    print("Initializing ClearML task...")
    task = Task.init(
        project_name="pershin-medailab/Vache_Oganisyan/CDC Natality Audit",
        task_name="TAE_2024_Locked_Replication"
    )
    
    # 2. Fetch dataset
    print("Fetching harmonized 2024 dataset...")
    dataset_path = Dataset.get(dataset_id="a90d2c4a631e4a92bac407936ff417b5").get_local_copy()
    csv_files = list(Path(dataset_path).rglob("*.csv"))
    if not csv_files:
        raise ValueError("No CSV found in the dataset")
    
    print(f"Loading {csv_files[0]}...")
    # Match numeric MRACE31 categories used to fit the frozen preprocessor.
    df = pd.read_csv(csv_files[0], low_memory=False)
    print(f"Loaded {len(df)} rows from dataset.")
    
    # Check if target columns exist, if not, compute them based on harmonization structure
    if "target_preterm" not in df.columns:
        df["gestation_oe_weeks"] = pd.to_numeric(df["gestation_oe_weeks"], errors="coerce")
        df["birth_weight_g"] = pd.to_numeric(df["birth_weight_g"], errors="coerce")
        
        df["target_preterm"] = np.where(
            (df["gestation_oe_weeks"] >= 17) & (df["gestation_oe_weeks"] <= 47),
            (df["gestation_oe_weeks"] < 37).astype(float),
            np.nan
        )
        
        df["target_nicu"] = np.where(
            df["admit_nicu"].notna(),
            pd.to_numeric(df["admit_nicu"], errors="coerce"),
            np.nan
        )
        
        df["target_lbw"] = np.where(
            (df["birth_weight_g"] >= 227) & (df["birth_weight_g"] <= 8165),
            (df["birth_weight_g"] < 2500).astype(float),
            np.nan
        )
    else:
        df["target_preterm"] = pd.to_numeric(df["target_preterm"], errors="coerce")
        df["target_nicu"] = pd.to_numeric(df["target_nicu"], errors="coerce")
        df["target_lbw"] = pd.to_numeric(df["target_lbw"], errors="coerce")
    
    # Ensure primary population base masks are numeric
    df["is_us_resident"] = pd.to_numeric(df["is_us_resident"], errors="coerce")
    df["is_singleton"] = pd.to_numeric(df["is_singleton"], errors="coerce")
    
    # 3. Dynamically download models
    print("Fetching models from canonical task...")
    canonical_task = Task.get_task(task_id="4f9d98dd39f3438181f48b256b23bd94")
    
    # Using get_local_copy() on the artifact
    models_dir = Path(canonical_task.artifacts["tae_canonical_models"].get_local_copy())
    print(f"Models downloaded to {models_dir}")
    
    targets = {
        "target_preterm": 0.124521,
        "target_nicu": 0.119584,
        "target_lbw": 0.118980
    }
    
    results = []
    
    for target, threshold in targets.items():
        print(f"\nProcessing {target}...")
        
        # 4 & 5. Load model and apply EXACT preprocessing logic
        model_path = models_dir / f"{target}__landmark_strict__lightgbm.joblib"
        data = joblib.load(model_path)
        preprocessor = data["preprocessor"]
        model = data["model"]
        features = data["features"]
        
        # Race is categorical semantically but its stored codes must be numeric.
        for f in features:
            df[f] = pd.to_numeric(df[f], errors="raise")
        
        # Filter for "all-record upper bound"
        # primary population: is_us_resident==1 & is_singleton==1 & target is known
        mask = (
            (df["is_us_resident"] == 1) & 
            (df["is_singleton"] == 1) & 
            df[target].notna()
        )
        df_eval = df[mask].copy()
        
        X_eval = df_eval[features]
        y_true = df_eval[target].astype(int).values
        
        print(f"Applying preprocessing to {len(X_eval)} records...")
        X_trans = preprocessor.transform(X_eval).astype(np.float32)
        
        print("Generating predictions...")
        y_pred_proba = model.predict_proba(X_trans)[:, 1]
        
        # Step 3: Metric Calculation
        e_all = y_true.sum()
        n_all = len(y_true)
        selected = (y_pred_proba >= threshold)
        selected_n = selected.sum()
        tp = y_true[selected].sum()
        
        precision = tp / selected_n if selected_n > 0 else 0.0
        capture = tp / e_all if e_all > 0 else 0.0
        
        print(f"  E_all: {e_all}")
        print(f"  N_all: {n_all}")
        print(f"  selected_n: {selected_n}")
        print(f"  TP: {tp}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Capture: {capture:.4f}")
        
        results.append({
            "target": target,
            "threshold": threshold,
            "E_all": int(e_all),
            "N_all": int(n_all),
            "selected_n": int(selected_n),
            "TP": int(tp),
            "Precision": precision,
            "Population Event Capture": capture
        })
        
    # Step 4: Save results
    out_dir = Path("results/tae_2026_replication")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "replication_2024_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    
    # Save to ClearML
    task.upload_artifact("replication_2024_results", str(out_path))
    task.close()
    print("Done!")

if __name__ == "__main__":
    main()
