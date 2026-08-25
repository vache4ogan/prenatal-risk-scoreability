import pandas as pd
import sys

def validate(csv_path):
    print(f"Validating {csv_path}...")
    
    expected_columns = {
        'is_us_resident', 'is_singleton', 'prenatal_care_month',
        'gestation_oe_weeks', 'admit_nicu', 'birth_weight_g',
        'mother_race', 'mother_height_inches', 'mother_bmi',
        'mother_weight_pre', 'prior_live_births', 'prior_dead_births',
        'prior_terminations', 'diab_pre', 'hyper_pre'
    }

    # Use chunking to avoid OOM
    chunks = pd.read_csv(csv_path, chunksize=100000, dtype=str)
    
    total_rows = 0
    missing_counts = {col: 0 for col in expected_columns}
    has_99_precare = False
    
    first_chunk = True
    
    for chunk in chunks:
        if first_chunk:
            actual_columns = set(chunk.columns)
            if actual_columns != expected_columns:
                print(f"ERROR: Column mismatch.")
                print(f"Missing expected: {expected_columns - actual_columns}")
                print(f"Unexpected extra: {actual_columns - expected_columns}")
                sys.exit(1)
            else:
                print("PASS: Exactly 15 required columns are present.")
            first_chunk = False
            
        total_rows += len(chunk)
        
        # Check prenatal_care_month for '99'
        if (chunk['prenatal_care_month'] == '99').any() or (chunk['prenatal_care_month'] == 99).any():
            has_99_precare = True
            
        # Count missingness
        for col in expected_columns:
            missing_counts[col] += chunk[col].isna().sum()

    if has_99_precare:
        print("ERROR: Found '99' in prenatal_care_month!")
        sys.exit(1)
    else:
        print("PASS: prenatal_care_month has no '99' values.")

    print("\n--- Summary Statistics ---")
    print(f"Total Rows: {total_rows:,}")
    print("Missingness:")
    for col in sorted(expected_columns):
        pct = (missing_counts[col] / total_rows) * 100 if total_rows > 0 else 0
        print(f"  {col:22s}: {missing_counts[col]:>10,} missing ({pct:>5.2f}%)")

if __name__ == '__main__':
    validate('data/cdc_2024/harmonized_cdc_2024.csv')
