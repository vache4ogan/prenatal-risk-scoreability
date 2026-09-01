import csv
import sys
from pathlib import Path

def harmonize(input_path, output_path):
    print(f"Reading from {input_path}")
    print(f"Writing to {output_path}")

    # Field coordinates (1-based, inclusive)
    fields = {
        'residence_status': (104, 104),
        'plurality': (454, 454),
        'prenatal_care_month': (224, 225),
        'gestation_oe_weeks': (499, 500),
        'admit_nicu': (519, 519),
        'birth_weight_g': (504, 507),
        'mother_race': (105, 106),
        'mother_height_inches': (280, 281),
        'mother_bmi': (283, 286),
        'mother_weight_pre': (292, 294),
        'prior_live_births': (171, 172),
        'prior_dead_births': (173, 174),
        'prior_terminations': (175, 176),
        'diab_pre': (313, 313),
        'hyper_pre': (315, 315),
    }

    # Final columns required
    output_columns = [
        'is_us_resident', 'is_singleton', 'prenatal_care_month',
        'gestation_oe_weeks', 'admit_nicu', 'birth_weight_g',
        'mother_race', 'mother_height_inches', 'mother_bmi',
        'mother_weight_pre', 'prior_live_births', 'prior_dead_births',
        'prior_terminations', 'diab_pre', 'hyper_pre'
    ]

    missing_map = {
        'prenatal_care_month': {'99', ''},
        'gestation_oe_weeks': {'99', ''},
        'admit_nicu': {'U', ''},
        'birth_weight_g': {'9999', ''},
        'mother_race': {'99', ''},
        'mother_height_inches': {'99', ''},
        'mother_bmi': {'99.9', ''},
        'mother_weight_pre': {'999', ''},
        'prior_live_births': {'99', ''},
        'prior_dead_births': {'99', ''},
        'prior_terminations': {'99', ''},
        'diab_pre': {'U', ''},
        'hyper_pre': {'U', ''},
    }

    processed = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8', newline='') as fout:
         
        writer = csv.DictWriter(fout, fieldnames=output_columns)
        writer.writeheader()

        for line in fin:
            line = line.rstrip("\r\n")
            if not line:
                continue

            row = {}
            for field, (start, end) in fields.items():
                if len(line) >= end:
                    row[field] = line[start-1:end].strip()
                else:
                    row[field] = ''
            
            out = {}
            
            # is_us_resident: 1, 2, 3 -> 1, 4 -> 0
            if row['residence_status'] in {'1', '2', '3'}:
                out['is_us_resident'] = '1'
            elif row['residence_status'] == '4':
                out['is_us_resident'] = '0'
            else:
                out['is_us_resident'] = ''

            # is_singleton: 1 -> 1, else -> 0
            if row['plurality'] == '1':
                out['is_singleton'] = '1'
            elif row['plurality'] in {'2', '3', '4', '5'}:
                out['is_singleton'] = '0'
            else:
                out['is_singleton'] = ''

            # Other variables
            for col in output_columns:
                if col in ['is_us_resident', 'is_singleton']:
                    continue
                val = row[col]
                # Map missing values to empty string (NaN for CSV)
                if val in missing_map.get(col, set()):
                    out[col] = ''
                elif val == 'Y':
                    out[col] = '1'
                elif val == 'N':
                    out[col] = '0'
                else:
                    # Keep raw values, taking care of leading zeros if needed
                    # but typically numeric strings are fine
                    # '00' in prenatal_care_month becomes '0'
                    if col == 'prenatal_care_month' and val.isdigit():
                        out[col] = str(int(val))
                    else:
                        out[col] = val
                        
            writer.writerow(out)
            processed += 1
            if processed % 500000 == 0:
                print(f"Processed {processed} rows...")

    print(f"Done. Processed total {processed} rows.")

if __name__ == '__main__':
    in_file = 'data/cdc_2024/Nat2024PublicUS.txt'
    out_file = 'data/cdc_2024/harmonized_cdc_2024.csv'
    # Use alternative input if provided via command line
    if len(sys.argv) > 1:
        in_file = sys.argv[1]
    
    # Let's ensure the file exists before running
    if not Path(in_file).exists():
        # Maybe it's named Nat2024us.txt
        alt_file = 'data/cdc_2024/Nat2024us.txt'
        if Path(alt_file).exists():
            in_file = alt_file
        else:
            print(f"Error: Could not find raw file {in_file} or {alt_file}")
            sys.exit(1)
            
    harmonize(in_file, out_file)
