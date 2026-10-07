"""Extract the frozen 15-column CDC 2024 evaluation schema from US public-use TXT."""

import argparse
import csv
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    harmonize(args.input, args.output)


if __name__ == "__main__":
    main()
