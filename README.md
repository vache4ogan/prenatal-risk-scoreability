# TAE 2026: Selective Availability (NeurIPS Submission)

This repository contains the official codebase and reproducibility artifacts for the paper evaluating **Selective Availability** and structural biases in clinical ML deployment, using the CDC Natality dataset (2022-2024).

## 📊 Repository Structure

* **`configs/`** - Locked configurations and feature sets for ClearML.
* **`scripts/preprocessing/`** - Data harmonization and parsing scripts.
* **`scripts/experiments/`** - Core ML pipeline (LightGBM training, Shapley decomposition, 2024 Locked Temporal Replication, and Ablation studies).
* **`paper/`** - LaTeX source files and generated figures.

## 🚀 How to Reproduce

Our pipeline relies on ClearML for strict lineage and artifact tracking.

### 1. Install Requirements
`bash
pip install -r pyproject.toml
`

### 2. Data Preparation
Due to DUA restrictions, raw CDC data is not provided. Download the Public Use Files (2022-2024) from the [NCHS website](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm) and place them in the `data/` directory. Run the harmonization scripts in `scripts/preprocessing/`.

### 3. Run Core Experiments
`bash
python scripts/experiments/run_tae_canonical_clearml.py
python scripts/experiments/run_tae_shapley_fixed_b_bootstrap_clearml.py
`

### 4. Run 2024 Locked Temporal Replication
`bash
python scripts/experiments/run_2024_locked_replication.py
`

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
