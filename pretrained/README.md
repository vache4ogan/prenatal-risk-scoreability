# Frozen research score instruments

Three LightGBM scores trained on CDC/NCHS 2022 natality data for the accepted
paper **Same Top Fraction, Different Workload: Auditing Eligibility and Capacity
in Clinical Risk Evaluation** (see the repository's `CITATION.cff`):

| Directory | Outcome |
| --- | --- |
| `target_preterm` | Preterm birth |
| `target_nicu` | NICU admission |
| `target_lbw` | Low birth weight |

Training population: target-specific U.S.-resident singleton births. Each score
has 350 trees and uses the frozen `landmark_strict` nine-feature specification.
These checkpoints support retrospective statistical evaluation. Clinical use
would require prospective validation of inputs, performance, and interventions.

Each target contains `model.txt` in the official LightGBM native format and
`preprocessing.json` with fitted summary statistics, category codes, feature
order, and frozen CDC 2022 calibration thresholds (fractions 0.05 and 0.1;
comparator `>=`). There are no individual records, individual predictions,
pickles, or private filesystem paths. No training or recalibration was performed
for this export. `manifest.json` records source bundle hashes and export versions;
`SHA256SUMS` covers every other file in this directory tree.

## Loading

```python
from scripts.experiments.native_model import load_native_target

bundle = load_native_target("pretrained", "target_preterm")
matrix = bundle["preprocessor"].transform(frame[bundle["features"]])
probability = bundle["model"].predict_proba(matrix, num_threads=1)[:, 1]
```

`frame` must use the canonical numeric feature representation. In particular,
`mother_race` uses numeric MRACE31 codes. NaN is imputed
using frozen numeric medians or the categorical mode. Unknown numeric categories
produce a zero one-hot block. String codes raise an error. Numeric
imputation and scaling precede categorical concatenation; the final CSR matrix
is cast to float32. Input columns can be reordered and extra columns are ignored.
Use `requirements.txt` for the parity-tested environment.

The loader also exposes `load_native_bundle(target_directory)` and
`find_native_bundle(models_root, target)`. It uses LightGBM's
[official native model reader](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html),
not a custom tree parser. Tests use artificial inputs only. Trusted-joblib parity
tests are opt-in via `TRUSTED_FROZEN_MODELS`; never point this at untrusted pickles.

## License and source data

Code, exported weights, and preprocessing parameters are released under the MIT
License, copyright (c) 2026 Vache Oganisyan, Dmitry Lvov, Ilya Pershin. See
[`LICENSE`](LICENSE). This permission does **not** license CDC/NCHS source data
under MIT. Source data remain subject to separate
[NCHS data-use terms](https://www.cdc.gov/nchs/policy/data-user-agreement.html).
Obtain and use those data separately; no individual-level CDC data are distributed
with these artifacts.
