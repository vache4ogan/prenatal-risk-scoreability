"""Local frozen-model four-cell audit. No training or ClearML writes.

Use the numeric category representation from canonical CDC 2022/2023 fitting.
The archived 2024 threshold script instead read race codes as strings, which
silently disabled their numeric one-hot categories. This runner rejects that
representation and records both input and model hashes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_tae_shapley_fixed_b_bootstrap_clearml as audit


TARGETS = ["target_preterm", "target_nicu", "target_lbw"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    numeric = set(audit.LANDMARK_STRICT_FEATURES) | {
        "is_us_resident", "is_singleton", "prenatal_care_month",
        "gestation_oe_weeks", "birth_weight_g", "admit_nicu", *TARGETS,
    }
    for column in numeric & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if "target_preterm" not in frame:
        weeks = frame["gestation_oe_weeks"]
        frame["target_preterm"] = (weeks < 37).astype(float).where(weeks.between(17, 47))
    if "target_lbw" not in frame:
        weight = frame["birth_weight_g"]
        frame["target_lbw"] = (weight < 2500).astype(float).where(weight.between(227, 8165))
    if "target_nicu" not in frame:
        frame["target_nicu"] = frame["admit_nicu"]
    for column in ["is_us_resident", "is_singleton", *TARGETS]:
        if not set(frame[column].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Nonbinary column: {column}")
    audit.validate_care_month(frame.prenatal_care_month.to_numpy(float), "temporal audit")
    race = frame.mother_race.dropna()
    if not race.between(1, 31).all() or not (race == race.round()).all():
        raise ValueError("Expected numeric CDC MRACE31 codes 1..31 or missing")
    return frame


def bootstrap(y, scores, care, lookup, *, repeats, seed, target, year):
    """Exact multinomial resampling; shared counts across all four cells and q."""
    n = len(y)
    early = audit.early_entry_mask(care)
    ranks = [audit.rank_eligible_indices(scores, early),
             audit.rank_eligible_indices(scores, np.ones(n, dtype=bool))]
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(repeats):
        sampled = rng.integers(0, n, size=n, dtype=np.int32)
        counts = np.bincount(sampled, minlength=n).astype(np.int32)
        del sampled
        events = int(counts[y == 1].sum(dtype=np.int64))
        ranked = []
        for order in ranks:
            w = counts[order]
            ranked.append((np.cumsum(w, dtype=np.int64),
                           np.cumsum(w * y[order], dtype=np.int64), y[order]))
        for q, point in lookup.items():
            b0, b1 = audit.fixed_observed_bootstrap_budgets(lookup, q)
            tp = {}
            for label, row, budget in [("C00", 0, b0), ("C01", 0, b1),
                                       ("C10", 1, b0), ("C11", 1, b1)]:
                cc, ce, yy = ranked[row]
                tp[label] = audit.selected_tp_from_cumulative(
                    cumulative_count=cc, cumulative_events=ce,
                    ordered_y=yy, budget=budget)
            comp = audit.component_values(*(tp[k] / events for k in ["C00", "C10", "C01", "C11"]))
            audit.assert_component_identities(comp, f"{target}/{q}/{rep}")
            rows.append({"year": year, "target": target, "nominal_fraction": q,
                         "replicate": rep, "full_event_n": events,
                         "B_early": b0, "B_all": b1,
                         **{f"{k}_tp": v for k, v in tp.items()}, **comp})
        if (rep + 1) % 100 == 0:
            print(f"{target}: bootstrap {rep+1}/{repeats}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=[2023, 2024], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-scores", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if args.bootstrap_replicates < 0:
        raise ValueError("Negative replicate count")
    observed_hash = sha256(args.csv)
    if observed_hash != args.expected_sha256:
        raise ValueError("Input CSV hash mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "year": args.year, "input_sha256": observed_hash,
                "script_sha256": sha256(Path(__file__)),
                "audit_module_sha256": sha256(Path(audit.__file__)),
                "bootstrap_replicates": args.bootstrap_replicates, "seed": args.seed,
                "target_seed_stride": 1000, "fractions": [0.05, 0.1],
                "model_fitting": False, "category_encoding": "numeric MRACE31, canonical representation",
                "python": platform.python_version(), "models": {},
                "versions": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scikit-learn", "lightgbm", "joblib"]}}
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    required = set(audit.LANDMARK_STRICT_FEATURES) | {"is_us_resident", "is_singleton", "prenatal_care_month"}
    extra = set(TARGETS) if args.year == 2023 else {"gestation_oe_weeks", "birth_weight_g", "admit_nicu"}
    print("Reading and validating numeric inputs", flush=True)
    frame = normalize_frame(pd.read_csv(args.csv, usecols=sorted(required | extra), low_memory=False))
    threshold_frame = pd.read_csv(args.thresholds)
    cells, decompositions, replicas, threshold_rows = [], [], [], []
    population = audit.resident_singleton_mask(frame)
    for index, target in enumerate(TARGETS):
        bundle = audit.load_frozen_model_bundle(args.models, target)
        manifest["models"][target] = bundle["_model_sha256"]
        mask = population & frame[target].notna().to_numpy()
        y = frame.loc[mask, target].to_numpy(np.int8)
        care = frame.loc[mask, "prenatal_care_month"].to_numpy(float)
        print(f"Scoring {target}: N={len(y)}, E={int(y.sum())}", flush=True)
        x = audit.consistent_matrix(bundle["preprocessor"].transform(frame.loc[mask, bundle["features"]]))
        scores = bundle["model"].predict_proba(x, num_threads=args.threads)[:, 1]
        if not np.isfinite(scores).all():
            raise ValueError("Invalid score vector")
        del x
        gc.collect()
        if args.save_scores:
            score_dir = args.output_dir / "predictions"
            score_dir.mkdir(exist_ok=True)
            np.savez_compressed(score_dir / f"{target}.npz", y=y, score=scores, care_month=care)
        target_cells, target_decomp, lookup = audit.four_cell_point_estimates(
            target=target, y=y, probabilities=scores, care_month=care)
        for row in target_cells + target_decomp:
            row["evaluation"] = f"temporal_evaluation_cohort_{args.year}"
            row["year"] = args.year
        for row in target_cells:
            row["eligible_recall"] = row["true_positive_n"] / row["candidate_event_n"]
            row["event_ceiling"] = row["candidate_event_n"] / row["full_population_event_n"]
        cells.extend(target_cells)
        decompositions.extend(target_decomp)
        for row in threshold_frame.loc[threshold_frame.target.eq(target)].to_dict("records"):
            threshold = float(row["threshold"])
            q = float(row["nominal_fraction"])
            for precision, cutoff in [("exact", threshold), ("rounded_6dp", round(threshold, 6))]:
                selected = scores >= cutoff
                selected_n = int(selected.sum())
                tp = int(y[selected].sum())
                threshold_rows.append({"year": args.year, "target": target,
                                       "nominal_fraction": q, "threshold_precision": precision,
                                       "threshold": cutoff, "N_all": len(y), "E_all": int(y.sum()),
                                       "selected_n": selected_n, "TP": tp,
                                       "selection_rate": selected_n / len(y),
                                       "population_event_capture": tp / int(y.sum()),
                                       "precision": tp / selected_n})
        replicas.extend(bootstrap(y, scores, care, lookup, repeats=args.bootstrap_replicates,
                                  seed=args.seed + index * 1000, target=target, year=args.year))
        pd.DataFrame(cells).to_csv(args.output_dir / "four_cells.csv", index=False)
        pd.DataFrame(decompositions).to_csv(args.output_dir / "decomposition.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(args.output_dir / "threshold_transport.csv", index=False)
        if replicas:
            pd.DataFrame(replicas).to_csv(args.output_dir / "bootstrap_replicates.csv", index=False)
        del bundle, scores, y, care
        gc.collect()
    points = pd.DataFrame(decompositions)
    summary = []
    for row in points.to_dict("records"):
        row["capacity_share"] = row["shapley_capacity"] / row["total_topq_contrast"]
        if replicas:
            rep = pd.DataFrame(replicas)
            rep = rep.loc[rep.target.eq(row["target"]) & rep.nominal_fraction.eq(row["nominal_fraction"])].copy()
            rep["capacity_share"] = rep.shapley_capacity / rep.total_topq_contrast
            for key in ["shapley_capacity", "shapley_availability", "interaction", "total_topq_contrast",
                        "availability_at_early_budget", "ordered_availability_at_all_budget", "capacity_share"]:
                row[f"{key}_lower"], row[f"{key}_upper"] = rep[key].quantile([0.025, 0.975])
        summary.append(row)
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)
    manifest.update(status="PASS", elapsed_seconds=time.time() - started,
                    hashes={p.name: sha256(p) for p in args.output_dir.glob("*.csv")})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"PASS: {args.year}, 24 cells, 6 contrasts; elapsed={manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
