"""
Phase 5 — Validation & Sanity Check
Covers: file structure, JSON integrity, MLflow artifacts,
        logical consistency, and reproducibility.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
WARN  = "⚠️  WARN"

failures = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg    = f"  {status} | {label}"
    if detail:
        msg += f" → {detail}"
    print(msg)
    if not condition:
        failures.append(label)
    return condition

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. File Structure
# ─────────────────────────────────────────────────────────────────────────────
section("1 · File Structure")

BASE = Path(".")
required_dirs = ["data", "src", "models", "results"]
for d in required_dirs:
    check(f"Directory exists: {d}/", (BASE / d).is_dir())

required_data = ["data/training_data.csv", "data/new_data.csv"]
for f in required_data:
    check(f"Data file exists: {f}", (BASE / f).is_file())

required_src = ["src/train.py", "src/tune.py", "src/register_model.py", "src/retrain.py"]
for f in required_src:
    check(f"Source file exists: {f}", (BASE / f).is_file())

# ─────────────────────────────────────────────────────────────────────────────
# 2. JSON Files — existence, format, keys, types
# ─────────────────────────────────────────────────────────────────────────────
section("2 · JSON File Validation")

JSON_SPECS = {
    "results/step1_s1.json": {
        "required_keys": ["experiment_name", "models", "best_model",
                          "best_metric_name", "best_metric_value", "best_rmse"],
        "numeric_keys":  ["best_metric_value", "best_rmse"],
    },
    "results/step2_s2.json": {
        "required_keys": ["search_type", "n_folds", "total_trials",
                          "best_params", "best_mae", "best_cv_mae",
                          "parent_run_name"],
        "numeric_keys":  ["n_folds", "total_trials", "best_mae", "best_cv_mae"],
    },
    "results/step3_s6.json": {
        "required_keys": ["registered_model_name", "version", "run_id",
                          "source_metric", "source_metric_value"],
        "numeric_keys":  ["version", "source_metric_value"],
    },
    "results/step4_s8.json": {
        "required_keys": ["original_data_rows", "new_data_rows",
                          "combined_data_rows", "champion_rmse",
                          "retrained_rmse", "improvement",
                          "min_improvement_threshold", "action",
                          "comparison_metric"],
        "numeric_keys":  ["original_data_rows", "new_data_rows",
                          "combined_data_rows", "champion_rmse",
                          "retrained_rmse", "improvement",
                          "min_improvement_threshold"],
    },
}

loaded = {}
for path, spec in JSON_SPECS.items():
    exists = (BASE / path).is_file()
    check(f"File exists: {path}", exists)
    if not exists:
        continue

    try:
        with open(path) as f:
            data = json.load(f)
        check(f"Valid JSON: {path}", True)
        loaded[path] = data
    except json.JSONDecodeError as e:
        check(f"Valid JSON: {path}", False, str(e))
        continue

    for key in spec["required_keys"]:
        check(f"  Key '{key}' present in {path}", key in data)

    for key in spec.get("numeric_keys", []):
        if key in data:
            check(f"  Key '{key}' is numeric", isinstance(data[key], (int, float)),
                  f"got {type(data[key]).__name__}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MLflow Artifact Validation
# ─────────────────────────────────────────────────────────────────────────────
section("3 · MLflow Artifact Validation")

client = mlflow.MlflowClient()

# 3a — Experiment exists
EXPERIMENT_NAME = "cartwave-return-probability-pct"
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
check(f"Experiment exists: '{EXPERIMENT_NAME}'", exp is not None)

if exp:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=200,
    )

    # 3b — Parent tuning run
    tuning_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == "tuning-cartwave"]
    check("Parent run 'tuning-cartwave' exists", len(tuning_runs) > 0)

    if tuning_runs:
        parent_run_id = tuning_runs[0].info.run_id
        nested_runs   = [r for r in runs
                         if r.data.tags.get("mlflow.parentRunId") == parent_run_id]
        check("18 nested tuning trials", len(nested_runs) == 18,
              f"found {len(nested_runs)}")

    # 3c — Final model run
    final_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == "final-model-cartwave"]
    check("Final run 'final-model-cartwave' exists", len(final_runs) > 0)

    # 3d — Retrain run
    retrain_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == "retrain-cartwave"]
    check("Retraining run 'retrain-cartwave' exists", len(retrain_runs) > 0)

# 3e — Registered model
REGISTERED_NAME = "cartwave-return-probability-pct-predictor"
try:
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    check(f"Registered model exists: '{REGISTERED_NAME}'", len(versions) > 0)
    check("At least 1 version registered", len(versions) >= 1,
          f"found {len(versions)} version(s)")
except Exception as e:
    check(f"Registered model exists: '{REGISTERED_NAME}'", False, str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Logical Consistency Checks
# ─────────────────────────────────────────────────────────────────────────────
section("4 · Logical Consistency Checks")

s1 = loaded.get("results/step1_s1.json", {})
s2 = loaded.get("results/step2_s2.json", {})
s3 = loaded.get("results/step3_s6.json", {})
s4 = loaded.get("results/step4_s8.json", {})

# 4a — Model type consistency
VALID_MODELS = {"RandomForestRegressor", "GradientBoostingRegressor"}
best_model = s1.get("best_model", "")
check("best_model is a known model type", best_model in VALID_MODELS, best_model)

# 4b — Phase 2 used 18 trials
check("Phase 2 total_trials == 18", s2.get("total_trials") == 18,
      str(s2.get("total_trials")))

# 4c — Phase 2 used 3 folds
check("Phase 2 n_folds == 3", s2.get("n_folds") == 3,
      str(s2.get("n_folds")))

# 4d — RMSE values are realistic (non-zero, non-NaN, positive)
for path, key in [
    ("results/step1_s1.json", "best_rmse"),
    ("results/step2_s2.json", "best_mae"),
    ("results/step3_s6.json", "source_metric_value"),
    ("results/step4_s8.json", "champion_rmse"),
    ("results/step4_s8.json", "retrained_rmse"),
]:
    data = loaded.get(path, {})
    val  = data.get(key, None)
    if val is not None:
        realistic = isinstance(val, (int, float)) and val > 0 and not np.isnan(val)
        check(f"Realistic value for {path} → {key}", realistic, str(val))

# 4e — improvement = champion_rmse - retrained_rmse (not reversed)
if s4:
    expected_improvement = round(s4.get("champion_rmse", 0) - s4.get("retrained_rmse", 0), 6)
    stored_improvement   = round(s4.get("improvement", 0), 6)
    check(
        "improvement = champion_rmse - retrained_rmse (not reversed)",
        abs(expected_improvement - stored_improvement) < 1e-3,
        f"expected≈{expected_improvement:.4f}, stored={stored_improvement:.4f}",
    )

# 4f — action follows the threshold rule
if s4:
    action      = s4.get("action", "")
    improvement = s4.get("improvement", 0)
    threshold   = s4.get("min_improvement_threshold", 1.0)
    expected    = "promoted" if improvement >= threshold else "kept_champion"
    check("action follows threshold rule", action == expected,
          f"improvement={improvement:.4f}, threshold={threshold}, action={action}")

# 4g — row counts add up
if s4:
    orig = s4.get("original_data_rows", 0)
    new  = s4.get("new_data_rows", 0)
    comb = s4.get("combined_data_rows", 0)
    check("combined_data_rows == original + new", orig + new == comb,
          f"{orig} + {new} = {orig+new}, stored={comb}")

# 4h — Phase 3 registered model name matches Phase 4
p3_name = s3.get("registered_model_name", "")
check("Registered model name consistent across Phase 3 & Phase 4",
      p3_name == REGISTERED_NAME, p3_name)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Reproducibility Check (re-run core training logic)
# ─────────────────────────────────────────────────────────────────────────────
section("5 · Reproducibility Check")

MODEL_MAP = {
    "RandomForestRegressor":     RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}
FEATURES = ['product_price', 'delivery_days', 'customer_rating', 'is_first_order']
TARGET   = 'return_probability_pct'
TOLERANCE = 1e-4   # floating point tolerance

try:
    # ── Phase 1 reproduction ─────────────────────────────────────────────────
    df  = pd.read_csv('data/training_data.csv')
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, Cls in MODEL_MAP.items():
        m = Cls(random_state=42)
        m.fit(X_train, y_train)
        rmse = float(np.sqrt(mean_squared_error(y_test, m.predict(X_test))))
        stored = s1.get("models", {}).get(name, {}).get("RMSE", None)
        if stored is not None:
            close = abs(rmse - stored) < TOLERANCE
            check(f"Phase 1 RMSE reproducible for {name}",
                  close, f"recomputed={rmse:.6f}, stored={stored:.6f}")
        else:
            print(f"  {WARN} | No stored RMSE for {name} to compare")

    # ── Phase 2 best params reproduction ─────────────────────────────────────
    best_params  = s2.get("best_params", {})
    ModelCls     = MODEL_MAP.get(best_model)
    if ModelCls:
        mp = {k: v for k, v in best_params.items() if k != 'learning_rate'} \
             if ModelCls == RandomForestRegressor else best_params.copy()
        mp["random_state"] = 42
        m2 = ModelCls(**mp)
        m2.fit(X_train, y_train)
        rmse2 = float(np.sqrt(mean_squared_error(y_test, m2.predict(X_test))))
        stored_rmse3 = s3.get("source_metric_value", None)
        if stored_rmse3 is not None:
            close2 = abs(rmse2 - stored_rmse3) < TOLERANCE
            check("Phase 3 RMSE reproducible (best params on original split)",
                  close2, f"recomputed={rmse2:.6f}, stored={stored_rmse3:.6f}")

    # ── Phase 4 combined-data reproduction ───────────────────────────────────
    df_new  = pd.read_csv('data/new_data.csv')
    df_comb = pd.concat([df, df_new], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    Xc, yc  = df_comb[FEATURES], df_comb[TARGET]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)

    m4 = ModelCls(**mp)
    m4.fit(Xc_tr, yc_tr)
    rmse4 = float(np.sqrt(mean_squared_error(yc_te, m4.predict(Xc_te))))
    stored_r4 = s4.get("retrained_rmse", None)
    if stored_r4 is not None:
        close4 = abs(rmse4 - stored_r4) < TOLERANCE
        check("Phase 4 retrained RMSE reproducible",
              close4, f"recomputed={rmse4:.6f}, stored={stored_r4:.6f}")

except Exception as exc:
    check("Reproducibility checks ran without error", False, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("Summary")

total_checks = sum(1 for line in open(__file__) if "check(" in line)
print(f"\n  Failures : {len(failures)}")
if failures:
    for f in failures:
        print(f"    ✗ {f}")
else:
    print("  All checks passed — pipeline is solid 🎉")

sys.exit(1 if failures else 0)
