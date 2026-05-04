import json
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

FEATURES = ['product_price', 'delivery_days', 'customer_rating', 'is_first_order']
TARGET   = 'return_probability_pct'

MODEL_MAP = {
    "RandomForestRegressor":     RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

# ─── 1. Load Data & Count Rows ────────────────────────────────────────────────
df_original = pd.read_csv('data/training_data.csv')
df_new      = pd.read_csv('data/new_data.csv')

original_data_rows = len(df_original)   # 25
new_data_rows      = len(df_new)        # 20

print(f"[Phase 4] Original rows : {original_data_rows}")
print(f"[Phase 4] New rows      : {new_data_rows}")

# ─── 2. Combine Dataset ───────────────────────────────────────────────────────
df_combined = pd.concat([df_original, df_new], ignore_index=True)

# Shuffle with a fixed seed so it's reproducible, not random chaos
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined_data_rows = len(df_combined)   # 45
print(f"[Phase 4] Combined rows : {combined_data_rows}")

# ─── 3. Load Phase 3 info & pull Champion from MLflow Registry ───────────────
with open('results/step3_s6.json', 'r') as f:
    step3 = json.load(f)

with open('results/step1_s1.json', 'r') as f:
    step1 = json.load(f)

with open('results/step2_s2.json', 'r') as f:
    step2 = json.load(f)

registered_name = step3['registered_model_name']
best_model_name = step1['best_model']
best_params     = step2['best_params']

print(f"[Phase 4] Champion registry : {registered_name}")
print(f"[Phase 4] Model type        : {best_model_name}")
print(f"[Phase 4] Best params       : {best_params}")

# Load champion model from MLflow Model Registry (latest version)
champion_uri   = f"models:/{registered_name}/latest"
champion_model = mlflow.sklearn.load_model(champion_uri)
print(f"[Phase 4] Champion loaded from registry ✓")

# ─── 4. Prepare Retrained Model ───────────────────────────────────────────────
ModelClass = MODEL_MAP[best_model_name]

if ModelClass == RandomForestRegressor:
    model_params = {k: v for k, v in best_params.items() if k != 'learning_rate'}
else:
    model_params = best_params.copy()

model_params['random_state'] = 42

# ─── 5. ONE combined split — both models evaluated on the same test set ────────
X_combined = df_combined[FEATURES]
y_combined  = df_combined[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# ── Retrained model: train on combined training split ────────────────────────
retrained_model = ModelClass(**model_params)
retrained_model.fit(X_train, y_train)

# ─── 6. Evaluate BOTH on the SAME test set ────────────────────────────────────
y_pred_champion  = champion_model.predict(X_test)
y_pred_retrained = retrained_model.predict(X_test)

champion_rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred_champion)))
retrained_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_retrained)))
improvement    = float(champion_rmse - retrained_rmse)

print(f"\n[Phase 4] Champion RMSE  : {champion_rmse:.4f}")
print(f"[Phase 4] Retrained RMSE : {retrained_rmse:.4f}")
print(f"[Phase 4] Improvement    : {improvement:.4f}")

# ─── 7. Decision Logic ────────────────────────────────────────────────────────
MIN_THRESHOLD = 1.0
action = "promoted" if improvement >= MIN_THRESHOLD else "kept_champion"
print(f"[Phase 4] Decision       : {action}")

# ─── Log retraining run to MLflow ─────────────────────────────────────────────
mlflow.set_experiment("cartwave-return-probability-pct")

with mlflow.start_run(run_name="retrain-cartwave") as run:
    mlflow.set_tag("domain", "e-commerce")
    mlflow.set_tag("phase", "retraining")

    mlflow.log_params(best_params)
    mlflow.log_metrics({
        "champion_rmse":  champion_rmse,
        "retrained_rmse": retrained_rmse,
        "improvement":    improvement,
    })
    mlflow.log_param("action", action)
    mlflow.log_param("combined_rows", combined_data_rows)

    # If promoted, log and register the new version
    if action == "promoted":
        mlflow.sklearn.log_model(
            sk_model=retrained_model,
            artifact_path="model",
            registered_model_name=registered_name,
        )
        print(f"[Phase 4] New model version registered under '{registered_name}' ✓")
    else:
        mlflow.sklearn.log_model(
            sk_model=retrained_model,
            artifact_path="model",
        )
        print(f"[Phase 4] Champion retained — retrained model logged but not promoted ✓")

# ─── 8. Save Output ───────────────────────────────────────────────────────────
output = {
    "original_data_rows":       original_data_rows,
    "new_data_rows":            new_data_rows,
    "combined_data_rows":       combined_data_rows,
    "champion_rmse":            champion_rmse,
    "retrained_rmse":           retrained_rmse,
    "improvement":              improvement,
    "min_improvement_threshold": MIN_THRESHOLD,
    "action":                   action,
    "comparison_metric":        "rmse",
}

os.makedirs('results', exist_ok=True)
with open('results/step4_s8.json', 'w') as f:
    json.dump(output, f, indent=4)

print("\n✅ Phase 4 complete!")
print(f"   Action  : {action}")
print(f"   Saved   → results/step4_s8.json")
