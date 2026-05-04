import json
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ─── 1. Load Best Model Info ──────────────────────────────────────────────────
with open('results/step1_s1.json', 'r') as f:
    step1 = json.load(f)

with open('results/step2_s2.json', 'r') as f:
    step2 = json.load(f)

best_model_name = step1['best_model']
best_params     = step2['best_params']

print(f"[Phase 3] Model type  : {best_model_name}")
print(f"[Phase 3] Best params : {best_params}")

MODEL_MAP = {
    "RandomForestRegressor":    RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

if best_model_name not in MODEL_MAP:
    raise ValueError(f"Unknown model type: {best_model_name}")

ModelClass = MODEL_MAP[best_model_name]

# ─── 2. Retrain Final Model ───────────────────────────────────────────────────
df = pd.read_csv('data/training_data.csv')

X = df[['product_price', 'delivery_days', 'customer_rating', 'is_first_order']]
y = df['return_probability_pct']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model params — strip keys unsupported by the chosen model class
if ModelClass == RandomForestRegressor:
    model_params = {k: v for k, v in best_params.items() if k != 'learning_rate'}
else:
    model_params = best_params.copy()

model_params['random_state'] = 42

model = ModelClass(**model_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = r2_score(y_test, y_pred)

print(f"[Phase 3] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# ─── 3. Log Final Model to MLflow ────────────────────────────────────────────
REGISTERED_NAME = "cartwave-return-probability-pct-predictor"
RUN_NAME        = "final-model-cartwave"

mlflow.set_experiment("cartwave-return-probability-pct")

with mlflow.start_run(run_name=RUN_NAME) as run:
    run_id = run.info.run_id

    mlflow.set_tag("domain", "e-commerce")
    mlflow.set_tag("phase", "final_model")

    # Log best_params (all of them, including learning_rate for audit trail)
    mlflow.log_params(best_params)

    mlflow.log_metrics({
        "MAE":  mae,
        "RMSE": rmse,
        "R2":   r2,
    })

    # Log model artifact AND register it in one call
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_NAME,
    )

print(f"[Phase 3] Run ID : {run_id}")

# ─── 5. Capture Version Number ───────────────────────────────────────────────
client = mlflow.MlflowClient()

# Fetch all versions for this model and pick the latest
versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
version_number = max(int(v.version) for v in versions)

print(f"[Phase 3] Registered model version: {version_number}")

# ─── 6. Save Output ──────────────────────────────────────────────────────────
output = {
    "registered_model_name": REGISTERED_NAME,
    "version":               version_number,
    "run_id":                run_id,
    "source_metric":         "rmse",
    "source_metric_value":   rmse,
}

os.makedirs('results', exist_ok=True)
with open('results/step3_s6.json', 'w') as f:
    json.dump(output, f, indent=4)

print("\n✅ Phase 3 complete!")
print(f"   Registered as : {REGISTERED_NAME}  v{version_number}")
print(f"   RMSE          : {rmse:.4f}")
print(f"   Saved         → results/step3_s6.json")
