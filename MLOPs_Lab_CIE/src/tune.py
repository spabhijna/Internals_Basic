import pandas as pd
import json
import os
import itertools
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─── 1. Load Best Model Type from Phase 1 ────────────────────────────────────
with open('results/step1_s1.json', 'r') as f:
    step1 = json.load(f)

best_model_name = step1['best_model']
print(f"[Phase 2] Best model from Phase 1: {best_model_name}")

MODEL_MAP = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

if best_model_name not in MODEL_MAP:
    raise ValueError(f"Unknown model: {best_model_name}")

ModelClass = MODEL_MAP[best_model_name]

# ─── 2. Prepare Data ─────────────────────────────────────────────────────────
df = pd.read_csv('data/training_data.csv')

X = df[['product_price', 'delivery_days', 'customer_rating', 'is_first_order']]
y = df['return_probability_pct']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── 3. Define Parameter Grid ────────────────────────────────────────────────
param_grid = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth':     [3, 5],
}

# All 3 × 3 × 2 = 18 combinations
all_combinations = list(itertools.product(
    param_grid['n_estimators'],
    param_grid['learning_rate'],
    param_grid['max_depth'],
))

assert len(all_combinations) == 18, f"Expected 18 trials, got {len(all_combinations)}"

# ─── 4 & 5. MLflow parent run + nested grid search ───────────────────────────
mlflow.set_experiment("cartwave-return-probability-pct")

best_rmse     = float('inf')
best_params   = {}
best_mae      = float('inf')
best_cv_mae   = float('inf')

PARENT_RUN_NAME = "tuning-cartwave"
N_FOLDS = 3

with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent_run:
    mlflow.set_tag("phase", "hyperparameter_tuning")
    mlflow.set_tag("domain", "e-commerce")
    mlflow.log_param("search_type", "grid")
    mlflow.log_param("n_folds", N_FOLDS)
    mlflow.log_param("total_trials", len(all_combinations))
    mlflow.log_param("best_model_type", best_model_name)

    for trial_idx, (n_est, lr, depth) in enumerate(all_combinations, start=1):

        # Build params dict — RandomForest ignores learning_rate at runtime
        # but we log it for consistency (exam requirement)
        params = {
            'n_estimators':  n_est,
            'learning_rate': lr,
            'max_depth':     depth,
            'random_state':  42,
        }

        # RandomForest doesn't accept learning_rate; pass only what it supports
        if ModelClass == RandomForestRegressor:
            model_params = {k: v for k, v in params.items() if k != 'learning_rate'}
        else:
            model_params = params.copy()

        model = ModelClass(**model_params)

        # 3-fold CV on training set (negative MAE → positive)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=N_FOLDS,
            scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()

        # Fit on full train split, evaluate on test split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # ── Nested MLflow run ────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"trial_{trial_idx:02d}", nested=True):
            # Log all params (including learning_rate for consistency)
            mlflow.log_params({
                'n_estimators':  n_est,
                'learning_rate': lr,
                'max_depth':     depth,
            })
            mlflow.log_metrics({'MAE': mae, 'RMSE': rmse, 'CV_MAE': cv_mae})
            mlflow.set_tag("domain", "e-commerce")
            mlflow.set_tag("trial", trial_idx)

        print(
            f"  Trial {trial_idx:02d}/18 | "
            f"n_est={n_est:3d} lr={lr} depth={depth} | "
            f"RMSE={rmse:.4f}  MAE={mae:.4f}  CV_MAE={cv_mae:.4f}"
        )

        # ── Track best ───────────────────────────────────────────────────────
        if rmse < best_rmse:
            best_rmse   = rmse
            best_mae    = mae
            best_cv_mae = cv_mae
            best_params = {
                'n_estimators':  n_est,
                'learning_rate': lr,
                'max_depth':     depth,
            }

    # Log best results to parent run
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metrics({
        'best_RMSE':   best_rmse,
        'best_MAE':    best_mae,
        'best_CV_MAE': best_cv_mae,
    })

# ─── 7. Save Output ──────────────────────────────────────────────────────────
output = {
    "search_type":      "grid",
    "n_folds":          N_FOLDS,
    "total_trials":     18,
    "best_params":      best_params,
    "best_mae":         best_mae,
    "best_cv_mae":      best_cv_mae,
    "parent_run_name":  PARENT_RUN_NAME,
}

os.makedirs('results', exist_ok=True)
with open('results/step2_s2.json', 'w') as f:
    json.dump(output, f, indent=4)

print("\n✅ Phase 2 complete!")
print(f"   Best params : {best_params}")
print(f"   Best MAE    : {best_mae:.4f}")
print(f"   Best CV_MAE : {best_cv_mae:.4f}")
print(f"   Best RMSE   : {best_rmse:.4f}")
print("   Saved → results/step2_s2.json")
