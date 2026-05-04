# CartWave — Return Probability MLOps Pipeline

---

## 🧠 1. Project Overview

**Problem Statement**
In e-commerce, product returns are expensive. This project trains a regression model to predict the probability (%) that a customer will return an item, given purchase and delivery attributes.

**Domain:** E-Commerce

**Objective:**
Build an end-to-end MLOps pipeline that covers:
- Experiment tracking with MLflow (parameters, metrics, run management)
- Hyperparameter tuning with 3-fold cross-validation grid search
- Model registration and versioning via MLflow Model Registry
- Automated retraining on new data with champion/challenger promotion logic

---

## ⚙️ 2. Project Structure

```
MLOPs_Lab_CIE/
│── data/
│   ├── training_data.csv       # 25 original training samples
│   └── new_data.csv            # 20 new samples (Phase 4 drift simulation)
│── src/
│   ├── train.py                # Phase 1 — model training & MLflow logging
│   ├── tune.py                 # Phase 2 — hyperparameter grid search
│   ├── register.py             # Phase 3 — final model registration
│   ├── retrain.py              # Phase 4 — retraining & promotion pipeline
│   └── validate.py             # Phase 5 — full validation & sanity checks
│── models/                     # Reserved for serialised model artefacts
│── results/
│   ├── step1_s1.json           # Phase 1 output
│   ├── step2_s2.json           # Phase 2 output
│   ├── step3_s6.json           # Phase 3 output
│   └── step4_s8.json           # Phase 4 output
│── README.md
```

**Features used for training:**

| Feature | Description |
|---|---|
| `product_price` | Price of the purchased product |
| `delivery_days` | Days taken for delivery |
| `customer_rating` | Customer satisfaction rating (1–5) |
| `is_first_order` | Whether this is the customer's first order (0/1) |

**Target:** `return_probability_pct` — percentage probability of return

---

## 📊 3. Phase 1 — Model Training & Tracking

**Script:** `src/train.py`
**MLflow Experiment:** `cartwave-return-probability-pct`

Both models were trained with default hyperparameters (`random_state=42`) on an 80/20 train-test split and tracked in MLflow with the tag `domain = "e-commerce"`.

### Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| RandomForestRegressor | 8.2835 | 10.7040 | -1.1523 |
| GradientBoostingRegressor | 8.4448 | 11.0203 | -1.2814 |

**Best Model:** `RandomForestRegressor`
**Best RMSE:** `10.7040`

> The negative R² values are expected given the small dataset size (25 samples); the model still generalises comparatively better across phases as data grows.

---

## 🔍 4. Phase 2 — Hyperparameter Tuning

**Script:** `src/tune.py`
**MLflow Run:** `tuning-cartwave` (parent) → 18 nested child runs

### Search Configuration

| Setting | Value |
|---|---|
| Search type | Grid Search |
| Cross-validation folds | 3 |
| Total trials | 18 (3 × 3 × 2) |
| Parameter: `n_estimators` | [50, 100, 200] |
| Parameter: `learning_rate` | [0.05, 0.1, 0.2] |
| Parameter: `max_depth` | [3, 5] |

> `learning_rate` is logged for all trials for consistency even though `RandomForestRegressor` ignores it at runtime.

### Best Configuration

```json
{
  "n_estimators": 200,
  "learning_rate": 0.05,
  "max_depth": 3
}
```

| Metric | Value |
|---|---|
| Best CV MAE | 4.1942 |
| Best MAE (test) | 7.8928 |

### MLflow Structure

```
[Parent Run] tuning-cartwave
  ├── [Nested] trial_01  — n_est=50,  lr=0.05, depth=3
  ├── [Nested] trial_02  — n_est=50,  lr=0.05, depth=5
  │   ...
  └── [Nested] trial_18  — n_est=200, lr=0.2,  depth=5
```

---

## 📦 5. Phase 3 — Model Registry

**Script:** `src/register.py`
**MLflow Run:** `final-model-cartwave`

The best model (tuned `RandomForestRegressor`) was retrained on the original training split and registered in the MLflow Model Registry.

### Registry Details

| Field | Value |
|---|---|
| Registered model name | `cartwave-return-probability-pct-predictor` |
| Version | `1` |
| Run ID | `ca41cd85e8d0483884ff5731a755bc64` |
| RMSE | `10.4325` |

**What is the Model Registry?**
The MLflow Model Registry provides centralised storage, versioning, and lifecycle management for trained models. Each registered version is traceable back to its exact training run, parameters, and metrics — making rollback, comparison, and deployment auditable by default.

---

## 🔁 6. Phase 4 — Retraining Pipeline

**Script:** `src/retrain.py`
**MLflow Run:** `retrain-cartwave`

New data arrived, simulating real-world distribution shift (higher prices, longer delivery windows, higher return rates). Both the champion and a freshly retrained model were evaluated on the **same** test split derived from the combined dataset to ensure a fair comparison.

### Dataset Sizes

| Dataset | Rows |
|---|---|
| Original (`training_data.csv`) | 25 |
| New (`new_data.csv`) | 20 |
| Combined | **45** |

### Model Comparison (same test set)

| Model | RMSE |
|---|---|
| Champion (registry v1, trained on 25 rows) | 26.8125 |
| Retrained (combined 45 rows, tuned params) | 9.7343 |

### Promotion Decision

| Metric | Value |
|---|---|
| Improvement | 17.0781 |
| Minimum threshold | 1.0 |
| **Action** | **`promoted`** |

The retrained model cleared the threshold by a wide margin. It was registered as **`cartwave-return-probability-pct-predictor` v2** in the MLflow Model Registry.

---

## 📈 7. MLflow Tracking Summary

**Experiment Name:** `cartwave-return-probability-pct`

| Run Name | Phase | What's Tracked |
|---|---|---|
| *(Phase 1 runs)* | Training | Model type, MAE, RMSE, R², `domain` tag |
| `tuning-cartwave` | Tuning | Search config, best params, best metrics |
| `trial_01` … `trial_18` | Tuning (nested) | Per-combination params, MAE, RMSE, CV_MAE |
| `final-model-cartwave` | Registration | Tuned params, MAE, RMSE, R², model artifact |
| `retrain-cartwave` | Retraining | Combined row count, champion/retrained RMSE, improvement, action |

**Tracked across all runs:**
- **Parameters** — model hyperparameters, search config, data split settings
- **Metrics** — MAE, RMSE, R², CV MAE, improvement delta
- **Tags** — `domain = "e-commerce"`, `phase`, `trial` index
- **Artifacts** — serialised `sklearn` model (logged via `mlflow.sklearn.log_model`)
- **Registry** — 2 versioned entries under `cartwave-return-probability-pct-predictor`

To explore all runs interactively:

```bash
cd MLOPs_Lab_CIE
source .venv/bin/activate
mlflow ui
# Open http://127.0.0.1:5000
```

---

## 🚀 Running the Pipeline

```bash
# Setup
cd MLOPs_Lab_CIE
python -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn mlflow

# Execute phases in order
python src/train.py      # Phase 1 — training & tracking
python src/tune.py       # Phase 2 — hyperparameter tuning
python src/register.py   # Phase 3 — model registration
python src/retrain.py    # Phase 4 — retraining & promotion

# Validate everything
python src/validate.py   # Phase 5 — sanity checks (exit 0 = all pass)
```

---

*Built as part of MLOps Lab CIE | Domain: E-Commerce | Experiment: `cartwave-return-probability-pct`*
