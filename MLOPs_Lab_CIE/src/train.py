import pandas as pd
import json
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def main():
    # Load data
    data_path = 'data/training_data.csv'
    df = pd.read_csv(data_path)
    
    # Split data
    X = df[['product_price', 'delivery_days', 'customer_rating', 'is_first_order']]
    y = df['return_probability_pct']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Setup MLflow
    EXPERIMENT_NAME = "cartwave-return-probability-pct"
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Log params
            mlflow.log_params(model.get_params())
            
            # Log metrics
            mlflow.log_metrics({
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })
            
            # Log tag
            mlflow.set_tag("domain", "e-commerce")
            
            # Store results
            results[model_name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            }
            
            # Check if best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name


    # Save results to JSON — schema matches final checklist requirements
    output_data = {
        "experiment_name": EXPERIMENT_NAME,
        "models": {
            model_name: {
                "MAE":  results[model_name]["MAE"],
                "RMSE": results[model_name]["RMSE"],
                "R2":   results[model_name]["R2"],
            }
            for model_name in results
        },
        "best_model":        best_model_name,
        "best_metric_name":  "rmse",
        "best_metric_value": best_rmse,
        # keep best_rmse alias so downstream scripts (tune/register/retrain) still work
        "best_rmse":         best_rmse,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/step1_s1.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved. Best model: {best_model_name} with RMSE: {best_rmse:.4f}")

if __name__ == '__main__':
    main()
