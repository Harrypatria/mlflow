import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# Siapkan tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Buat nama eksperimen dengan timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"model-comparison-{timestamp}"
mlflow.set_experiment(experiment_name)
print(f"Experiment: {experiment_name}")

# Muat dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Feature names: {feature_names}")

# Buat dataframe untuk visualisasi
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Bagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisi model yang akan dibandingkan
models = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [4, 6, 8]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "ElasticNet": {
        "model": ElasticNet(random_state=42),
        "params": {
            "alpha": [0.1, 0.5, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9]
        }
    }
}

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model with cross-validation and test set."""
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, 
                               cv=cv, scoring='neg_mean_squared_error')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # CV metrics
    cv_mse = -cv_scores.mean()
    cv_mse_std = cv_scores.std()
    
    metrics = {
        "cv_mse": cv_mse,
        "cv_mse_std": cv_mse_std,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "train_mae": train_mae,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "test_mae": test_mae
    }
    
    return metrics, test_pred

def plot_residuals(y_true, y_pred, title):
    """Create residual plot."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {title}')
    
    residual_path = f"residuals_{title.replace(' ', '_').lower()}.png"
    plt.savefig(residual_path)
    return residual_path

# Jalankan grid search sederhana dan track dengan MLflow
best_models = {}
best_metrics = {}

# For each model type
for model_name, model_config in models.items():
    print(f"\nEvaluating {model_name} models...")
    
    # Combinatorial parameter search
    best_score = float('inf')  # Lower MSE is better
    best_params = None
    best_model = None
    
    # Generate all parameter combinations
    from itertools import product
    param_names = model_config["params"].keys()
    param_values = model_config["params"].values()
    
    # For each parameter combination
    for i, param_combo in enumerate(product(*param_values)):
        params = dict(zip(param_names, param_combo))
        
        # Create a descriptive run name
        param_str = "_".join([f"{k}={v}" for k, v in params.items()])
        run_name = f"{model_name}_{param_str}"
        
        print(f"  Run {i+1}: {run_name}")
        
        # Create model with these parameters
        model = model_config["model"].__class__(**params, random_state=42)
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("python_version", "3.10.0")
            
            # Evaluate
            metrics, test_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log artifacts - residual plot
            residual_path = plot_residuals(y_test, test_pred, run_name)
            mlflow.log_artifact(residual_path)
            
            # For tree-based models, log feature importance
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                importances = model.feature_importances_
                indices = np.argsort(importances)
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.title(f'Feature Importance - {run_name}')
                
                importance_path = f"importance_{run_name.replace(' ', '_').lower()}.png"
                plt.savefig(importance_path)
                mlflow.log_artifact(importance_path)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Keep track of best model for this type
            if metrics["test_mse"] < best_score:
                best_score = metrics["test_mse"]
                best_params = params
                best_model = model
                best_metrics[model_name] = metrics
            
            # Print results
            print(f"    Test MSE: {metrics['test_mse']:.2f}, R2: {metrics['test_r2']:.2f}")
    
    # Store best model of this type
    best_models[model_name] = (best_model, best_params)
    print(f"  Best {model_name} parameters: {best_params}")
    print(f"  Best {model_name} test MSE: {best_score:.2f}")

# Determine overall best model
best_model_type = min(best_metrics, key=lambda k: best_metrics[k]["test_mse"])
best_model, best_params = best_models[best_model_type]

print("\n" + "="*50)
print(f"Best overall model: {best_model_type}")
print(f"Parameters: {best_params}")
print(f"Test MSE: {best_metrics[best_model_type]['test_mse']:.2f}")
print(f"Test R2: {best_metrics[best_model_type]['test_r2']:.2f}")
print("="*50)

# Log the best model in a final run
with mlflow.start_run(run_name=f"best_model_{best_model_type}"):
    # Log model type and params
    mlflow.log_param("model_type", best_model_type)
    mlflow.log_params(best_params)
    
    # Log metrics
    mlflow.log_metrics(best_metrics[best_model_type])
    
    # Log the model
    mlflow.sklearn.log_model(best_model, "best_model")
    
    print("\nBest model logged to MLflow")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"View results in MLflow UI: run 'mlflow ui' and open http://localhost:5000")