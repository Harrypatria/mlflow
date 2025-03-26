import sys
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Get command line arguments
dataset_name = sys.argv[1]
model_type = sys.argv[2]
n_estimators = int(sys.argv[3])
max_depth = int(sys.argv[4])

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(f"{dataset_name}-{model_type}")

# Load dataset
if dataset_name == "diabetes":
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
elif dataset_name == "california":
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

# Prepare data
X = data.data
y = data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
if model_type == "rf":
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif model_type == "gbr":
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else:
    raise ValueError(f"Unknown model type: {model_type}")

# Train and evaluate
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "dataset": dataset_name,
        "model_type": model_type,
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Manual calculation
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2
    })
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
    
    # Create signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log model
    mlflow.sklearn.log_model(model, "model", signature=signature)
    
    print(f"Model trained. RMSE: {rmse:.4f}, R²: {r2:.4f}")