import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from mlflow.models.signature import infer_signature

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("california-housing")

# Load data
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

# Create DataFrame for visualization
df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter combinations to try
param_sets = [
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
    {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 5}
]

# Train models with different parameters
for i, params in enumerate(param_sets):
    run_name = f"gbr_model_{i+1}"
    print(f"\nTraining model {i+1} with parameters: {params}")
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = GradientBoostingRegressor(**params, random_state=42)
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
        
        # Feature importance plot
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save and log plot
        plt.savefig(f"feature_importance_{i+1}.png")
        mlflow.log_artifact(f"feature_importance_{i+1}.png")
        
        # Create signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log model
        mlflow.sklearn.log_model(model, f"gbr_model_{i+1}", signature=signature)
        
        print(f"Model {i+1} trained. RMSE: {rmse:.4f}, R²: {r2:.4f}")

print("\nAll models trained. Check MLflow UI to compare results.")