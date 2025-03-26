import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("hyperparameter-tuning")

# Prepare data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10]
}

# Base model
rf = RandomForestRegressor(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

# Fit grid search
print("Training models with grid search...")
grid_search.fit(X_train, y_train)

# Get best model
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict and evaluate
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Manual calculation
r2 = r2_score(y_test, y_pred)

# Log with MLflow
with mlflow.start_run(run_name="grid_search_best"):
    # Log best parameters
    mlflow.log_params(best_params)
    
    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2
    })
    
    # Define signature
    signature = infer_signature(X_train, best_rf.predict(X_train))
    
    # Log best model
    mlflow.sklearn.log_model(best_rf, "best_model", signature=signature)
    
    # Log all results as CSV
    import pandas as pd
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv("grid_search_results.csv", index=False)
    mlflow.log_artifact("grid_search_results.csv")
    
    print(f"Best parameters: {best_params}")
    print(f"Best model performance - RMSE: {rmse:.2f}, R²: {r2:.4f}")