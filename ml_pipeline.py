import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ml-pipeline")

# Load data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
])

# Train and log
with mlflow.start_run(run_name="diabetes_pipeline"):
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "scaler": "StandardScaler"
    }
    mlflow.log_params(params)
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Manual calculation of RMSE
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2
    })
    
    # Create signature
    signature = infer_signature(X_train, pipeline.predict(X_train))
    
    # Log model - entire pipeline
    mlflow.sklearn.log_model(pipeline, "pipeline_model", signature=signature)
    
    print(f"Pipeline trained. RMSE: {rmse:.2f}, R²: {r2:.4f}")