import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("model-registry")

# Prepare data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Manual calculation
r2 = r2_score(y_test, y_pred)

# Log with MLflow
with mlflow.start_run(run_name="production_model"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 6
    })
    
    # Log metrics
    mlflow.log_metrics({
        "rmse": rmse,
        "r2": r2
    })
    
    # Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log and register model
    mlflow.sklearn.log_model(
        model, 
        "diabetes_predictor",
        signature=signature,
        registered_model_name="DiabetesPredictor"
    )
    
    print(f"Model registered. RMSE: {rmse:.2f}, R²: {r2:.4f}")