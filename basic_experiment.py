import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature  # Tambahkan import ini

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("basic-experiment")

# Prepare data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Train model
with mlflow.start_run():
    # Parameter
    n_estimators = 100
    max_depth = 6
    
    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    
    # Inferring the model signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Optional: Tambahkan contoh input
    input_example = X_train[0:2]
    
    # Log model dengan signature dan input example
    mlflow.sklearn.log_model(
        model, 
        "random_forest_model",
        signature=signature,
        input_example=input_example
    )
    
    print(f"Model trained with RMSE: {rmse:.2f}")
    print(f"Model logged with signature")