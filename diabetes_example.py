import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Buat direktori untuk artifact jika belum ada
os.makedirs("mlruns", exist_ok=True)

# Siapkan tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = f"diabetes-experiment-fix"
mlflow.set_experiment(experiment_name)

# Muat dan persiapkan data
print("Loading diabetes dataset...")
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

# Parameter model
n_estimators = 100
max_depth = 6

# Mulai MLflow run dengan error handling
try:
    with mlflow.start_run(run_name=f"rf_model_{n_estimators}_{max_depth}"):
        print(f"MLflow run started with run_id: {mlflow.active_run().info.run_id}")
        
        # Log parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("python_version", "3.10")
        
        # Latih model
        print("Training Random Forest model...")
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluasi model
        print("Evaluating model...")
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        
        # Hitung metrik
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Log metrik
        print("Logging metrics...")
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_r2": train_r2,
            "test_r2": test_r2
        })
        
        # Buat dan simpan plot
        print("Creating feature importance plot...")
        plt.figure(figsize=(10, 6))
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), diabetes.feature_names[indices], rotation=90)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Simpan plot lokasi yang jelas
        plt_path = os.path.join(os.getcwd(), "feature_importance.png")
        plt.savefig(plt_path)
        print(f"Plot saved to {plt_path}")
        
        # Log artifact dengan full path
        print("Logging artifact...")
        mlflow.log_artifact(plt_path)
        
        # Log model dengan explicit signature
        print("Logging model...")
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_train_pred)
        mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature)
        
        print(f"Training complete! Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model metrics - Training RMSE: {train_rmse:.2f}, Testing RMSE: {test_rmse:.2f}")
        print(f"R² scores - Training: {train_r2:.4f}, Testing: {test_r2:.4f}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()