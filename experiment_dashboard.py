import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Setup tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("california-housing")
if experiment is None:
    print("Experiment not found!")
    exit(1)

# Get all runs for the experiment
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"]
)

# Collect data for visualization
run_data = []
for run in runs:
    if run.info.status == "FINISHED":
        run_data.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "Unknown"),
            "rmse": run.data.metrics.get("rmse", 0),
            "r2": run.data.metrics.get("r2", 0),
            "n_estimators": run.data.params.get("n_estimators", ""),
            "max_depth": run.data.params.get("max_depth", ""),
            "learning_rate": run.data.params.get("learning_rate", "")
        })

# Convert to DataFrame
df = pd.DataFrame(run_data)
if df.empty:
    print("No completed runs found!")
    exit(1)

print("Top 5 runs by RMSE:")
print(df[["run_name", "rmse", "r2", "n_estimators", "max_depth", "learning_rate"]].head())

# Visualize RMSE comparison
plt.figure(figsize=(12, 6))
plt.barh(df["run_name"], df["rmse"])
plt.xlabel("RMSE (lower is better)")
plt.ylabel("Run Name")
plt.title("RMSE Comparison Across Runs")
plt.tight_layout()
plt.savefig("rmse_comparison.png")

# Visualize R2 comparison
plt.figure(figsize=(12, 6))
plt.barh(df["run_name"], df["r2"])
plt.xlabel("R² (higher is better)")
plt.ylabel("Run Name")
plt.title("R² Comparison Across Runs")
plt.tight_layout()
plt.savefig("r2_comparison.png")

print("Visualizations saved as 'rmse_comparison.png' and 'r2_comparison.png'")