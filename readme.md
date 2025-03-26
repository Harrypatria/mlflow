# MLflow Tutorial Project

A complete guide to MLflow implementation for machine learning workflow management, including experimentation tracking, model registry, and API deployment.

![MLflow Dashboard](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/_static/images/quickstart/mlflow-ui.png)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tutorial Steps](#-tutorial-steps)
  - [1. Setup Environment](#1-setup-environment)
  - [2. Experiment Tracking](#2-experiment-tracking)
  - [3. Model Registry](#3-model-registry)
  - [4. API Deployment](#4-api-deployment)
  - [5. Monitoring & Maintenance](#5-monitoring--maintenance)
- [Sample Results](#-sample-results)
- [Common Issues & Solutions](#-common-issues--solutions)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

This repository provides a step-by-step tutorial for implementing MLflow in a machine learning project. It covers the entire lifecycle from experimentation to deployment, using diabetes prediction as a demonstration case.

MLflow helps you track experiments, package code into reproducible runs, and share & deploy models. This project demonstrates best practices for integrating MLflow into your ML workflow.

## ✨ Features

- **Experiment Tracking**: Log and compare parameters, metrics, and artifacts
- **Model Registry**: Version control for ML models with staging transitions
- **Hyperparameter Optimization**: Systematic model tuning with MLflow tracking
- **REST API**: Flask-based prediction API that serves production models
- **Automated Updates**: CI/CD-like process for model refresh
- **Monitoring Dashboard**: Performance tracking and reporting

## 📂 Project Structure

```
mlflow-project/
├── .vscode/                    # VS Code configuration
├── data/                       # Data storage (generated during runtime)
├── mlruns/                     # MLflow tracking data (generated during runtime)
├── model_monitoring/           # Monitoring reports & visualizations
├── venv/                       # Python virtual environment
├── basic_experiment.py         # Simple MLflow tracking example
├── california_housing.py       # Additional dataset experiment
├── hyperparameter_tuning.py    # Hyperparameter optimization with MLflow
├── ml_pipeline.py              # End-to-end ML pipeline with tracking
├── model_monitoring.py         # Monitoring dashboard generator
├── model_refresh.py            # Automatic model updating service
├── MLProject                   # MLflow project definition
├── prediction_api.py           # Flask API for model serving
├── promote_models.py           # Script to promote models to production
├── python_env.yaml             # Python environment specification
├── register_models.py          # Model registry management
├── requirements.txt            # Python dependencies
├── test_api_client.py          # Test client for prediction API
└── train_base_models.py        # Main model training script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+ (3.10 recommended)
- pip (package installer)
- git

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/Harrypatria/mlflow.git
   cd mlflow
   ```

2. Create and activate virtual environment
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Tutorial Steps

### 1. Setup Environment

Initialize the environment and verify MLflow is working:

```bash
# Verify MLflow installation
python -c "import mlflow; print(mlflow.__version__)"

# Start MLflow UI (keep this running in a separate terminal)
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access the MLflow UI at http://localhost:5000

### 2. Experiment Tracking

#### Train Base Models

```bash
# Run the base model training script
python train_base_models.py
```

This script:
- Sets up an MLflow experiment
- Trains RandomForest, GradientBoosting, and ElasticNet models on diabetes dataset
- Logs parameters, metrics, and artifacts (feature importance plots)
- Creates model signatures for deployment

#### Hyperparameter Optimization

```bash
# Run hyperparameter tuning
python hyperparameter_tuning.py
```

This demonstrates:
- Nested runs for hyperparameter search
- Automated parameter and metric logging
- Best model selection and saving

#### Try Another Dataset

```bash
# Run California housing dataset experiment
python california_housing.py
```

Explore how to adapt MLflow tracking to different datasets.

### 3. Model Registry

#### Register Models

```bash
# Register best models to MLflow Model Registry
python register_models.py
```

This step:
- Finds best performing models from experiments
- Registers them with the Model Registry
- Adds descriptions and metadata

#### Promote Models to Production

```bash
# Evaluate and promote models to Staging/Production
python promote_models.py
```

This script:
- Evaluates model performance
- Promotes best models to Staging
- Selects top performer for Production
- Archives outdated versions

#### Verify in MLflow UI

Check the "Models" section in the MLflow UI to see registered models and their stages.

### 4. API Deployment

#### Start Prediction API

```bash
# Launch the prediction API server
python prediction_api.py
```

The API:
- Automatically loads the Production model
- Provides prediction endpoints
- Includes model information endpoints
- Offers a model reload mechanism

#### Test the API

```bash
# Run the API test client
python test_api_client.py
```

This tests:
- Model info retrieval
- Single and batch predictions
- API performance benchmarking

#### Sample API Request

```bash
curl -X POST http://localhost:5001/predict -H "Content-Type: application/json" -d "{\"instances\": [[0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235, -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]]}"
```

### 5. Monitoring & Maintenance

#### Generate Monitoring Dashboard

```bash
# Create performance monitoring reports
python model_monitoring.py
```

This creates:
- Performance comparison across experiments
- Visualizations of model metrics
- Registry status report
- HTML dashboard

#### Setup Model Refresh Service

```bash
# Start the automatic model refresh service
python model_refresh.py
```

This service:
- Periodically checks for new production models
- Automatically updates the API's model
- Logs all refresh activities

## 📊 Sample Results

After following this tutorial, you'll have:

- Multiple tracked experiments in MLflow
- Registered models in different stages (Development/Staging/Production)
- A running prediction API serving your best model
- Performance monitoring dashboards

Example model comparison from the MLflow UI:
![Model Comparison](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/_static/images/tutorial-compare-runs.png)

## 🛠 Common Issues & Solutions

### "No module named 'mlflow'"
Ensure you've activated your virtual environment and installed requirements.

### "Error: Could not find suitable Python environment"
Check your python_env.yaml file and ensure Python version matches your installation.

### "Registered Model not found"
Run register_models.py before running prediction_api.py or use a specific run_id instead.

### "squared parameter not found"
This occurs with older scikit-learn versions. Calculate RMSE manually: `np.sqrt(mean_squared_error(y_test, y_pred))`.

### MLflow UI not showing experiments
Verify your tracking URI is set consistently across scripts: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Made with ❤️ by [Harry](https://github.com/Harrypatria)
