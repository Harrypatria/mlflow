from flask import Flask, request, jsonify, render_template_string
import mlflow.sklearn
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
import os

# Setup MLflow client
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Create Flask app
app = Flask(__name__)

# Function to load production model
def load_production_model():
    # Find model in Production stage
    for model in client.search_registered_models():
        for version in client.get_latest_versions(model.name, stages=["Production"]):
            print(f"Found production model: {model.name}, version {version.version}")
            model_uri = f"models:/{model.name}/{version.version}"
            return mlflow.sklearn.load_model(model_uri), model.name, version.version
    
    # If no production model, try to find staging models
    for model in client.search_registered_models():
        for version in client.get_latest_versions(model.name, stages=["Staging"]):
            print(f"Found staging model: {model.name}, version {version.version}")
            model_uri = f"models:/{model.name}/{version.version}"
            return mlflow.sklearn.load_model(model_uri), model.name, version.version
    
    raise Exception("No production or staging models found")

# Load model when app starts
try:
    model, model_name, model_version = load_production_model()
    print(f"Loaded model: {model_name}, version {model_version}")
except Exception as e:
    print(f"Error loading model: {e}")
    model, model_name, model_version = None, None, None

# HTML template for the homepage
homepage_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Diabetes Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1 { color: #2c3e50; }
        .container { max-width: 800px; margin: 0 auto; }
        .model-info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .endpoint { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
        .footer { margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Diabetes Prediction API</h1>
        
        <div class="model-info">
            <h2>Model Information</h2>
            <p><strong>Model Name:</strong> {{ model_name }}</p>
            <p><strong>Model Version:</strong> {{ model_version }}</p>
            <p><strong>Status:</strong> {{ "Running" if model_name else "No model loaded" }}</p>
        </div>
        
        <h2>API Endpoints</h2>
        
        <div class="endpoint">
            <h3>POST /predict</h3>
            <p>Make predictions with the deployed model</p>
            <h4>Request Format:</h4>
            <pre>
{
  "instances": [
    [0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235, -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]
  ]
}
            </pre>
            <h4>Response Format:</h4>
            <pre>
{
  "predictions": [152.14],
  "model_info": {
    "name": "DiabetesPredictorRandomForest",
    "version": "1"
  }
}
            </pre>
        </div>
        
        <div class="endpoint">
            <h3>GET /model-info</h3>
            <p>Get information about the currently deployed model</p>
            <h4>Response Format:</h4>
            <pre>
{
  "model_name": "DiabetesPredictorRandomForest",
  "model_version": "1",
  "input_features": [
    "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
  ]
}
            </pre>
        </div>
        
        <div class="footer">
            <p>Diabetes Prediction API | Powered by MLflow and Flask</p>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    """Render homepage with API documentation"""
    return render_template_string(homepage_template, model_name=model_name, model_version=model_version)

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the deployed model"""
    if model is None:
        return jsonify({"error": "No model loaded"}), 503
    
    # Diabetes dataset features
    features = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    
    return jsonify({
        "model_name": model_name,
        "model_version": model_version,
        "input_features": features
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with the deployed model"""
    if model is None:
        return jsonify({"error": "No model loaded"}), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'instances' not in data:
            return jsonify({"error": "Request must include 'instances' field"}), 400
        
        # Convert to numpy array
        instances = np.array(data['instances'])
        
        # Make prediction
        predictions = model.predict(instances)
        
        # Return JSON response
        return jsonify({
            'predictions': predictions.tolist(),
            'model_info': {
                'name': model_name,
                'version': model_version
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Reload the production model"""
    global model, model_name, model_version
    
    try:
        model, model_name, model_version = load_production_model()
        return jsonify({
            "success": True,
            "model_name": model_name,
            "model_version": model_version
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)