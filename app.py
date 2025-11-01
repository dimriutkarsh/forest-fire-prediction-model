from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load model
try:
    model = joblib.load("forest_fire_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Don't exit in Vercel environment
    model = None
    scaler = None

FEATURE_ORDER = ["temperature", "humidity", "smoke", "temp_max", "temp_min",
                "pressure", "clouds_all", "wind_speed", "wind_deg", "temp_local", "wind_gust"]

@app.route("/")
def home():
    return jsonify({"message": "🌲 Forest Fire Prediction API is running on Vercel!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Check required fields
        missing_fields = [field for field in FEATURE_ORDER if field not in data and field != "wind_gust"]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Prepare features
        features = []
        for field in FEATURE_ORDER:
            if field == "wind_gust":
                value = data.get(field, 0)
            else:
                value = data[field]
            features.append(float(value))
        
        # Predict
        features_array = np.array([features])
        scaled_features = scaler.transform(features_array)
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0, 1])
        
        response = jsonify({
            "fire_risk": prediction,
            "probability": round(probability, 3),
            "message": "🔥 Forest Fire Detected!" if prediction == 1 else "✅ No Fire Detected."
        })
        
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def _build_cors_preflight_response():
    response = jsonify({"status": "preflight"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

# Vercel requires the app variable to be named 'app'
if __name__ == "__main__":
    app.run(debug=False)
