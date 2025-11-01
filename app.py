from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and scaler from current directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "forest_fire_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Forest Fire Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        temp = float(data["temperature"])
        humidity = float(data["humidity"])
        smoke = float(data["smoke"])

        temp_hum = temp / (humidity + 1)
        temp_smoke = temp / (smoke + 1)
        smoke_hum = smoke / (humidity + 1)

        X_new = np.array([[temp, humidity, smoke, temp_hum, temp_smoke, smoke_hum]])
        X_scaled = scaler.transform(X_new)

        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0, 1])

        return jsonify({
            "fire_risk": pred,
            "probability": round(prob, 3),
            "message": "🔥 Forest Fire Detected!" if pred == 1 else "✅ No Fire Detected."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Export app object for Vercel serverless runtime
handler = app
