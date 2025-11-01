from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

MODEL_PATH = "forest_fire_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route("/")
def home():
    return jsonify({"message": "Forest Fire Prediction API running successfully"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        smoke = float(data["smoke"])

        temp_hum = temperature / (humidity + 1)
        temp_smoke = temperature / (smoke + 1)
        smoke_hum = smoke / (humidity + 1)

        X_new = np.array([[temperature, humidity, smoke, temp_hum, temp_smoke, smoke_hum]])
        X_scaled = scaler.transform(X_new)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0, 1]

        if pred == 1:
            return jsonify({
                "fire_risk": 1,
                "message": "🔥 Forest Fire Detected!",
                "probability": round(float(prob), 2)
            })
        else:
            return jsonify({
                "fire_risk": 0,
                "message": "✅ No Fire Detected.",
                "probability": round(float(prob), 2)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
