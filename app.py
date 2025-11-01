from flask import Flask, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("forest_fire_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Forest Fire Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()

        temp = float(data["temperature"])
        humidity = float(data["humidity"])
        smoke = float(data["smoke"])

        # Feature engineering (same as your Streamlit code)
        temp_hum = temp / (humidity + 1)
        temp_smoke = temp / (smoke + 1)
        smoke_hum = smoke / (humidity + 1)

        X_new = np.array([[temp, humidity, smoke, temp_hum, temp_smoke, smoke_hum]])
        X_scaled = scaler.transform(X_new)

        # Predict
        pred = int(model.predict(X_scaled)[0])
        prob = float(model.predict_proba(X_scaled)[0, 1])

        # Return result as JSON
        return jsonify({
            "fire_risk": pred,
            "probability": round(prob, 3),
            "message": "🔥 Forest Fire Detected!" if pred == 1 else "✅ No Fire Detected."
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
