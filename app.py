import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load trained pipeline components
saved_artifact = joblib.load("model.joblib")
model = saved_artifact["model"]
features = saved_artifact["features"]

# Value mapping dictionary for conversion
MAPPER = {"yes": 1, "no": 0, "male": 1, "female": 0}


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "Early-stage diabetes prediction API is running.",
            "required_features": features,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        # Build feature vector
        processed_data = {}
        for feat in features:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400

            val = data[feat]
            # Convert strings like 'Yes'/'No'/'Male'/'Female' or accept direct 0/1
            if isinstance(val, str):
                val_clean = val.strip().lower()
                if val_clean in MAPPER:
                    processed_data[feat] = MAPPER[val_clean]
                else:
                    return jsonify(
                        {"error": f"Invalid string value '{val}' for {feat}"}
                    ), 400
            else:
                processed_data[feat] = float(val)

        # Predict
        input_df = pd.DataFrame([processed_data])[features]
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        return jsonify(
            {
                "prediction": "Positive" if pred == 1 else "Negative",
                "risk_probability": round(float(prob), 4),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)