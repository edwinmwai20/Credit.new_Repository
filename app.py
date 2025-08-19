from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.h5")

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize Flask
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
    
    try:
        data = request.get_json()

        # Validate required inputs for the 7 features
        required_features = [
            "distance_from_home",
            "distance_from_last_transaction",
            "ratio_to_median_purchase_price",
            "repeat_retailer",
            "used_chip",
            "used_pin_number",
            "online_order"
        ]

        if not data or not all(feat in data for feat in required_features):
            return jsonify({"error": "Invalid input: Missing one or more required features."}), 400

        # Create a list of feature values in the correct order for the model
        feature_values = [
            data["distance_from_home"],
            data["distance_from_last_transaction"],
            data["ratio_to_median_purchase_price"],
            data["repeat_retailer"],
            data["used_chip"],
            data["used_pin_number"],
            data["online_order"]
        ]

        # Prepare input for model: convert the list of features to a NumPy array with the correct shape (1, 7)
        features = np.array([feature_values], dtype=float)

        # Predict
        prediction = model.predict(features)
        
        # Determine the predicted class based on a 0.5 threshold
        predicted_class = int((prediction > 0.5).astype("int32")[0][0])
        
        # Get the probability for the predicted class
        fraud_probability = float(prediction[0][0])

        # Response
        result = {
            "prediction": "Fraudulent" if predicted_class == 1 else "Legit",
            "fraud_probability": fraud_probability,
            "features_received": data # Optional: For debugging
        }

        return jsonify(result), 200

    except Exception as e:
        # Catch any other errors and return a general error message
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) # Get port from env variable, default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)
