from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import gc
import psutil

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_optimized.tflite")

# Global variables for lazy loading
interpreter = None
input_details = None
output_details = None

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def load_model():
    """Lazy load the TensorFlow Lite model"""
    global interpreter, input_details, output_details
    
    if interpreter is None:
        try:
            print(f"Loading TensorFlow Lite model from: {MODEL_PATH}")
            print(f"Memory before loading: {get_memory_usage():.2f} MB")
            
            # Load TensorFlow Lite model
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"Model loaded successfully!")
            print(f"Memory after loading: {get_memory_usage():.2f} MB")
            print(f"Input shape: {input_details[0]['shape']}")
            
        except Exception as e:
            print(f"Error loading TensorFlow Lite model: {e}")
            interpreter = None
            return False
    
    return True

# Initialize Flask
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    model_loaded = interpreter is not None
    memory_mb = get_memory_usage()
    
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "memory_usage_mb": memory_mb
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Lazy load model on first prediction
    if not load_model():
        return jsonify({"error": "Model failed to load. Check server logs."}), 500
    
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
            return jsonify({
                "error": "Invalid input: Missing one or more required features.",
                "required_features": required_features
            }), 400
        
        # Create feature array in correct order
        feature_values = [
            float(data["distance_from_home"]),
            float(data["distance_from_last_transaction"]),
            float(data["ratio_to_median_purchase_price"]),
            float(data["repeat_retailer"]),
            float(data["used_chip"]),
            float(data["used_pin_number"]),
            float(data["online_order"])
        ]
        
        # Prepare input for TensorFlow Lite model
        features = np.array([feature_values], dtype=np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], features)
        
        # Run inference
        interpreter.invoke()
        
        # Get prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Determine predicted class (assuming binary classification)
        fraud_probability = float(prediction[0][0])
        predicted_class = int(fraud_probability > 0.5)
        
        # Clean up memory
        gc.collect()
        
        # Response
        result = {
            "prediction": "Fraudulent" if predicted_class == 1 else "Legit",
            "fraud_probability": fraud_probability,
            "confidence": abs(fraud_probability - 0.5) * 2,  # Distance from decision boundary
            "features_received": data  # Optional: For debugging
        }
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": f"Invalid data format: {str(e)}"}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.after_request
def cleanup(response):
    """Clean up memory after each request"""
    gc.collect()
    return response

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    print(f"Looking for model at: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=False)
