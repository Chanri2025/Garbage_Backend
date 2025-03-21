from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model, scaler, and PCA
try:
    model = joblib.load("waste_classification_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    logging.info("Model, Scaler, and PCA loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or preprocessing objects: {str(e)}")
    model, scaler, pca = None, None, None


def preprocess_image(image):
    """
    Preprocess the input image by:
    - Resizing to 64x64.
    - Converting to grayscale.
    - Extracting HOG features.
    - Applying the scaler and PCA transformations.
    """
    try:
        # Resize the image
        img_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

        # Convert to grayscale if necessary
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

        # Extract HOG features
        features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys')
        features = np.array(features).reshape(1, -1)

        # Apply transformations
        features = scaler.transform(features)
        features = pca.transform(features)

        return features
    except Exception as e:
        logging.error(f"Error in image preprocessing: {str(e)}")
        raise ValueError("Failed to preprocess image.")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image classification request.
    """
    if model is None or scaler is None or pca is None:
        return jsonify({"error": "Model or preprocessing objects are not loaded."}), 500

    # Validate request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()

    # Convert image to NumPy array
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        # Process and classify the image
        features = preprocess_image(img)
        prediction = model.predict(features)
        class_label = "Segregated Waste" if prediction[0] == 0 else "Non-Segregated Waste"

        return jsonify({"waste_type": class_label})
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/')
def home():
    return "Waste Classification API is running!"


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))  # Use environment PORT or default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)
