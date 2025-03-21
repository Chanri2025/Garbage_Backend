from flask import Blueprint, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import logging
import xgboost as xgb

waste_classification_bp = Blueprint("waste_classification", __name__)

try:
    # Load XGBoost model from JSON instead of pkl
    model = xgb.Booster()
    model.load_model("waste_classification_xgb.json")

    # Load scaler and PCA from pkl
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    logging.info("Model (JSON), Scaler, and PCA loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or preprocessing objects: {str(e)}")
    model, scaler, pca = None, None, None

def preprocess_image(image):
    try:
        img_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized

        features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys')
        features = np.array(features).reshape(1, -1)

        features = scaler.transform(features)
        features = pca.transform(features)

        return features
    except Exception as e:
        logging.error(f"Error in image preprocessing: {str(e)}")
        raise ValueError("Failed to preprocess image.")

@waste_classification_bp.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or pca is None:
        return jsonify({"error": "Model or preprocessing objects are not loaded."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()

    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        features = preprocess_image(img)
        dmatrix = xgb.DMatrix(features)  # Convert to DMatrix for XGBoost
        prediction = model.predict(dmatrix)
        class_label = "Segregated Waste" if prediction[0] < 0.5 else "Non-Segregated Waste"

        return jsonify({"waste_type": class_label})
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
