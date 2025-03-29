from flask import Blueprint, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import logging
import xgboost as xgb
import cloudinary
import cloudinary.uploader
from datetime import datetime
import io

# Configure Cloudinary
cloudinary.config(
    cloud_name="dxh2mrunr",
    api_key="649126329594194",
    api_secret="07ntTHEndaWtj9fHDPZMfBWIWoo",  # Replace with your actual API secret
    secure=True
)

waste_classification_bp = Blueprint("waste_classification", __name__)

try:
    # Load XGBoost model from JSON
    model = xgb.Booster()
    model.load_model("waste_classification_xgb.json")

    # Load scaler and PCA
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


def compress_image(image):
    """Compress the image to ensure it's under 50KB."""
    try:
        # Resize image to a smaller size (keeping aspect ratio)
        max_size = (300, 300)  # Adjust based on needs
        image = cv2.resize(image, max_size, interpolation=cv2.INTER_AREA)

        # Convert image to bytes with compression
        is_success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not is_success:
            raise ValueError("Image compression failed.")

        # Check size and further reduce if needed
        image_bytes = buffer.tobytes()
        while len(image_bytes) > 50 * 1024:  # 50KB
            is_success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if not is_success:
                raise ValueError("Image compression failed.")
            image_bytes = buffer.tobytes()

        return io.BytesIO(image_bytes)
    except Exception as e:
        logging.error(f"Error in image compression: {str(e)}")
        raise ValueError("Failed to compress image.")


@waste_classification_bp.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or pca is None:
        return jsonify({"error": "Model or preprocessing objects are not loaded."}), 500

    if 'image' not in request.files or 'house_id' not in request.form:
        return jsonify({"error": "Image and house_id are required"}), 400

    house_id = request.form['house_id']
    file = request.files['image']

    # Convert the image to NumPy array
    image_bytes = file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        # Process the image and get features
        features = preprocess_image(img)
        dmatrix = xgb.DMatrix(features)
        prediction = model.predict(dmatrix)

        # Determine class label
        waste_type = 1 if prediction[0] >= 0.5 else 0
        waste_description = "Non-Segregated Waste" if waste_type == 1 else "Segregated Waste"

        # Compress image to under 50KB
        compressed_image = compress_image(img)

        # Upload compressed image to Cloudinary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{house_id}_{timestamp}.jpg"
        upload_result = cloudinary.uploader.upload(compressed_image, public_id=filename, resource_type="image")

        image_url = upload_result.get("secure_url", "")

        # Optimize delivery further via Cloudinary
        optimized_url, _ = cloudinary.utils.cloudinary_url(filename, fetch_format="auto", quality="auto:low")

        return jsonify({
            "waste_type": waste_type,
            "waste_description": waste_description,
            "image_url": image_url,
            "optimized_image_url": optimized_url
        })
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
