from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model, scaler, and PCA objects once at startup.
model = joblib.load("waste_classification_xgb.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")


def preprocess_image(image):
    """
    Preprocess the input image by:
    - Resizing to 64x64.
    - Converting to grayscale (if not already).
    - Extracting HOG features.
    - Applying the scaler and PCA transformations.
    """
    # Resize the image to 64x64.
    img_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    # Check if the image is colored. If so, convert to grayscale.
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_resized

    # Extract HOG features with the same parameters used during training.
    features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys')
    features = np.array(features).reshape(1, -1)

    # Apply scaler and PCA transformations.
    features = scaler.transform(features)
    features = pca.transform(features)

    return features


@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is part of the request.
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()

    # Convert image bytes to a NumPy array and decode it (read as a color image).
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        # Preprocess image and extract features.
        features = preprocess_image(img)

        # Get prediction from the model.
        prediction = model.predict(features)
        class_label = "Segregated Waste" if prediction[0] == 0 else "Non-Segregated Waste"

        return jsonify({"waste_type": class_label})
    except Exception as e:
        # Return any errors encountered during processing.
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "Hello World!"


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))  # Use PORT from the environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)
