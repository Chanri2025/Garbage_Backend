from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model, scaler, and PCA objects once at startup.
model = joblib.load("waste_classification_xgb.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part.
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400

    file = request.files['image']

    # Read image bytes from the file
    image_bytes = file.read()
    # Convert image bytes to a NumPy array and decode the image using OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess the image to match training conditions:
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    # Extract HOG features using the same parameters used during training.
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys')
    hog_features = np.array(hog_features).reshape(1, -1)

    # Apply the scaler and PCA transformations
    hog_features = scaler.transform(hog_features)
    hog_features = pca.transform(hog_features)

    # Get the prediction from the model
    prediction = model.predict(hog_features)
    class_label = "Segregated Waste" if prediction[0] == 0 else "Non-Segregated Waste"

    # Return the prediction as a JSON response
    return jsonify({"waste_type": class_label})

@app.route('/')
def home():
    return "Hello Word!"


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render
    app.run(host="0.0.0.0", port=port, debug=False)
