import io
import cv2
import numpy as np
import logging
import requests
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from ultralytics import YOLO

import cloudinary
import cloudinary.uploader
import cloudinary.utils

from PIL import Image
import pillow_heif  # For HEIC support

pillow_heif.register_heif_opener()

from config.mongo_connection import db

# ─── Cloudinary Configuration ────────────────────────────────────────
cloudinary.config(
    cloud_name="dxh2mrunr",
    api_key="595799598446626",
    api_secret="lqx5xZKmkBYQFEo2geMo8Ml_wto",
    secure=True,
)

# ─── Blueprint & Mongo Collection ────────────────────────────────────
waste_classification_bp = Blueprint("waste_classification", __name__)
predictions_collection = db["prediction"]

# ─── Load YOLO ONNX Model ────────────────────────────────────────────
yolo = YOLO("best.onnx")  # Adjust model path as needed


def get_public_ip():
    try:
        return requests.get("https://api.ipify.org?format=json").json().get("ip", "Unknown")
    except Exception as e:
        logging.error(f"Error fetching IP: {e}")
        return "Unknown"


def compress_to_bytes(img: np.ndarray) -> io.BytesIO:
    """
    Resize & JPEG-compress an image until it's under 50KB,
    then return it as an in-memory BytesIO.
    """
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    quality = 50
    while True:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Image compression failed")
        data = buf.tobytes()
        if len(data) <= 50 * 1024 or quality <= 10:
            return io.BytesIO(data)
        quality -= 10


@waste_classification_bp.route("/predict", methods=["POST"])
def predict():
    # 1) Validate input
    if "image" not in request.files or "house_id" not in request.form:
        return jsonify({"error": "Image file and house_id are required"}), 400

    house_id = request.form["house_id"]
    subfolder = request.form.get("subfolder", "test")
    upload = request.files["image"]

    # 2) Read raw bytes & convert to OpenCV image
    raw = upload.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)

        arr = np.frombuffer(buf.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 failed to decode image")
    except Exception as e:
        logging.error(f"Failed to load/convert image: {e}")
        return jsonify({"error": "Invalid image format"}), 400

    try:
        # 3) Run ONNX YOLO Detection
        results = yolo.predict(
            source=img,
            imgsz=768,
            conf=0.25,
            verbose=False
        )[0]

        # 4) Build Raw and Filtered Detections
        raw_detections = []
        filtered_detections = []
        for box, conf, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy()
        ):
            det = {
                "class": results.names[int(cls)],
                "confidence": float(conf),
                "box": box.astype(int).tolist()
            }
            raw_detections.append(det)
            if conf >= 0.6:
                filtered_detections.append({
                    "class": det["class"],
                    "confidence": det["confidence"]
                })

        # 5) Build Raw and Filtered Counts
        raw_counts = {}
        filtered_counts = {}
        for det in raw_detections:
            raw_counts[det["class"]] = raw_counts.get(det["class"], 0) + 1
        for det in filtered_detections:
            filtered_counts[det["class"]] = filtered_counts.get(det["class"], 0) + 1

        # 6) Compress & Upload to Cloudinary with Fallback
        try:
            compressed = compress_to_bytes(img)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            public_id = f"Garbage/{subfolder}/{house_id}_{ts}"

            up = cloudinary.uploader.upload(
                compressed,
                public_id=public_id,
                resource_type="image",
                overwrite=True
            )
            url = up.get("secure_url", "Not Uploaded")
            opt_url, _ = cloudinary.utils.cloudinary_url(
                public_id, fetch_format="auto", quality="auto:low"
            )
        except Exception as e:
            logging.error(f"Cloudinary upload failed: {e}")
            url = "Not Uploaded"
            opt_url = "Not Uploaded"

        # 7) Save Raw Data to MongoDB
        record = {
            "house_id": house_id,
            "detections": raw_detections,
            "counts": raw_counts,
            "performance": {
                "pre_ms": round(results.speed["preprocess"], 2),
                "inf_ms": round(results.speed["inference"], 2),
                "post_ms": round(results.speed["postprocess"], 2),
                "shape": [1, 3, 768, 768]
            },
            "model": {
                "name": "best.onnx",
                "imgsz": 768,
                "provider": "CPUExecutionProvider"
            },
            "image_url": url,
            "optimized_url": opt_url,
            "ip_address": get_public_ip(),
            "timestamp": datetime.now()
        }
        ins = predictions_collection.insert_one(record)
        record["_id"] = str(ins.inserted_id)

        # 8) Prepare Filtered API Response
        response = {
            "id": record["_id"],
            "house_id": record["house_id"],
            "timestamp": record["timestamp"].astimezone(timezone.utc).isoformat(),
            "image": {
                "original": record["image_url"],
                "optimized": record["optimized_url"]
            },
            "counts": filtered_counts,
            "detections": filtered_detections
        }

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500
