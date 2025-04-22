from flask import Flask
from flask_cors import CORS
import logging
from routes.carbon_footprint import carbon_footprint_bp
from routes.waste_classification import waste_classification_bp

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Register Blueprints
app.register_blueprint(carbon_footprint_bp,url_prefix="")
app.register_blueprint(waste_classification_bp)

@app.route("/")
def home():
    return "SWM AI is running!"

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 6000))
    app.run(host="0.0.0.0", port=port, debug=True)
