from flask import Blueprint, request, jsonify
from datetime import datetime

carbon_footprint_bp = Blueprint("carbon_footprint", __name__)

@carbon_footprint_bp.route("/calculate", methods=["POST"])
def calculate_carbon_footprint():
    try:
        data = request.json
        date_str = data.get("date")  # Expecting date in "DD/MM/YYYY" format
        wet_waste = float(data.get("wet_waste", 0))
        dry_waste = float(data.get("dry_waste", 0))

        if date_str:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        else:
            return jsonify({"error": "Date is required"}), 400

        carbon_footprint = ((wet_waste * 0.7 * 26) + (dry_waste * 1.45)) / 1000

        return jsonify({
            "message": "Carbon footprint calculated successfully",
            "carbon_footprint": round(carbon_footprint, 2)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
