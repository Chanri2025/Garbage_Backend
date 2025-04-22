from flask import Blueprint, request, jsonify
from datetime import datetime
import pytz
from config.mongo_connection import db

carbon_footprint_bp = Blueprint("carbon_footprint", __name__)
carbon_credit_collection = db["carbon_footprint_collections"]


@carbon_footprint_bp.route("/calculate", methods=["POST"])
def calculate_carbon_footprint():
    try:
        data = request.json
        device_id = data.get("device_id")
        house_id = data.get("house_id")
        date_str = data.get("date")
        total_waste = float(data.get("total_waste", 0))

        # Waste breakdown and emissions calculation
        wet_waste = 0.7991 * total_waste
        dry_waste = 0.2009 * total_waste
        methane_eq = 0.07 * wet_waste
        gwp_100_co2_eq = 26 * methane_eq
        dry_waste_c02_eq = 1.45 * dry_waste
        total_co2_eq = gwp_100_co2_eq + dry_waste_c02_eq

        if not date_str:
            return jsonify({"error": "Date is required"}), 400

        # Try parsing with seconds first, fallback to minutes only
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
        except ValueError:
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y %H:%M")
            except ValueError:
                return jsonify({"error": "Date format must be 'DD/MM/YYYY HH:MM[:SS]'"}), 400

        # Calculate carbon footprint in kg
        carbon_footprint = ((wet_waste * 0.7 * 26) + (dry_waste * 1.45)) / 1000

        # Get current timestamp in IST
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist).strftime("%d/%m/%Y %H:%M:%S")

        # Prepare MongoDB document
        doc = {
            "device_id": device_id,
            "house_id": house_id,
            "carbon_footprint": round(carbon_footprint, 2),
            "total_waste": round(total_waste, 2),
            "wet_waste": round(wet_waste, 2),
            "dry_waste": round(dry_waste, 2),
            "methane_eq": round(methane_eq, 2),
            "gwp_100_co2_eq": round(gwp_100_co2_eq, 2),
            "dry_waste_c02_eq": round(dry_waste_c02_eq, 2),
            "total_co2_eq": round(total_co2_eq, 2),
            "date": date_obj.strftime("%d/%m/%Y %H:%M"),
            "timestamp": timestamp
        }

        # Insert into MongoDB
        carbon_credit_collection.insert_one(doc)

        # Remove MongoDB _id field before returning JSON
        doc.pop("_id", None)

        # Return response
        return jsonify({
            "message": "Carbon footprint calculated successfully",
            **doc
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
