import pickle
import xgboost as xgb

# Load the .pkl file
with open("waste_classification_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Save the model in JSON format
model.save_model("waste_classification_xgb.json")

print("Model successfully converted to JSON format.")
