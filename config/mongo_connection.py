# mongo_connection.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    db = client["swm"]
    print("MongoDB connected")
except Exception as e:
    print("MongoDB connection error:", str(e))
    db = None
