from convex import ConvexClient
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv(".env.local")

CONVEX_URL = os.getenv("CONVEX_URL")
print(f"Connecting to: {CONVEX_URL}")

if not CONVEX_URL:
    print("Error: CONVEX_URL is missing in .env.local")
else:
    try:
        client = ConvexClient(CONVEX_URL)
        print("Connected! Attempting to save a test prediction...")
        
        id = client.mutation("predictions:savePrediction", {
            "imageUrl": "test_image.png",
            "riskScore": 0.85,
            "result": "High Risk",
            "timestamp": int(time.time() * 1000)
        })
        print(f"Success! Data saved with ID: {id}")
        print("Now check your Convex Dashboard. The table should not be empty.")
    except Exception as e:
        print(f"Failed to save data: {e}")
