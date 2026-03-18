import os
import cv2
import joblib
import numpy as np

# 1. Configuration
DATASET_DIR = 'heart_dataset'
MODEL_PATH = 'heart_model_lite.pkl'

def find_medium_risk():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Run train_model_lite.py first.")
        return

    model = joblib.load(MODEL_PATH)
    
    hog = cv2.HOGDescriptor(_winSize=(128,128),
                            _blockSize=(16,16),
                            _blockStride=(8,8),
                            _cellSize=(8,8),
                            _nbins=9)

    print("Searching for images that the AI thinks are 'Medium Risk' (35%-70%)...")
    
    found_count = 0
    # Check both folders for variety
    for folder in ['Low_Risk', 'High_Risk']:
        path = os.path.join(DATASET_DIR, folder)
        if not os.path.exists(path): continue
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img_resized = cv2.resize(img, (128, 128))
            features = hog.compute(img_resized).flatten().reshape(1, -1)
            
            # Get probability
            prob = model.predict_proba(features)[0][1]
            
            # Medium Risk Range
            if 0.35 <= prob <= 0.70:
                print(f"Found Medium Risk Image: {folder}/{img_name} (Score: {prob*100:.1f}%)")
                found_count += 1
                if found_count >= 5: # Just find 5 examples
                    break
        if found_count >= 5: break

    if found_count == 0:
        print("No exact Medium Risk images found in this small batch. Try uploading a few different ones from High_Risk!")
    else:
        print("\nUse the images listed above for your Medium Risk demo!")

if __name__ == "__main__":
    find_medium_risk()
