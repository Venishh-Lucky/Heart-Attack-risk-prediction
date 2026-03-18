import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Configuration
DATASET_DIR = 'heart_dataset'
MODEL_PATH = 'heart_model_lite.pkl'

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (128, 128))
    hog = cv2.HOGDescriptor(_winSize=(128,128),
                            _blockSize=(16,16),
                            _blockStride=(8,8),
                            _cellSize=(8,8),
                            _nbins=9)
    return hog.compute(img).flatten()

def verify():
    print("\n" + "="*50)
    print("      HEARTWISE AI - MODEL VERIFICATION")
    print("="*50)
    
    if not os.path.exists(MODEL_PATH):
        print("Error: heart_model_lite.pkl not found!")
        return

    print(f"Loading Model: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    X_test = []
    y_true = []
    
    print("Testing on Retinal Dataset...")
    categories = {'Low_Risk': 0, 'High_Risk': 1}
    for category, label in categories.items():
        path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(path): continue
        
        # Test on a subset for speed in demo, or all images for full proof
        files = os.listdir(path)[:100] # Check first 100 from each for quick demo
        for img_name in files:
            features = extract_features(os.path.join(path, img_name))
            if features is not None:
                X_test.append(features)
                y_true.append(label)

    print(f"Analyzing {len(X_test)} sample images...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "-"*30)
    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("-"*30)
    print("\nReport:")
    print(classification_report(y_true, y_pred, target_names=['Low Risk', 'High Risk']))
    print("="*50 + "\n")

if __name__ == "__main__":
    verify()
