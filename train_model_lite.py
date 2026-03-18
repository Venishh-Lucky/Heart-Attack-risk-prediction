import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Configuration
DATASET_DIR = 'heart_dataset'
MODEL_SAVE_PATH = 'heart_model_lite.pkl'

def extract_features(img_path):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize to 128x128 for high quality
    img = cv2.resize(img, (128, 128))
    
    # Professional Feature Extraction: HOG (Histogram of Oriented Gradients)
    # This is how medical AI identifies vessels and patterns!
    hog = cv2.HOGDescriptor(_winSize=(128,128),
                            _blockSize=(16,16),
                            _blockStride=(8,8),
                            _cellSize=(8,8),
                            _nbins=9)
    
    features = hog.compute(img)
    return features.flatten()

def train():
    print("Starting Lite Training (using Scikit-Learn)...")
    
    X = []
    y = []

    # Load images from folders
    categories = {'Low_Risk': 0, 'High_Risk': 1}
    for category, label in categories.items():
        folder_path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(folder_path):
            continue
            
        print(f"Loading {category} images...")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("Error: No images found in 'heart_dataset'. Please run prepare_retinamnist.py first.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Use a more powerful SVM classifier which works great with HOG
    print(f"Training on {len(X_train)} samples...")
    # Using probability=True so it works with our API's risk score
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    # Validate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # If SVM is good, we are done. If not, results might be rounded for project demo.
    print(f"Training Complete! Validated Accuracy: {acc*100:.2f}%")

    # Save the model
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
