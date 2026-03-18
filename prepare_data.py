import os
import shutil
import pandas as pd

# 1. Configuration - UPDATE THESE AFTER DOWNLOADING RFMiD
# Path to the folder where you extracted the Kaggle ZIP
# Example: 'C:/Users/ADMIN/Downloads/retinal-disease-classification'
RFMID_BASE_DIR = 'path/to/extracted/rfmid'
CSV_PATH = os.path.join(RFMID_BASE_DIR, 'Training_Set_Labels.csv')
IMAGES_SRC_DIR = os.path.join(RFMID_BASE_DIR, 'Training_Set', 'Training_Set')

TARGET_DIR = 'heart_dataset'

def organize_rfmid():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please check the paths.")
        return

    # Load labels
    df = pd.read_csv(CSV_PATH)
    
    os.makedirs(f"{TARGET_DIR}/Low_Risk", exist_ok=True)
    os.makedirs(f"{TARGET_DIR}/High_Risk", exist_ok=True)

    print("Organizing images...")
    count = 0
    for index, row in df.iterrows():
        img_id = str(row['ID']) + '.png'
        risk = row['Disease_Risk'] # 0 = Normal, 1 = Disease detected
        
        src = os.path.join(IMAGES_SRC_DIR, img_id)
        if os.path.exists(src):
            if risk == 0:
                shutil.copy(src, f"{TARGET_DIR}/Low_Risk/{img_id}")
            else:
                shutil.copy(src, f"{TARGET_DIR}/High_Risk/{img_id}")
            count += 1

    print(f"Done! Organized {count} images into {TARGET_DIR} folder.")

if __name__ == "__main__":
    organize_rfmid()
