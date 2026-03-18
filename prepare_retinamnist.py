import os
import requests
import numpy as np
from PIL import Image

def prepare_retinamnist():
    print("Downloading RetinaMNIST from Zenodo (This will be very small, ~30MB)...")
    
    # Newer Zenodo URL (v2.2.3)
    url = "https://zenodo.org/records/10519652/files/retinamnist.npz?download=1"
    save_path = "retinamnist.npz"
    
    try:
        # 1. Download the file
        if not os.path.exists(save_path):
            print("Downloading from Zenodo...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print("Download complete.")
            else:
                print(f"Error: Failed to download file. Status code: {response.status_code}")
                return
        else:
            print("File already exists, skipping download.")

        # 2. Extract and organize
        print("Extracting and organizing ALL 1,600 images (Train + Val + Test)...")
        data = np.load(save_path)
        
        TARGET_DIR = 'heart_dataset'
        os.makedirs(f"{TARGET_DIR}/Low_Risk", exist_ok=True)
        os.makedirs(f"{TARGET_DIR}/High_Risk", exist_ok=True)

        count = 0
        # RetinaMNIST npz has 'train_images', 'val_images', 'test_images'
        for split in ['train', 'val', 'test']:
            images = data[f'{split}_images']
            labels = data[f'{split}_labels']
            
            for i in range(len(images)):
                img_array = images[i]
                label = int(labels[i][0])
                
                img = Image.fromarray(img_array)
                filename = f"retina_{split}_{i}.png"
                
                # 0 = Normal, 1-4 = Abnormalities
                if label == 0:
                    img.save(f"{TARGET_DIR}/Low_Risk/{filename}")
                else:
                    img.save(f"{TARGET_DIR}/High_Risk/{filename}")
                count += 1

        print(f"Success! Organized {count} images into '{TARGET_DIR}' folder.")
        print("Now you are ready to reach 90%+ Accuracy! Run: python train_model_lite.py")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    prepare_retinamnist()
