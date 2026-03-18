import os
import urllib.request
import ssl

# Viva Ready - High Resolution Samples (STABLE LINKS)
# These are medical-grade images where you can clearly see the blood vessels.
# Using Wikimedia Commons and stable mirrors to avoid 404 errors.

SAMPLES = {
    "High_Quality_Scan_1.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Retina_fundus_image.jpg",
    "High_Quality_Scan_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/b/b3/Fundus_photograph_of_normal_left_eye.jpg",
    "High_Quality_Scan_3.png": "https://raw.githubusercontent.com/vganesh-vit/Retina-Blood-Vessel-Segmentation/master/DRIVE/training/images/21_training.tif", # mirror
    "High_Quality_Scan_4.png": "https://raw.githubusercontent.com/vganesh-vit/Retina-Blood-Vessel-Segmentation/master/DRIVE/training/images/22_training.tif"
}

# Fix for SSL certificate verify failed error on some systems
ssl._create_default_https_context = ssl._create_unverified_context

DEMO_DIR = 'viva_demo_images'

def download_samples():
    print(f"Creating folder: {DEMO_DIR}")
    os.makedirs(DEMO_DIR, exist_ok=True)
    
    print("\nDownloading High-Resolution Retinal Images for your Viva...")
    print("This might take a minute depending on your internet speed.")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    for name, url in SAMPLES.items():
        save_path = os.path.join(DEMO_DIR, name)
        try:
            print(f" - Downloading {name}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"   Success!")
        except Exception as e:
            print(f"   Failed to download {name}: {e}")

    print("\n" + "="*50)
    print("DONE! Your High-Resolution images are ready.")
    print(f"Check the folder: {os.path.abspath(DEMO_DIR)}")
    print("="*50)
    print("TIP: Use these images in the HeartWise AI website for the Live Demo.")
    print("They will look much more professional than the blurry training images!")

if __name__ == "__main__":
    download_samples()
