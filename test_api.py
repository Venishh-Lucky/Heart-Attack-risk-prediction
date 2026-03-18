import requests
import base64

# 1. Test URL Prediction
def test_url_prediction():
    print("Testing URL prediction...")
    url = "http://localhost:8000/predict"
    # Direct image link from Wikimedia (very reliable)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Monarch_Butterfly_In_August.jpg/800px-Monarch_Butterfly_In_August.jpg"
    
    payload = {"image_url": image_url}
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

# 2. Test Base64 Prediction
def test_base64_prediction():
    print("\nTesting Base64 prediction...")
    url = "http://localhost:8000/predict"
    
    # Create a simple transparent 1x1 pixel image as base64
    base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    
    payload = {"base64_data": base64_img}
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    try:
        test_url_prediction()
        test_base64_prediction()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your server is running with 'python main.py' first!")
