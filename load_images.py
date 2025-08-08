import os
import requests

save_dir = "road_backgrounds"
os.makedirs(save_dir, exist_ok=True)

for i, item in enumerate(data["images"]):  # adjust key depending on structure
    url = item["image_url"]  # or build from file_name if local
    img_path = os.path.join(save_dir, f"road_{i}.jpg")
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(img_path, "wb") as f:
            f.write(response.content)