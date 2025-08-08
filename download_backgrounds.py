import os
import requests
import random

ACCESS_KEY = "REDACTED"
query = "landscapes"  # Change to whatever background type you want
num_images = 200
save_dir = "unsplash_sedans"
os.makedirs(save_dir, exist_ok=True)

per_page = 30  # Max allowed by Unsplash API per request
downloaded = 0
page = 1

while downloaded < num_images:
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "client_id": ACCESS_KEY,
        "order_by": "latest"
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        print("No more images found.")
        break

    for item in data["results"]:
        if downloaded >= num_images:
            break

        img_url = item["urls"]["regular"]
        img_id = item["id"]
        img_path = os.path.join(save_dir, f"{img_id}.jpg")

        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(img_response.content)
            print(f"Downloaded {img_path}")
            downloaded += 1
        else:
            print(f"Failed to download image {img_id}")

    page += 1

print(f"Downloaded {downloaded} images to '{save_dir}'")