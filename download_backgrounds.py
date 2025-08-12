import os
import requests
import random

'''
This script was used to scrape free-to-use backgrounds from unsplash using an
API 
'''
# Access key here
ACCESS_KEY = "REDACTED"
# Query to get images
query = "landscapes" 
# Number of images to obtain
num_images = 200
# Save directory
save_dir = "unsplash_sedans"
os.makedirs(save_dir, exist_ok=True)

# Max allowed images per page by Unsplash API per request
per_page = 30
# Count of downloaded images and count of page number
downloaded = 0
page = 1

# Downloaded images while under the max allowed
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

    # If no more images are found, break
    if "results" not in data or len(data["results"]) == 0:
        print("No more images found.")
        break

    # If there are >= downloaded than the number of images, break
    for item in data["results"]:
        if downloaded >= num_images:
            break

        # Image url, id, and path
        img_url = item["urls"]["regular"]
        img_id = item["id"]
        img_path = os.path.join(save_dir, f"{img_id}.jpg")

        # Image response (to see if download was successful or not)
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(img_response.content)
            print(f"Downloaded {img_path}")
            downloaded += 1
        else:
            print(f"Failed to download image {img_id}")

    # Increment the page count by 1
    page += 1

# After finishing, say  how many downloaded images there are and to which save directory
print(f"Downloaded {downloaded} images to '{save_dir}'")