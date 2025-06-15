import os
import requests
import pandas as pd
from tqdm import tqdm

IMAGES_DIR = "../isic_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

NUM_IMAGES = 500  # Use 500 for testing, increase as needed
PAGE_SIZE = 100
METADATA_URL = "https://api.isic-archive.com/api/v2/images"

def fetch_metadata(num_images):
    all_metadata = []
    page = 1
    with tqdm(total=num_images, desc="Fetching metadata") as pbar:
        while len(all_metadata) < num_images:
            params = {"limit": PAGE_SIZE, "page": page}
            response = requests.get(METADATA_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break
            all_metadata.extend(results)
            pbar.update(len(results))
            if not data.get('next'):
                break
            page += 1
    return all_metadata[:num_images]

def save_metadata_csv(metadata_list, csv_path="isic_metadata.csv"):
    rows = []
    for item in metadata_list:
        meta = item.get('metadata', {})
        clinical = meta.get('clinical', {})
        acquisition = meta.get('acquisition', {})
        files = item.get('files', {})
        row = {
            'isic_id': item.get('isic_id'),
            'image_url': files.get('full', {}).get('url'),
            'image_size': files.get('full', {}).get('size'),
            'pixels_x': acquisition.get('pixels_x'),
            'pixels_y': acquisition.get('pixels_y'),
            'image_type': acquisition.get('image_type'),
            'sex': clinical.get('sex'),
            'anatom_site_general': clinical.get('anatom_site_general'),
            'diagnosis_1': clinical.get('diagnosis_1'),
            'diagnosis_2': clinical.get('diagnosis_2'),
            'diagnosis_3': clinical.get('diagnosis_3'),
            'diagnosis_4': clinical.get('diagnosis_4'),
            'diagnosis_5': clinical.get('diagnosis_5'),
            'melanocytic': clinical.get('melanocytic'),
            'age_approx': clinical.get('age_approx'),
            'public': item.get('public')
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata for {len(df)} images to {csv_path}")

def download_images(metadata_csv="isic_metadata.csv"):
    df = pd.read_csv(metadata_csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        img_id = row['isic_id']
        img_url = row['image_url']
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if pd.notna(img_url) and not os.path.exists(img_path):
            try:
                img_data = requests.get(img_url, timeout=30)
                img_data.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(img_data.content)
            except Exception as e:
                print(f"Failed to download {img_id}: {e}")

if __name__ == "__main__":
    metadata = fetch_metadata(NUM_IMAGES)
    save_metadata_csv(metadata)
    download_images()
    print("Download complete. Images and metadata saved.")
