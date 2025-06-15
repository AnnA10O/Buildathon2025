import os
import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH = r'/symptom2disease/backend/isic_metadata.csv'
IMAGES_DIR = r'/symptom2disease/backend/isic_images'
os.makedirs(IMAGES_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df_unique = df.drop_duplicates(subset=['isic_id'])

for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Downloading images"):
    isic_id = row['isic_id']
    url = row['image_url']
    out_path = os.path.join(IMAGES_DIR, f"{isic_id}.jpg")
    if not os.path.exists(out_path):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(out_path, 'wb') as f:
                f.write(resp.content)
        except Exception as e:
            print(f"Failed to download {isic_id}: {e}")
