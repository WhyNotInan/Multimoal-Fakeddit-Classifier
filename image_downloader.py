import os
import sys
import pandas as pd
import urllib.request
import urllib.error
from tqdm import tqdm
import concurrent.futures

if len(sys.argv) < 2:
    print("Usage: python image_downloader.py <path_to_tsv_file>")
    sys.exit(1)

tsv_file = sys.argv[1]

# 1. Ensure the output directory exists
os.makedirs("images", exist_ok=True)

# 2. Load the dataset
print(f"Loading {tsv_file}...")
df = pd.read_csv(tsv_file, sep="\t")

url_col = "image_url" if "image_url" in df.columns else "url"
id_col = "id"

print(f"Found {len(df)} total rows. Starting multithreaded download...")

# 3. Define the worker function for a single download
def download_image(row):
    img_id = str(row[id_col])
    img_url = str(row[url_col])
    save_path = os.path.join("images", f"{img_id}.jpg")

    # Skip if already downloaded
    if os.path.exists(save_path):
        return "skipped"

    # Skip invalid or missing URLs
    if not img_url or str(img_url).lower() == "nan" or not str(img_url).startswith("http"):
        return "invalid"

    try:
        req = urllib.request.Request(
            img_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=5) as response, open(save_path, "wb") as out_file:
            out_file.write(response.read())
        return "success"
    except Exception:
        return "failed"

# 4. Execute downloads concurrently
# Adjust max_workers up or down depending on your internet connection speed (20-50 is usually safe)
MAX_WORKERS = 30 
results = {"success": 0, "skipped": 0, "invalid": 0, "failed": 0}

# We convert the dataframe rows to a list of dicts for faster processing
rows = df.to_dict('records')

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks to the thread pool
    futures = {executor.submit(download_image, row): row for row in rows}
    
    # Process them as they complete, updating the progress bar
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows), desc="Downloading"):
        status = future.result()
        results[status] += 1

print("\nFinished!")
print(f"Successfully downloaded new images: {results['success']}")
print(f"Already existed (Skipped): {results['skipped']}")
print(f"Dead links / Errors: {results['failed'] + results['invalid']}")
