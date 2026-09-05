import os
import pandas as pd

# 1. Load the original dataset
print("Loading the full dataset...")
df = pd.read_csv("multimodal_train.tsv", sep="\t")
initial_count = len(df)

# 2. Handle Missing Text (Drop NaNs)
print("Dropping rows with missing text...")
df = df.dropna(subset=['clean_title'])
text_clean_count = len(df)

# 3. Image Verification & Filtering
print("Verifying existing images on disk...")
df['image_exists'] = df['id'].apply(lambda x: os.path.exists(os.path.join("images", f"{x}.jpg")))
df_clean = df[df['image_exists'] == True].copy()
final_clean_count = len(df_clean)

print("\n--- CLEANING SUMMARY ---")
print(f"Initial rows: {initial_count}")
print(f"Rows after removing blank text: {text_clean_count}")
print(f"Rows with valid downloaded images: {final_clean_count}")

# 4. Save the Full Clean Dataset
output_file = "full_clean_dataset.csv"
print(f"\nSaving all {final_clean_count} clean samples to {output_file}...")
df_clean.to_csv(output_file, index=False)
print("Saved successfully!")