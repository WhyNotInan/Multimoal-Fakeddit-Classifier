import os
import pandas as pd

# 1. Load the original dataset
print("Loading the massive dataset...")
df = pd.read_csv("multimodal_train.tsv", sep="\t")
initial_count = len(df)

# 2. Handle Missing Text (Drop NaNs)
print("Dropping rows with missing text...")
df = df.dropna(subset=['clean_title'])
text_clean_count = len(df)

# 3. Image Verification & Filtering
print("Verifying which images actually exist on your D: Drive (This may take a minute)...")
# We check if "images/<id>.jpg" exists for every row
df['image_exists'] = df['id'].apply(lambda x: os.path.exists(os.path.join("images", f"{x}.jpg")))

# Keep only the rows where the image is present
df_clean = df[df['image_exists'] == True].copy()
final_clean_count = len(df_clean)

print(f"\n--- CLEANING SUMMARY ---")
print(f"Initial rows: {initial_count}")
print(f"Rows after removing blank text: {text_clean_count}")
print(f"Rows with successful image downloads: {final_clean_count}")

# 4. Create the Small Prototype Sample
print("\nCreating a 5,000-sample prototype...")
# random_state=42 ensures you get the exact same random 5,000 rows every time you run this
sample_df = df_clean.sample(n=5000, random_state=42)
sample_df.to_csv("prototype_5k.csv", index=False)
print("Saved to prototype_5k.csv!")

# 5. Exploratory Data Analysis (EDA) on the Sample
print("\n--- EDA ON THE 5,000 SAMPLE ---")

# Check Class Balance
print("\n2-Way Label Balance (Real vs Fake):")
print(sample_df['2_way_label'].value_counts(normalize=True) * 100) # Shows percentage

print("\n6-Way Label Balance:")
print(sample_df['6_way_label'].value_counts())

# Check Text Lengths
sample_df['word_count'] = sample_df['clean_title'].apply(lambda x: len(str(x).split()))
print(f"\nAverage word count in titles: {sample_df['word_count'].mean():.1f} words")
print(f"Max word count in a title: {sample_df['word_count'].max()} words")