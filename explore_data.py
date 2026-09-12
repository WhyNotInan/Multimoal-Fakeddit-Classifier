import os
import pandas as pd

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
tsv_path = os.path.join(script_dir, "multimodal_train.tsv")

print(f"Loading dataset from: {tsv_path}")
df = pd.read_csv(tsv_path, sep="\t")

print("\n--- DATASET INFO ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- COLUMN NAMES ---")
for col in df.columns:
    print(f"- {col}")

print("\n--- 2-WAY LABEL DISTRIBUTION ---")
print(df['2_way_label'].value_counts())

print("\n--- 6-WAY LABEL DISTRIBUTION ---")
print(df['6_way_label'].value_counts())

print("\n--- SAMPLE POST ---")
sample = df.sample(1).iloc[0]
for index, value in sample.items():
    print(f"{index}: {value}")
