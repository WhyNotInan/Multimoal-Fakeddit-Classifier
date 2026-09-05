import os
import glob
import torch

checkpoint_files = sorted(
    glob.glob("checkpoints/chunk_*.pt"),
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

print(f"Found {len(checkpoint_files)} checkpoint files. Merging...")

full_dataset = []
for file in checkpoint_files:
    print(f"Loading {file}...")
    chunk_data = torch.load(file, weights_only=False)
    full_dataset.extend(chunk_data)

output_path = "fakeddit_features_full.pt"
torch.save(full_dataset, output_path)
print(f"Successfully consolidated {len(full_dataset)} total samples into {output_path}!")