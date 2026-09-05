import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Extracting features using device: {device}")

# Load model
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

df = pd.read_csv("full_clean_dataset.csv")
os.makedirs("checkpoints", exist_ok=True)

BATCH_SIZE = 64
SAVE_INTERVAL = 20000

extracted_features = []
chunk_idx = 1
total_rows = len(df)

print(f"Starting batched offline extraction (Batch size: {BATCH_SIZE})...")

with torch.no_grad():
    for start_idx in tqdm(range(0, total_rows, BATCH_SIZE)):
        batch_df = df.iloc[start_idx : start_idx + BATCH_SIZE]
        
        valid_images = []
        valid_texts = []
        valid_rows = []

        for _, row in batch_df.iterrows():
            try:
                img_path = os.path.join("images", f"{row['id']}.jpg")
                img = Image.open(img_path).convert("RGB")
                valid_images.append(img)
                valid_texts.append(str(row['clean_title']))
                valid_rows.append(row)
            except Exception:
                continue

        if not valid_images:
            continue

        inputs = processor(
            text=valid_texts,
            images=valid_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds
        
        # Calculate Cosine Similarities across the batch
        cos_sims = torch.nn.functional.cosine_similarity(img_embeds, txt_embeds)

        img_np = img_embeds.cpu().numpy()
        txt_np = txt_embeds.cpu().numpy()
        cos_np = cos_sims.cpu().numpy()

        for idx, row in enumerate(valid_rows):
            extracted_features.append({
                "id": row['id'],
                "label_2way": int(row['2_way_label']),
                "label_6way": int(row['6_way_label']),
                "image_vector": img_np[idx],
                "text_vector": txt_np[idx],
                "cosine_similarity": float(cos_np[idx])
            })

        # Save checkpoint periodically
        if len(extracted_features) >= SAVE_INTERVAL:
            checkpoint_path = f"checkpoints/chunk_{chunk_idx}.pt"
            torch.save(extracted_features, checkpoint_path)
            extracted_features = []
            chunk_idx += 1

    # Save remaining features
    if extracted_features:
        torch.save(extracted_features, f"checkpoints/chunk_{chunk_idx}.pt")

print("Offline extraction complete! All chunks are saved in the 'checkpoints' folder.")